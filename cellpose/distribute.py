import functools
import operator
import numpy as np
import dask.array as da
from dask import delayed
from dask.distributed import wait
from ClusterWrap.decorator import cluster
from cellpose import models
from cellpose.io import logger_setup
import dask_image.ndmeasure._utils._label as label
from scipy.ndimage import distance_transform_edt, find_objects
import zarr
from numcodecs import Blosc


@cluster
def distributed_eval(
    zarr_array,
    blocksize,
    write_path,
    mask=None,
    preprocessing_steps=[],
    model_kwargs={},
    eval_kwargs={},
    cluster=None,
    cluster_kwargs={},
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image
    Distributed over cluster or workstation resources with Dask
    Optionally run preprocessing steps on the blocks before running cellpose

    Current Limitations
    -------------------
    Only accepts zarr file inputs. Can be easily generalized to include N5 inputs.
    Other inputs (CZI, stacks of tiffs) would take more work.

    Method for stitching separate segmentations between blocks is pretty simple
    and could be improved.

    Current Dask implementation may not be optimal - Dask struggles with large
    numbers of blocks (greater than 2K). This produces too many tasks and the
    scheduler is sometimes very slow.

    You need to be very knowledgeable about the resources (RAM, cpu count, GPU count,
    GPU RAM) your job will need for this to be successful. These items would be set
    via `cluster_kwargs`

    Parameters
    ----------
    zarr_path : string
        Path to zarr file on disk containing image data

    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    dataset_path : string (default: None)
        If the image data within the zarr file is in a subgroup; the path to
        that subgroup.

    mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. smaller) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. The physical size of the mask and the image data must be the
        same; i.e. the boundaries are the same locations in physical space.

    preprocessing_steps : list of tuples (default: [])
        Preprocessing steps to run on blocks before running cellpose. This might
        include things like Gaussian smoothing or gamma correction. The correct
        formatting is as follows:
            preprocessing_steps = [(pp1, {'arg1':val1}), (pp2, {'arg1':val1}), ...]
        Where `pp1` and `pp2` are functions; the first argument to which should
        be an ndarray (i.e. image data). The second item in each tuple is a
        dictionary containing the arguments to that function.

    model_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose

    eval_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose.eval

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster

    write_path : string (default: None)
        The location of a zarr file on disk where you'd like to write your results
        If `write_path == None` then results are returned to the calling python
        process (e.g. your jupyter process), which for any big data case is going
        to crash. So, for large images (greater than 8GB) be sure to specify this!

    Returns
    -------
    if `write_path != None`, reference to the zarr file on disk containing the
    stitched cellpose results for your entire image

    if `write_path == None`, a numpy.ndarray containing the stitched
    cellpose results for your entire image (may be too large for local RAM!)
    """

    # set default values
    if 'diameter' not in eval_kwargs.keys():
        eval_kwargs['diameter'] = 30
    overlap = eval_kwargs['diameter'] * 2
    blocksize = np.array(blocksize)

    # construct array of block coordinate slices
    nblocks = np.ceil(np.array(zarr_array.shape) / blocksize).astype(np.int16)
    block_coords = np.empty(nblocks, dtype=tuple)
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(zarr_array.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))
        block_coords[i, j, k] = coords

    # construct array of booleans for masking
    block_flags = np.ones(nblocks, dtype=bool)
    if mask is not None:
        mask_blocksize = np.round(blocksize * mask.shape / zarr_array.shape).astype(int)
        for (i, j, k) in np.ndindex(*nblocks):
            start = mask_blocksize * (i, j, k)
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_coords = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_coords]): block_flags[i, j, k] = False

    # convert to dask arrays
    block_coords = da.from_array(block_coords, chunks=(1,)*block_coords.ndim)
    block_flags = da.from_array(block_flags, chunks=(1,)*block_flags.ndim)

    # pipeline to run on each block
    def preprocess_and_segment(coords, flag, block_info=None):

        # skip block if no foreground, otherwise read image data
        if not flag.item():
            return np.zeros(blocksize, dtype=np.uint32)
        coords = coords.item()
        image = zarr_array[coords]

        # preprocess
        for pp_step in preprocessing_steps:
            image = pp_step[0](image, **pp_step[1])

        # segment
        logger_setup()
        model = models.Cellpose(**model_kwargs)
        segmentation = model.eval(image, **eval_kwargs)[0].astype(np.uint32)

        # get all segment bounding boxes, adjust to global coordinates
        boxes = find_objects(segmentation)
        for iii, box in enumerate(boxes):
            boxes[iii] = tuple(slice(a.start+b.start, a.start+b.stop) for a, b in zip(coords, box))

        # remap segment ids to globally unique values
        # TODO: casting string to uint32 will overflow witout warning or exception
        #    so, using uint32 this only works if:
        #    number of blocks is less than or equal to 42950
        #    number of segments per block is less than or equal to 99999
        #    if max number of blocks becomes insufficient, could pack values
        #    in binary instead of decimal, then split into two 16 bit chunks
        #    then max number of blocks and max number of cells per block
        #    are both 2**16
        a = block_info[0]['chunk-location']
        b = block_info[0]['num-chunks']
        p = str(np.ravel_multi_index(a, b))
        remap = [np.uint32(p+str(x).zfill(5)) for x in range(len(boxes)+1)]
        remap[0] = np.uint32(0)
        segmentation = np.array(remap)[segmentation]

        # remove overlaps
        for axis in range(segmentation.ndim):

            # left side
            slc = [slice(None),]*segmentation.ndim
            if coords[axis].start != 0:
                slc[axis] = slice(overlap, None)
                segmentation = segmentation[tuple(slc)]

            # right side
            slc = [slice(None),]*segmentation.ndim
            if segmentation.shape[axis] > blocksize[axis]:
                slc[axis] = slice(None, blocksize[axis])
                segmentation = segmentation[tuple(slc)]

        # package and return results
        result = np.empty((1,)*image.ndim + (3,), dtype=object)
        result[(0,)*image.ndim + (0,)] = segmentation
        result[(0,)*image.ndim + (1,)] = boxes
        result[(0,)*image.ndim + (2,)] = remap[1:]
        return result

    # run segmentation
    results = da.map_blocks(
        preprocess_and_segment,
        block_coords,
        block_flags,
        dtype=object,
        new_axis=[3,],
        chunks=(1,)*zarr_array.ndim + (3,),
    ).persist()
    wait(results)

    # unpack results to seperate dask arrays
    boxes, box_ids = [], []
    segmentation = np.empty(nblocks, dtype=object)
    for (i, j, k) in np.ndindex(*nblocks):

        # references to array, boxes, and box ids
        a = results[i, j, k, 0:1]
        b = results[i, j, k, 1:2]
        c = results[i, j, k, 2:3]

        # create new dask array with correct metadata
        # [0][0] unwraps the arrays created by to_delayed and the return construct
        a = da.from_delayed(a.to_delayed()[0][0], shape=blocksize, dtype=np.uint32)
        segmentation[i, j, k] = a

        # bring the boxes and box ids to local memory
        boxes += b.compute()[0]
        box_ids += c.compute()[0]

    # reassemble segmentation dask array, adjust for assumed map_overlap shape
    segmentation = da.block(segmentation.tolist())
    segmentation = segmentation[tuple(slice(0, s) for s in zarr_array.shape)]

    # determine mergers
    label_range = np.max(box_ids)
    label_groups = _block_face_adjacency_graph(segmentation, label_range)
    new_labeling = label.connected_components_delayed(label_groups).compute()
    # XXX: new_labeling is returned as int32. Potentially a problem.

    # TESTING: check details of new_labeling
    print(new_labeling.shape, new_labeling.dtype, np.max(new_labeling))

    # map unused label ids to zero and remap remaining ids to [0, 1, 2, ...]
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[box_ids] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)

    # TESTING: learn number of cells that were merged
    print(label_range + 1)  # nominal number of labels used
    print(len(boxes))  # number of cells originally found
    print(len(unique), flush=True)  # number of cells after merger

    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]

    # TESTING: make sure new_labeling array is correct
    u2 = np.unique(new_labeling)
    print(new_labeling.shape, new_labeling.dtype, np.max(new_labeling), len(u2), u2.max(), flush=True)

    new_labeling = da.from_array(new_labeling, chunks=-1)

    # TESTING: check dask array details
    print(new_labeling, flush=True)

    # execute mergers and relabeling
    relabeled = label.relabel_blocks(segmentation, new_labeling)

    # TESTING: check relabeled dask array details
    print(relabeled, flush=True)

    # TODO: take union of merged label boxes
    # TODO: eliminate merged box ids

    # compute and store final result, return segmentation and boxes
    da.to_zarr(relabeled, write_path)
    return zarr.open(write_path, 'r+'), boxes


def _block_face_adjacency_graph(labels, nlabels):
    """
    Shrink labels in face plane, then find which labels touch across the
    ace boundary
    """

    # get all boundary faces
    faces = label._chunk_faces(labels.chunks, labels.shape)
    all_mappings = [da.empty((2, 0), dtype=label.LABEL_DTYPE, chunks=1)]

    for face_slice in faces:
        # get boundary region
        face = labels[face_slice]

        # shrink labels in plane
        sl0 = tuple(slice(0, 1) if d==2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d==2 else slice(None) for d in face.shape)
        a = _shrink_labels_delayed(face[sl0], 1.0)
        b = _shrink_labels_delayed(face[sl1], 1.0)
        face = da.concatenate((a, b), axis=np.argmin(a.shape))

        # connectivity structure normal to face plane
        structure = np.zeros((3,)*face.ndim, dtype=bool)
        sl = tuple(slice(None) if d==2 else slice(1, 2) for d in face.shape)
        structure[sl] = True  # the line above will fail for really tiny chunks

        # find connections
        mapped = label._across_block_label_grouping_delayed(face, structure)
        all_mappings.append(mapped)

    # reorganize as csr_matrix
    all_mappings = da.concatenate(all_mappings, axis=1)
    i, j = all_mappings
    mat = label._to_csr_matrix(i, j, nlabels + 1)

    # return matrix
    return mat


def _shrink_labels(plane, threshold):
    """
    Shrink labels in plane by some distance from their boundary
    """

    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def _shrink_labels_delayed(plane, threshold):
    """
    Delayed version of `_shrink_labels`
    """

    _shrink_labels_d = delayed(_shrink_labels)
    shrunk_labels = _shrink_labels_d(plane, threshold)
    return da.from_delayed(
        shrunk_labels, shape=plane.shape, dtype=plane.dtype,
    )

