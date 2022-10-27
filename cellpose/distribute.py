import functools, os
import operator
import numpy as np
from dask import delayed
from dask.distributed import wait
import dask.array as da
from ClusterWrap.decorator import cluster
import CircuitSeeker.utility as ut
from cellpose import models
from cellpose.io import logger_setup
import dask_image.ndmeasure._utils._label as label
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.ndimage import generate_binary_structure
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import zarr
from numcodecs import Blosc
import tempfile


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
    temporary_directory=None,
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
    if mask is not None:
        mask_blocksize = np.round(blocksize * mask.shape / zarr_array.shape).astype(int)

    # get all foreground block coordinates
    nblocks = np.ceil(np.array(zarr_array.shape) / blocksize).astype(np.int16)
    block_coords = []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(zarr_array.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        # check foreground
        foreground = True
        if mask is not None:
            start = mask_blocksize * (i, j, k)
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_coords = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_coords]): foreground = False
        if foreground:
            block_coords.append( ((i, j, k), coords) )

    # construct zarr file for output
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    output_zarr_path = temporary_directory.name + '/segmentation_unstitched.zarr'
    output_zarr = ut.create_zarr(
        output_zarr_path,
        zarr_array.shape,
        blocksize,
        np.uint32,
    )

    # pipeline to run on each block
    def preprocess_and_segment(coords):

        # parse inputs and print
        block_index = coords[0]
        coords = coords[1]
        image = zarr_array[coords]
        print('SEGMENTING BLOCK: ', block_index, '\tREGION: ', coords, flush=True)

        # preprocess
        for pp_step in preprocessing_steps:
            image = pp_step[0](image, **pp_step[1])

        # segment
        logger_setup()
        model = models.Cellpose(**model_kwargs)
        segmentation = model.eval(image, **eval_kwargs)[0].astype(np.uint32)

        # remove overlaps
        new_coords = list(coords)
        for axis in range(image.ndim):

            # left side
            if coords[axis].start != 0:
                slc = [slice(None),]*image.ndim
                slc[axis] = slice(overlap, None)
                segmentation = segmentation[tuple(slc)]
                a, b = coords[axis].start, coords[axis].stop
                new_coords[axis] = slice(a + overlap, b)

            # right side
            if segmentation.shape[axis] > blocksize[axis]:
                slc = [slice(None),]*image.ndim
                slc[axis] = slice(None, blocksize[axis])
                segmentation = segmentation[tuple(slc)]
                a = new_coords[axis].start
                new_coords[axis] = slice(a, a + blocksize[axis])

        # get all segment bounding boxes, adjust to global coordinates
        boxes = find_objects(segmentation)
        boxes = [b for b in boxes if b is not None]
        for iii, box in enumerate(boxes):
            boxes[iii] = tuple(slice(a.start+b.start, a.start+b.stop) for a, b in zip(new_coords, box))

        # remap segment ids to globally unique values
        # TODO: casting string to uint32 will overflow witout warning or exception
        #    so, using uint32 this only works if:
        #    number of blocks is less than or equal to 42950
        #    number of segments per block is less than or equal to 99999
        #    if max number of blocks becomes insufficient, could pack values
        #    in binary instead of decimal, then split into two 16 bit chunks
        #    then max number of blocks and max number of cells per block
        #    are both 2**16
        unique, unique_inverse = np.unique(segmentation, return_inverse=True)
        p = str(np.ravel_multi_index(block_index, nblocks))
        remap = [np.uint32(p+str(x).zfill(5)) for x in range(len(unique))]
        remap[0] = np.uint32(0)
        segmentation = np.array(remap)[unique_inverse.reshape(segmentation.shape)]

        # write segmentaiton block
        output_zarr[tuple(new_coords)] = segmentation

        # get the block faces
        faces = []
        for iii in range(3):
            a = [slice(None),] * 3
            a[iii] = slice(0, 1)
            faces.append(segmentation[tuple(a)])
            a = [slice(None),] * 3
            a[iii] = slice(-1, None)
            faces.append(segmentation[tuple(a)])

        # package and return results
        return block_index, (faces, boxes, remap[1:])

    # submit all segmentations, reformat to dict[block_index] = (faces, boxes, box_ids)
    results = cluster.client.gather(cluster.client.map(
        preprocess_and_segment, block_coords,
    ))
    results = {a:b for a, b in results}
    print('REDUCE STEP, SHOULD TAKE 5-10 MINUTES')

    # find face adjacency pairs
    faces, boxes, box_ids = [], [], []
    for block_index, result in results.items():
        for ax, index in enumerate(block_index):
            neighbor_index = tuple(a+1 if i==ax else a for i, a in enumerate(block_index))
            if neighbor_index not in results.keys(): continue
            a = result[0][2*ax + 1]
            b = results[neighbor_index][0][2*ax]
            faces.append( np.concatenate((a, b), axis=ax) )
        boxes += result[1]
        box_ids += result[2]

    # determine mergers
    label_range = np.max(box_ids)
    label_groups = _block_face_adjacency_graph(faces, label_range)
    new_labeling = connected_components(label_groups, directed=False)[1]
    # XXX: new_labeling is returned as int32. Potentially a problem.

    # map unused label ids to zero and remap remaining ids to [0, 1, 2, ...]
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[box_ids] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
    print('REDUCTION COMPLETE')

    # execute mergers and relabeling, write result to disk
    new_labeling_da = da.from_array(new_labeling, chunks=-1)
    segmentation_da = da.from_zarr(output_zarr)
    relabeled = label.relabel_blocks(segmentation_da, new_labeling_da)
    da.to_zarr(relabeled, write_path)

    # merge boxes
    merged_boxes = []
    new_box_ids = new_labeling[box_ids]
    boxes_array = np.empty(len(boxes), dtype=object)
    boxes_array[...] = boxes
    for iii in range(1, len(unique)):
        merge_indices = np.argwhere(new_box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = _merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)

    # return segmentation and boxes
    return zarr.open(write_path, mode='r'), merged_boxes


def _block_face_adjacency_graph(faces, nlabels):
    """
    Shrink labels in face plane, then find which labels touch across the
    face boundary
    """

    all_mappings = []
    structure = generate_binary_structure(3, 1)
    for face in faces:
        # shrink labels in plane
        sl0 = tuple(slice(0, 1) if d==2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d==2 else slice(None) for d in face.shape)
        a = _shrink_labels(face[sl0], 1.0)  # TODO: look at merges, consider increasing threshold
        b = _shrink_labels(face[sl1], 1.0)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))

        # find connections
        mapped = label._across_block_label_grouping(face, structure)
        all_mappings.append(mapped)

    # reformat as csr_matrix and return
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    return coo_matrix((v, (i, j)), shape=(nlabels+1, nlabels+1)).tocsr()


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


def _merge_boxes(boxes):
    """
    Take union of parallelpipeds
    """

    box1 = boxes[0]
    for iii in range(1, len(boxes)):
        union = []
        for s1, s2 in zip(box1, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            union.append(slice(start, stop))
        box1 = tuple(union)
    return box1

