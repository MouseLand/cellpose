import functools
import operator
import numpy as np
import dask.array as da
import dask.delayed as delayed
import ClusterWrap
from cellpose import models
import dask_image.ndmeasure._utils._label as label
from scipy.ndimage import distance_transform_edt
import zarr
from numcodecs import Blosc
from itertools import product


def distributed_eval(
    zarr_path,
    blocksize,
    dataset_path=None,
    mask=None,
    preprocessing_steps=[],
    model_kwargs={},
    eval_kwargs={},
    cluster_kwargs={},
    write_path=None,
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

    # set eval defaults
    if 'diameter' not in eval_kwargs.keys():
        eval_kwargs['diameter'] = 30

    # compute overlap
    overlap = eval_kwargs['diameter'] * 2

    # open zarr file for shape and data type
    metadata = zarr.open(zarr_path, 'r')
    if dataset_path is not None:
        metadata = metadata[dataset_path]
    full_shape = metadata.shape

    # pipeline to run on each block
    def preprocess_and_segment(index, mask):

        # squeeze the index
        index = index.squeeze()

        # skip block if there is no foreground
        if mask is not None:
            ratio = np.array(full_shape) / mask.shape
            xyz = np.round( index / ratio ).astype(int)
            rad = np.round( blocksize / ratio ).astype(int)
            mask_slice = tuple(slice(x, x+r) for x, r in zip(xyz, rad))
            if not np.any(mask[mask_slice]):
                return np.zeros(blocksize, dtype=np.int32)

        # open zarr file
        zarr_file = zarr.open(zarr_path, 'r')
        if dataset_path is not None:
            zarr_file = zarr_file[dataset_path]

        # read data block (with overlaps)
        xyz = [max(0, iii-overlap) for iii in index]
        image = zarr_file[xyz[0]:xyz[0]+blocksize[0]+2*overlap,
                          xyz[1]:xyz[1]+blocksize[1]+2*overlap,
                          xyz[2]:xyz[2]+blocksize[2]+2*overlap]

        # run preprocessing steps
        for pp_step in preprocessing_steps:
            image = pp_step[0](image, **pp_step[1])

        # segment
        model = models.Cellpose(**model_kwargs)
        segmentation = model.eval(image, **eval_kwargs)[0]

        # crop out overlaps
        for axis in range(segmentation.ndim):

            # crop left side
            slc = [slice(None),]*segmentation.ndim
            if index[axis] != 0:
                slc[axis] = slice(overlap, None)
                segmentation = segmentation[tuple(slc)]

            # crop right side
            slc = [slice(None),]*segmentation.ndim
            if segmentation.shape[axis] > blocksize[axis]:
                slc[axis] = slice(None, blocksize[axis])
                segmentation = segmentation[tuple(slc)]

        # return result
        return segmentation.astype(np.int32)

    # start cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # determine block indices, convert to dask array
        slc = tuple(slice(None, x, y) for x, y in zip(full_shape, blocksize))
        block_indices = np.moveaxis(np.array(np.mgrid[slc]), 0, -1)
        d = len(full_shape)
        block_indices_da = da.from_array(block_indices, chunks=(1,)*d + (d,))

        # send mask to all workers
        mask_d = None
        if mask is not None:
            mask_d = cluster.client.scatter(mask, broadcast=True)

        # run segmentation
        segmentation = da.map_blocks(
            preprocess_and_segment, block_indices_da,
            mask=mask_d,
            dtype=np.int32,
            drop_axis=[len(block_indices_da.shape)-1],
            chunks=blocksize,
        )

        # crop back to original shape
        slc = tuple(slice(0, x) for x in full_shape)
        segmentation = segmentation[slc]

        # create container for and iterator over blocks
        updated_blocks = np.empty(segmentation.numblocks, dtype=object)
        block_iter = zip(
            np.ndindex(*segmentation.numblocks),
            map(functools.partial(operator.getitem, segmentation),
                da.core.slices_from_chunks(segmentation.chunks))
        )

        # convert local labels to unique global labels
        index, block = next(block_iter)
        updated_blocks[index] = block
        total = da.max(block)
        for index, block in block_iter:
            local_max = da.max(block)
            block += da.where(block > 0, total, 0)
            updated_blocks[index] = block
            total += local_max

        # put blocks back together as dask array
        updated_blocks = da.block(updated_blocks.tolist())

        # intermediate computation, may smooth out dask errors
        updated_blocks.persist()

        # stitch
        label_groups = _block_face_adjacency_graph(
            updated_blocks, total
        )
        new_labeling = label.connected_components_delayed(label_groups)
        relabeled = label.relabel_blocks(updated_blocks, new_labeling).astype(np.int32)

        # if writing zarr file
        if write_path is not None:
            compressor = Blosc(
                cname='zstd',
                clevel=4,
                shuffle=Blosc.BITSHUFFLE,
            )
            zarr_disk = zarr.open(
                write_path, 'w',
                shape=relabeled.shape,
                chunks=relabeled.chunksize,
                dtype=relabeled.dtype,
                compressor=compressor,
            )
            da.to_zarr(relabeled, zarr_disk)
            return zarr_disk

        # or if user wants numpy array
        else:
            return relabeled.compute()


def _block_face_adjacency_graph(labels, nlabels):
    """
    Shrink labels in face plane, then find which labels touch across the
    face boundary
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
    return mat.astype(np.int32)


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

