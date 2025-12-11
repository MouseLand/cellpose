# stdlib imports
import logging, os, tempfile, time, traceback

# non-stdlib core dependencies
import numpy as np
import scipy
import tifffile
import torch


# distributed dependencies
import dask_image.ndmeasure as di_ndmeasure
import dask_jobqueue
import yaml
import zarr


from cellpose import transforms
from cellpose.models import assign_device, CellposeModel
from cellpose.contrib.dask_utils import cluster
from cellpose.contrib.block_utils import (get_block_crops, get_nblocks, prepare_blocksize,
                                          prepare_overlaps, remove_overlaps)

from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import as_completed
from typing import List


logger = logging.getLogger(__name__)


######################## File format functions ################################
def numpy_array_to_zarr(write_path, array, chunks):
    """
    Store an in memory numpy array to disk as a chunked Zarr array

    Parameters
    ----------
    write_path : string
        Filepath where Zarr array will be created

    array : numpy.ndarray
        The already loaded in-memory numpy array to store as zarr

    chunks : tuple, must be array.ndim length
        How the array will be chunked in the Zarr array

    Returns
    -------
    zarr.core.Array
        A read+write reference to the zarr array on disk
    """

    zarr_array = zarr.open(
        write_path,
        mode='w',
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
    )
    zarr_array[...] = array
    return zarr_array


def wrap_folder_of_tiffs(
    filename_pattern,
    block_index_pattern=r'_(Z)(\d+)(Y)(\d+)(X)(\d+)',
):
    """
    Wrap a folder of tiff files with a zarr array without duplicating data.
    Tiff files must all contain images with the same shape and data type.
    Tiff file names must contain a pattern indicating where individual files
    lie in the block grid.

    Distributed computing requires parallel access to small regions of your
    image from different processes. This is best accomplished with chunked
    file formats like Zarr and N5. This function can accommodate a folder of
    tiff files, but it is not equivalent to reformating your data as Zarr or N5.
    If your individual tiff files/tiles are huge, distributed performance will
    be poor or not work at all.

    It does not make sense to use this function if you have only one tiff file.
    That tiff file will become the only chunk in the zarr array, which means all
    workers will have to load the entire image to fetch their crop of data anyway.
    If you have a single tiff image, you should just reformat it with the
    numpy_array_to_zarr function. Single tiff files too large to fit into system
    memory are not be supported.

    Parameters
    ----------
    filename_pattern : string
        A glob pattern that will match all needed tif files

    block_index_pattern : regular expression string (default: r'_(Z)(\d+)(Y)(\d+)(X)(\d+)')
        A regular expression pattern that indicates how to parse tiff filenames
        to determine where each tiff file lies in the overall block grid
        The default pattern assumes filenames like the following:
            {any_prefix}_Z000Y000X000{any_suffix}
            {any_prefix}_Z000Y000X001{any_suffix}
            ... and so on

    Returns
    -------
    zarr.core.Array
    """

    # define function to read individual files
    def imread(fname):
        with open(fname, 'rb') as fh:
            return imagecodecs.tiff_decode(fh.read(), index=None)

    # create zarr store, open it as zarr array and return
    store = tifffile.imread(
        filename_pattern,
        aszarr=True,
        imread=imread,
        pattern=block_index_pattern,
        axestiled={x:x for x in range(3)},
    )
    return zarr.open(store=store)


######################## Cluster related functions ############################

#----------------------- config stuff ----------------------------------------#
DEFAULT_CONFIG_FILENAME = 'distributed_cellpose_dask_config.yaml'

def _config_path(config_name):
    """Add config directory path to config filename"""
    return str(pathlib.Path.home()) + '/.config/dask/' + config_name


def _modify_dask_config(
    config,
    config_name=DEFAULT_CONFIG_FILENAME,
):
    """
    Modifies dask config dictionary, but also dumps modified
    config to disk as a yaml file in ~/.config/dask/. This
    ensures that workers inherit config options.
    """
    dask.config.set(config)
    with open(_config_path(config_name), 'w') as f:
        yaml.dump(dask.config.config, f, default_flow_style=False)


def _remove_config_file(
    config_name=DEFAULT_CONFIG_FILENAME,
):
    """Removes a config file from disk"""
    config_path = _config_path(config_name)
    if os.path.exists(config_path):
        os.remove(config_path)


#----------------------- clusters --------------------------------------------#
class myLocalCluster(distributed.LocalCluster):
    """
    This is a thin wrapper extending dask.distributed.LocalCluster to set
    configs before the cluster or workers are initialized.

    For a list of full arguments (how to specify your worker resources) see:
    https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster
    You need to know how many cpu cores and how much RAM your machine has.

    Most users will only need to specify:
    n_workers
    ncpus (number of physical cpu cores per worker)
    memory_limit (which is the limit per worker, should be a string like '16GB')
    threads_per_worker (for most workflows this should be 1)

    You can also modify any dask configuration option through the
    config argument.

    If your workstation has a GPU, one of the workers will have exclusive
    access to it by default. That worker will be much faster than the others.
    You may want to consider creating only one worker (which will have access
    to the GPU) and letting that worker process all blocks serially.
    """

    def __init__(
        self,
        ncpus,
        config={},
        config_name=DEFAULT_CONFIG_FILENAME,
        persist_config=False,
        **kwargs,
    ):
        # config
        self.config_name = config_name
        self.persist_config = persist_config
        scratch_dir = f"{os.getcwd()}/"
        scratch_dir += f".{getpass.getuser()}_distributed_cellpose/"
        config_defaults = {'temporary-directory':scratch_dir}
        config = {**config_defaults, **config}
        _modify_dask_config(config, config_name)

        # construct
        if "host" not in kwargs:
            kwargs["host"] = ""
        super().__init__(**kwargs)
        self.client = distributed.Client(self)

        # set environment variables for workers (threading)
        environment_vars = {
            'MKL_NUM_THREADS':str(ncpus),
            'NUM_MKL_THREADS':str(ncpus),
            'OPENBLAS_NUM_THREADS':str(ncpus),
            'OPENMP_NUM_THREADS':str(ncpus),
            'OMP_NUM_THREADS':str(ncpus),
        }
        def set_environment_vars():
            for k, v in environment_vars.items():
                os.environ[k] = v
        self.client.run(set_environment_vars)

        print("Cluster dashboard link: ", self.dashboard_link)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.persist_config:
            _remove_config_file(self.config_name)
        self.client.close()
        super().__exit__(exc_type, exc_value, traceback)


class janeliaLSFCluster(dask_jobqueue.LSFCluster):
    """
    This is a thin wrapper extending dask_jobqueue.LSFCluster,
    which in turn extends dask.distributed.SpecCluster. This wrapper
    sets configs before the cluster or workers are initialized. This is
    an adaptive cluster and will scale the number of workers, between user
    specified limits, based on the number of pending tasks. This wrapper
    also enforces conventions specific to the Janelia LSF cluster.

    For a full list of arguments see
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html

    Most users will only need to specify:
    ncpus (the number of cpu cores per worker)
    min_workers
    max_workers
    """

    def __init__(
        self,
        ncpus,
        min_workers,
        max_workers,
        config={},
        config_name=DEFAULT_CONFIG_FILENAME,
        persist_config=False,
        **kwargs
    ):

        # store all args in case needed later
        self.locals_store = {**locals()}

        # config
        self.config_name = config_name
        self.persist_config = persist_config
        scratch_dir = f"/scratch/{getpass.getuser()}/"
        config_defaults = {
            'temporary-directory':scratch_dir,
            'distributed.comm.timeouts.connect':'180s',
            'distributed.comm.timeouts.tcp':'360s',
        }
        config = {**config_defaults, **config}
        _modify_dask_config(config, config_name)

        # threading is best in low level libraries
        job_script_prologue = [
            f"export MKL_NUM_THREADS={2*ncpus}",
            f"export NUM_MKL_THREADS={2*ncpus}",
            f"export OPENBLAS_NUM_THREADS={2*ncpus}",
            f"export OPENMP_NUM_THREADS={2*ncpus}",
            f"export OMP_NUM_THREADS={2*ncpus}",
        ]

        # set scratch and log directories
        if "local_directory" not in kwargs:
            kwargs["local_directory"] = scratch_dir
        if "log_directory" not in kwargs:
            log_dir = f"{os.getcwd()}/dask_worker_logs_{os.getpid()}/"
            pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)
            kwargs["log_directory"] = log_dir

        # graceful exit for lsf jobs (adds -d flag)
        class quietLSFJob(dask_jobqueue.lsf.LSFJob):
            cancel_command = "bkill -d"

        # construct
        super().__init__(
            ncpus=ncpus,
            processes=1,
            cores=1,
            memory=str(15*ncpus)+'GB',
            mem=int(15e9*ncpus),
            job_script_prologue=job_script_prologue,
            job_cls=quietLSFJob,
            **kwargs,
        )
        self.client = distributed.Client(self)
        print("Cluster dashboard link: ", self.dashboard_link)

        # set adaptive cluster bounds
        self.adapt_cluster(min_workers, max_workers)


    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.persist_config:
            _remove_config_file(self.config_name)
        self.client.close()
        super().__exit__(exc_type, exc_value, traceback)


    def adapt_cluster(self, min_workers, max_workers):
        _ = self.adapt(
            minimum_jobs=min_workers,
            maximum_jobs=max_workers,
            interval='10s',
            wait_count=6,
        )


    def change_worker_attributes(
        self,
        min_workers,
        max_workers,
        **kwargs,
    ):
        """WARNING: this function is dangerous if you don't know what
           you're doing. Don't call this unless you know exactly what
           this does."""
        self.scale(0)
        for k, v in kwargs.items():
            self.new_spec['options'][k] = v
        self.adapt_cluster(min_workers, max_workers)


#----------------------- decorator -------------------------------------------#
def cluster(func):
    """
    This decorator ensures a function will run inside a cluster
    as a context manager. The decorated function, "func", must
    accept "cluster" and "cluster_kwargs" as parameters. If
    "cluster" is not None then the user has provided an existing
    cluster and we just run func. If "cluster" is None then
    "cluster_kwargs" are used to construct a new cluster, and
    the function is run inside that cluster context.
    """
    @functools.wraps(func)
    def create_or_pass_cluster(*args, **kwargs):
        # TODO: this only checks if args are explicitly present in function call
        #       it does not check if they are set correctly in any way
        assert 'cluster' in kwargs or 'cluster_kwargs' in kwargs, \
        "Either cluster or cluster_kwargs must be defined"
        if not 'cluster' in kwargs:
            cluster_constructor = myLocalCluster
            F = lambda x: x in kwargs['cluster_kwargs']
            if F('ncpus') and F('min_workers') and F('max_workers'):
                cluster_constructor = janeliaLSFCluster
            with cluster_constructor(**kwargs['cluster_kwargs']) as cluster:
                kwargs['cluster'] = cluster
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return create_or_pass_cluster


#----------------------- The main function -----------------------------------#
@cluster
def distributed_eval(
    input_zarr: zarr.Array,
    input_timeindex: int|None,
    input_channels: List[int]|None,
    blocksize,
    output_zarr,
    mask=None,
    preprocessing_steps=[],
    cellpose_model_args={},
    normalize_args={},
    cellpose_eval_args={},
    label_dist_th=1.0,
    cluster=None,
    cluster_kwargs={},
    skip_merge=False,
    temp_dir=None,
):
    distributed_eval_results = run_distributed_eval(
        input_zarr,
        input_timeindex,
        input_channels,
        blocksize,
        output_zarr,
        cluster.client,
        blockoverlaps=(),
        mask=mask,
        preprocessing_steps=preprocessing_steps,
        cellpose_model_args=cellpose_model_args,
        normalize_args=normalize_args,
        cellpose_eval_args=cellpose_eval_args,
        label_dist_th=label_dist_th,
    )
    seg_blocks_zarr, seg_blocks, seg_block_faces, seg_boxes, seg_box_ids = distributed_eval_results

    if skip_merge:
        return seg_blocks_zarr, seg_boxes

    # resize the cluster before merging labels - the GPU workers no longer needed
    if isinstance(cluster, dask_jobqueue.core.JobQueueCluster): 
        cluster.change_worker_attributes(
            min_workers=cluster.locals_store['min_workers'],
            max_workers=cluster.locals_store['max_workers'],
            ncpus=1,
            memory="15GB",
            mem=int(15e9),
            queue=None,
            job_extra_directives=[],
        )

    return merge_labels(
        seg_blocks_zarr,
        seg_blocks,
        seg_block_faces,
        seg_boxes,
        seg_box_ids,
        output_zarr,
        cluster.client,
        label_dist_th=label_dist_th,
        temp_dir=temp_dir,
    )


def run_distributed_eval(
    input_zarr: zarr.Array,
    input_timeindex: int|None,
    input_channels: int|List[int]|None,
    blocksize,
    labels_zarr,
    dask_client,
    blockoverlaps=(),
    mask=None,
    preprocessing_steps=[],
    cellpose_model_args={},
    normalize_args={},
    cellpose_eval_args={},
    label_dist_th=1.0,
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask to ignore background regions in image.

    The dask client must be present but it can be either a remote client that references
    a Dask Scheduler's IP or a local client.

    Parameters
    ----------
    input_zarr : zarr.Array
        Image data as a zarr array

    timeindex : string
        if the image is a 5-D TCZYX ndarray specify which timeindex to use

    input_channels : sequence[int] | None
        channels used for segmentation. If not set, it uses all channels
                
    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    dask_client : dask.distributed.Client
        A remote or locakl dask client.

        mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    Returns
    -------
    Two values are returned:
    (1) A reference to a dask array containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """
    image_shape = input_zarr.shape
    logger.info((
        f'3D: {cellpose_eval_args.get("do_3D")}, '
        f'shape: {image_shape}, '
        f'process blocks: {blocksize} with {blockoverlaps} overlaps '
        f'timeindex: {input_timeindex} '
        f'image channels {input_channels} '
    ))
    diameter = cellpose_eval_args.get('diameter')
    blocksize = prepare_blocksize(image_shape, blocksize)
    blockoverlaps = prepare_overlaps(image_shape, blocksize, blockoverlaps,
                                     default_overlap=2 * diameter if diameter is not None else None)
    block_indices, block_crops = get_block_crops(
        image_shape, blocksize, blockoverlaps, mask,
    )

    logger.info((
        f'Start segmenting: {len(block_indices)} {blocksize} blocks '
        f'with overlap {blockoverlaps} '
        f'from a {image_shape} image'
    ))

    if isinstance(input_channels, int):
        # if the input channel is a single int value make it a list
        segmentation_input_channels = [input_channels]
    else:
        segmentation_input_channels = input_channels
    futures = dask_client.map(
        _process_block,
        block_indices,
        block_crops,
        input_zarr=input_zarr,
        input_timeindex=input_timeindex,
        input_channels=segmentation_input_channels,
        blocksize=blocksize,
        blockoverlaps=blockoverlaps,
        labels_zarr=labels_zarr,
        preprocessing_steps=preprocessing_steps,
        cellpose_model_args=cellpose_model_args,
        normalize_args=normalize_args,
        cellpose_eval_args=cellpose_eval_args,
    )

    results = dask_client.gather(futures)
    logger.info((
        f'Finished segmenting: {len(block_indices)} {blocksize} blocks '
        f'with overlap {blockoverlaps}'
        ' - start label merge process'
    ))

    label_block_indices, faces, boxes_, per_block_box_ids = list(zip(*results))
    logger.info((
        'Segmentation results contain '
        f'faces: {len(faces)}, boxes: {len(boxes_)}, box_ids: {len(per_block_box_ids)}'
    ))

    boxes = [box for sublist in boxes_ for box in sublist]
    box_ids = np.concatenate(per_block_box_ids).astype(np.uint32)

    return labels_zarr, label_block_indices, faces, boxes, box_ids


def merge_labels(segmented_blocks_zarr:zarr.Array,
                 segmented_block_indices,
                 segmented_block_faces,
                 segmented_boxes,
                 segmented_box_ids,
                 output_labels_zarr:zarr.Array,
                 dask_client,
                 label_dist_th=1.0,
                 temp_dir=None):
    logger.info((
        f'Relabel {segmented_box_ids.shape} blocks of type {segmented_box_ids.dtype} - '
        f'use {len(segmented_block_faces)} faces for merging labels'
    ))
    final_labeling = _determine_merge_relabeling(
        segmented_block_indices,
        segmented_block_faces,
        segmented_box_ids,
        label_dist_th=label_dist_th
    )
    with tempfile.TemporaryDirectory(
        prefix='.',
        suffix='distributed_cellpose_tempdir',
        dir=temp_dir or os.getcwd()) as temporary_directory:
        # save the relabeling so that we only pass the file name to the workers
        final_labeling_path = f'{temporary_directory}/final_labeling.npy'
        _write_new_labeling(final_labeling_path, final_labeling)

        label_slices = slices_from_chunks(
            normalize_chunks(output_labels_zarr.chunks, shape=output_labels_zarr.shape)
        )
        relabel_futures = dask_client.map(
            _copy_relabeled_block,
            label_slices,
            final_labeling=final_labeling_path,
            src_data=segmented_blocks_zarr,
            dest_data=output_labels_zarr,
        )
        # wait for relabeling results
        relabel_res = True
        for f, r in as_completed(relabel_futures, with_results=True):
            if f.cancelled():
                exc = f.exception()
                logger.exception(f'Block relabel exception: {exc}')
                relabel_res = False
            else:
                relabel_res = relabel_res and r
        logger.info(f'Completed relabeling all blocks: {relabel_res}')

        logger.info(f'Relabel {segmented_box_ids.shape} blocks from {final_labeling_path}')

        merged_boxes = _merge_all_boxes(segmented_boxes, final_labeling[segmented_box_ids.astype(np.int32)])
        return output_labels_zarr, merged_boxes


######################## the function to run on each block ####################
def _process_block(
    block_index,
    crop,
    input_zarr,
    input_timeindex,
    input_channels,
    blocksize,
    blockoverlaps,
    labels_zarr,
    preprocessing_steps=[],
    cellpose_model_args={},
    normalize_args={},
    cellpose_eval_args={},
):
    """
    Preprocess and segment one block, of many, with eventual merger
    of all blocks in mind. The block is processed as follows:

    (1) Read block from disk, preprocess, and segment.
    (2) Remove overlaps.
    (3) Get bounding boxes for every segment.
    (4) Remap segment IDs to globally unique values.
    (5) Write segments to disk.
    (6) Get segmented block faces.

    (5) return remapped segments as a numpy array, boxes, and box_ids

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid

    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    image_container_path : string
        Path to image container.

    image_subpath : string
        Dataset path relative to image container.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image block before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    blocksize : iterable (list, tuple, np.ndarray)
        The number of voxels (the shape) of blocks without overlaps

    blocksoverlap : iterable (list, tuple, np.ndarray)
        The number of voxels added to the blocksize to provide context
        at the edges

    labels_output_zarr : zarr.core.Array
        A location where segments can be stored temporarily before
        merger is complete

    Returns
    -------
    faces : a list of numpy arrays - the faces of the block segments
    boxes : a list of crops (tuples of slices), bounding boxes of segments
    box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                boxes
    """
    logger.info((
        f'RUNNING BLOCK: {block_index}, '
        f'REGION: {crop}, '
        f'blocksize: {blocksize}, '
        f'blocksoverlap: {blockoverlaps}, '
        f'cellpose eval opts: {cellpose_eval_args}, '
        f'cellpose model opts: {cellpose_model_args}, '
    ))
    segmentation = read_preprocess_and_segment(
        input_zarr,
        input_timeindex,
        input_channels,
        crop, 
        preprocessing_steps=preprocessing_steps,
        cellpose_model_args=cellpose_model_args,
        normalize_args=normalize_args,
        cellpose_eval_args=cellpose_eval_args,
    )
    seg_ndim = segmentation.ndim
    # labels are single channel so if the input was multichannel remove the channel coords
    labels_shape = labels_zarr.shape[-seg_ndim:]
    labels_block_index = block_index[-seg_ndim:]
    labels_coords = crop[-seg_ndim:]
    labels_overlaps = blockoverlaps[-seg_ndim:]
    labels_blocksize = blocksize[-seg_ndim:]

    logger.debug((
        f'adjusted labels image shape to {labels_shape} '
        f'labels block index to {labels_block_index} '
        f'labels block coords to {labels_coords} '
        f'labels block overlaps to {labels_overlaps} '
        f'labels block size to {labels_blocksize} '
    ))
    logger.info(f'Remove {labels_overlaps} overlaps from {segmentation.shape} labels')
    segmentation, labels_coords = remove_overlaps(segmentation, labels_coords, labels_overlaps, labels_blocksize)

    boxes = _bounding_boxes_in_global_coordinates(segmentation, labels_coords)
    nblocks = get_nblocks(labels_shape, labels_blocksize)
    segmentation, remap = _global_segment_ids(segmentation, labels_block_index, nblocks)
    if remap[0] == 0:
        remap = remap[1:]
    if labels_zarr.ndim != seg_ndim:
        labels_zarr_coords = (0,)*(labels_zarr.ndim-seg_ndim)+tuple(labels_coords)
        labels_zarr[labels_zarr_coords] = segmentation
    else:
        labels_zarr[tuple(labels_coords)] = segmentation
    faces = _block_faces(segmentation)
    return labels_block_index, faces, boxes, remap


#----------------------- component functions ---------------------------------#
def read_preprocess_and_segment(
    input_zarr,
    input_timeindex,
    input_channels,
    crop,
    preprocessing_steps=[],
    cellpose_model_args={},
    normalize_args={},
    cellpose_eval_args={},
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""

    input_channel_axis = cellpose_eval_args.get('channel_axis')

    block_coords_list = [c for c in crop]
    if input_channel_axis is not None and input_channels:
        block_coords_list[input_channel_axis] = input_channels

    if input_timeindex is not None:
        # this should only be set for OME images if timepoints are present
        # and timepoints if present are the first dimension
        block_coords_list[0] = input_timeindex

    block_coords = tuple(block_coords_list)
    logger.info((
        f'Reading {block_coords} block from the input zarr '
        f'based on the input crop: {crop} '
        f'timeindex {input_timeindex} '
        f'channels {input_channels} '
        f'input channel axis {input_channel_axis} '
    ))

    image_block = input_zarr[block_coords]
    block_shape = image_block.shape
    block_ndim = image_block.ndim

    do_3D = cellpose_eval_args.get('do_3D', False)
    input_z_axis = cellpose_eval_args.get('z_axis')
    spatial_dims = 3 if do_3D else 2

    if input_z_axis is not None:
        # z axis is specified
        if input_timeindex is not None and input_z_axis > 0:
            z_axis = input_z_axis - 1
        else:
            z_axis = input_z_axis
    else:
        # z_axis is not specified
        if not do_3D:
            z_axis = None
        else:
            if block_ndim >= spatial_dims:
                z_axis = -3
            else:
                raise ValueError(f'Cannot handle {spatial_dims}-D segmentation for block of shape {block_shape}')

    if input_channel_axis is not None:
        # channel axis is specified
        if input_timeindex is not None and input_channel_axis > 0:
            channel_axis = input_channel_axis - 1
        else:
            channel_axis = input_channel_axis
    else:
        # channel axis is not specified
        if block_ndim == spatial_dims:
            # append a dimension for the channel if channel dimension is missing
            new_block_shape = (1,) + (block_shape)
            image_block = np.reshape(image_block, new_block_shape)
            channel_axis = 0 # channel is the first dimension
        else:
            # assume the channel axis is before the spatial axes
            channel_axis = block_ndim - spatial_dims - 1
    cellpose_eval_args['channel_axis'] = channel_axis
    cellpose_eval_args['z_axis'] = z_axis

    start_time = time.time()

    for pp_step in preprocessing_steps:
        logger.debug(f'Apply preprocessing step: {pp_step}')
        image_block = pp_step[0](image_block, **pp_step[1])

    model = _get_segmentation_model(cellpose_model_args)

    if normalize_args.get('normalize'):
        logger.info(f'Normalize {image_block.shape} block at {crop} params: {normalize_args}')
        image_block = transforms.normalize_img(image_block, axis=channel_axis,
                                               **normalize_args)
    logger.info(f'Eval {image_block.shape} block at {crop} args: {cellpose_eval_args}')
    try:
        labels = model.eval(image_block, **cellpose_eval_args)[0].astype(np.uint32)
    except Exception as e:
        logger.error((
            f'ERROR eval {image_block.shape} block at {crop} args: {cellpose_eval_args} '
            f'err={e} {traceback.format_exception(e)}'
        ))
        raise e

    end_time = time.time()
    unique_labels = np.unique(labels)
    logged_block_message = (f'for block: {crop}' 
                            if crop is not None
                            else 'for entire image')
    logger.info((
        'Finished model eval '
        f'{logged_block_message} => '
        f'found {len(unique_labels)} unique labels '
        f'in the {labels.shape} image '
        f'in {end_time-start_time}s '
    ))
    return labels


def _get_segmentation_model(cellpose_model_args):
    use_gpu = cellpose_model_args.get('use_gpu', True)
    gpu_device = cellpose_model_args.get('gpu_device', 0)
    if use_gpu:
        available_gpus = torch.cuda.device_count()
        logger.info(f'Found {available_gpus} GPUs')
        if available_gpus > 1:
            # if multiple gpus are available try to find one that can be used
            segmentation_device, gpu = None, False
            for gpui in range(available_gpus):
                try:
                    logger.debug(f'Try GPU: {gpui}')
                    segmentation_device, gpu = assign_device(gpu=use_gpu, device=gpui)
                    logger.debug(f'Result for GPU: {gpui} => {segmentation_device}:{gpu}')
                    if gpu:
                        break
                    # because of a bug in cellpose trying the other devices explicitly here
                    torch.cuda.set_device(gpui)
                    segmentation_device = torch.device(f'cuda:{gpui}')
                    logger.info(f'Device {segmentation_device} present and usable')
                    _ = torch.zeros((1,1)).to(segmentation_device)
                    logger.info(f'Device {segmentation_device} tested and it is usable')
                    gpu = True
                    break
                except Exception as e:
                    logger.warning(f'cuda:{gpui} present but not usable: {e}')
        else:
            segmentation_device, gpu = assign_device(gpu=use_gpu, device=gpu_device)
    else:
        segmentation_device, gpu = assign_device(gpu=use_gpu, device=gpu_device)

    return CellposeModel(
        gpu=gpu,
        device=segmentation_device,
        pretrained_model=cellpose_model_args.get('pretrained_model'),
    )


def _bounding_boxes_in_global_coordinates(segmentation, crop):
    """
    bounding boxes (tuples of slices) are super useful later
    best to compute them now while things are distributed
    """
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]

    def _translate(a, b):
        return slice(a.start+b.start, a.start+b.stop)

    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(_translate(a, b) for a, b in zip(crop, box))
    return boxes


def _global_segment_ids(segmentation, block_index, nblocks):
    """
    Pack the block index into the segment IDs so they are
    globally unique. Everything gets remapped to [1..N] later.
    A label is split into 5 digits on left and 5 digits on right.
    This creates limits: 42950 maximum number of blocks and
    99999 maximum number of segments per block
    """
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    logger.debug((
        f'Block {block_index} out of {nblocks} blocks '
        f'- has {len(unique)} unique labels '
    ))
    p = str(np.ravel_multi_index(block_index, nblocks))
    remap = [int(p+str(x).zfill(5)) for x in unique]
    if unique[0] == 0:
        remap[0] = 0  # 0 should just always be 0
    logger.debug(f'Remap: {remap}')
    segmentation = np.array(remap, dtype=np.uint32)[unique_inverse.reshape(segmentation.shape)]
    return segmentation, remap


def _block_faces(segmentation):
    """Slice faces along every axis"""
    faces = []
    for iii in range(segmentation.ndim):
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces


def _determine_merge_relabeling(block_indices, faces, labels,
                               label_dist_th=1.0):
    """Determine boundary segment mergers, remap all label IDs to merge
       and put all label IDs in range [1..N] for N global segments found"""
    faces = _adjacent_faces(block_indices, faces)
    logger.debug(f'Determine relabeling for {labels.shape} of type {labels.dtype}')
    used_labels = labels.astype(int)
    label_range = int(np.max(used_labels) + 1)
    label_groups = _block_face_adjacency_graph(faces, label_range,
                                              label_dist_th=label_dist_th)
    logger.debug((
        f'Build connected components for {label_groups.shape} label groups'
        f'{label_groups}'
    ))
    new_labeling = scipy.sparse.csgraph.connected_components(label_groups,
                                                             directed=False)[1]
    logger.debug(f'Initial {new_labeling.shape} connected labels:, {new_labeling}')
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique))[unique_inverse]
    logger.debug(f'Re-arranged {len(new_labeling)} connected labels:, {new_labeling}')
    return new_labeling


def _adjacent_faces(block_indices, faces):
    """Find faces which touch and pair them together in new data structure"""
    face_pairs = []
    faces_index_lookup = {bi: f for bi, f in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2*ax + 1]
                b = faces_index_lookup[neighbor_index][2*ax]
                face_pairs.append(np.concatenate((a, b), axis=ax))
            except KeyError:
                continue
    return face_pairs


def _block_face_adjacency_graph(faces, labels_range, label_dist_th=1.0):
    """
    Shrink labels in face plane, then find which labels touch across the face boundary
    """
    logger.info(f'Create adjacency graph for {labels_range} labels')
    all_mappings = [np.empty((2, 0), dtype=np.uint32)]
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    for face in faces:
        sl0 = tuple(slice(0, 1) if d == 2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d == 2 else slice(None) for d in face.shape)
        a = _shrink_labels(face[sl0], label_dist_th)
        b = _shrink_labels(face[sl1], label_dist_th)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))
        mapped = di_ndmeasure._utils._label._across_block_label_grouping(
            face,
            structure
        )
        all_mappings.append(mapped)
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    csr_mat = scipy.sparse.coo_matrix((v, (i, j)),
                                      shape=(labels_range,labels_range)).tocsr()
    logger.debug(f'Labels mapping as csr matrix {csr_mat}')
    return csr_mat


def _shrink_labels(plane, threshold):
    """
    Shrink labels in plane by some distance from their boundary
    """
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def _merge_all_boxes(boxes, box_ids):
    """Merge all boxes that map to the same box_ids"""
    merged_boxes = []
    boxes_array = np.array(boxes, dtype=object)
    for iii in np.unique(box_ids):
        merge_indices = np.argwhere(box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = _merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)
    return merged_boxes


def _merge_boxes(boxes):
    """Take union of two or more parallelpipeds"""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union


def _write_new_labeling(new_labeling_path, new_labeling):
    new_labeling_dir = os.path.dirname(new_labeling_path)
    os.makedirs(new_labeling_dir, exist_ok=True)
    np.save(new_labeling_path, new_labeling)


def _copy_relabeled_block(block_coords, final_labeling=None,
                          src_data=[], dest_data=[]):
    if final_labeling is None:
        return False
    else:
        logger.info(f'Relabel block: {block_coords}')
        block = src_data[block_coords]
        new_labeling_array = np.load(final_labeling)
        dest_data[block_coords] = new_labeling_array[block]
        return True
