import functools
import os
import numpy as np
import dask.array as da
import bigstream.utility as ut
from cellpose import models
from cellpose.io import logger_setup
import dask_image.ndmeasure._utils._label as label
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.ndimage import generate_binary_structure
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import zarr
import tempfile
from pathlib import Path
import yaml
import dask.config
from dask.distributed import Client, LocalCluster
from dask_jobqueue.lsf import LSFJob


######################## function(s) run on each block ########################

#----------------------- The main function -----------------------------------#
def process_block(
    coords,
    input_zarr,
    preprocessing_steps,
    model_kwargs,
    eval_kwargs,
    overlap,
    blocksize,
    nblocks,
    output_zarr,
):
    """
    Main pipeline that is run on each block.
    (1) Read block from disk, preprocess, and segment.
    (2) Remove overlaps.
    (3) Get bounding boxes for every segment.
    (4) Remap segment IDs to globally unique values.
    (5) Write segments to disk.
    (6) Get segmented block faces.
    """
    block_index, coords = coords[0], coords[1]
    print('RUNNING BLOCK: ', block_index, '\tREGION: ', coords, flush=True)
    segmentation = read_preprocess_and_segment(
        input_zarr, coords, preprocessing_steps, model_kwargs, eval_kwargs,
    )
    segmentation, coords = remove_overlaps(
        segmentation, coords, overlap, blocksize,
    )
    boxes = bounding_boxes_in_global_coordinates(segmentation, coords)
    segmentation, remap = global_segment_ids(segmentation, block_index, nblocks)
    output_zarr[tuple(coords)] = segmentation
    faces = block_faces(segmentation)
    return block_index, (faces, boxes, remap[1:])

#----------------------- component functions ---------------------------------#
def read_preprocess_and_segment(
    input_zarr,
    coords,
    preprocessing_steps,
    model_kwargs,
    eval_kwargs,
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""
    image = input_zarr[coords]
    for pp_step in preprocessing_steps:
        pp_step[1]['coords'] = coords
        image = pp_step[0](image, **pp_step[1])
    logger_setup()
    model = models.Cellpose(**model_kwargs)
    return model.eval(image, **eval_kwargs)[0].astype(np.uint32)


def remove_overlaps(array, coords, overlap, blocksize):
    """overlaps only there to provide context for boundary voxels
       reslice array to remove overlap regions"""
    coords_trimmed = list(coords)
    for axis in range(array.ndim):
        if coords[axis].start != 0:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(overlap, None)
            array = array[tuple(slc)]
            a, b = coords[axis].start, coords[axis].stop
            coords_trimmed[axis] = slice(a + overlap, b)
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None),]*array.nd
            slc[axis] = slice(None, blocksize[axis])
            array = array[tuple(slc)]
            a = coords_trimmed[axis].start
            coords_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, coords_trimmed


def bounding_boxes_in_global_coordinates(segmentation, coords):
    """bounding boxes (tuples of slices) are super useful later
       best to compute them now while things are distributed"""
    boxes = find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]
    translate = lambda a, b: slice(a.start+b.start, a.start+b.stop)
    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(translate(a, b) for a, b in zip(coords, box))
    return boxes


def global_segment_ids(segmentation, block_index, nblocks):
    """Independent blocks will have overlapping segment IDs
       pack the block index into the segment IDs so they are
       globally unique. Everything gets remapped to [1..N] later."""
    # NOTE: casting string to uint32 will overflow witout warning or exception
    #    so, using uint32 this only works if:
    #    number of blocks is less than 42950
    #    number of segments per block is less than or equal to 99999
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    p = str(np.ravel_multi_index(block_index, nblocks))
    remap = [np.uint32(p+str(x).zfill(5)) for x in unique]
    if unique[0] == 0: remap[0] = np.uint32(0)  # 0 should just always be 0
    segmentation = np.array(remap)[unique_inverse.reshape(segmentation.shape)]
    return segmentation, remap


def block_faces(segmentation):
    """slice faces along every axis"""
    faces = []
    for iii in range(segmentation.ndim):
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces


######################## Distributed Cellpose #################################

#----------------------- The main function -----------------------------------#
@cluster
def distributed_eval(
    input_zarr,
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
    Optionally use a mask to ignore background regions in image

    Parameters
    ----------
    input_zarr : zarr.core.Array
        A zarr.core.Array instance containing the image data you want to
        segment.

    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    write_path : string
        The location of a zarr file on disk where you'd like to write your results

    mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: [])
        Preprocessing steps to run on blocks before running cellpose. This might
        include things like Gaussian smoothing or gamma correction. The correct
        formatting is as follows:
            preprocessing_steps = [(pp1, {'arg1':val1}), (pp2, {'arg1':val1}), ...]
        Where `pp1` and `pp2` are functions. The first argument to these
        functions must be an nd-array, the image data to be preprocessed.
        These functions must also take a keyword argument named "coords"
        even if that argument is never used. The second item in each tuple
        is a dictionary containing any additional arguments to the function.

    model_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose

    eval_kwargs : dict (default: {})
        Arguments passed to cellpose.models.Cellpose.eval

    cluster : A dask cluster object (default: None)
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a dask cluster for the duration of this function,
        then close it when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments used to parameterize your cluster.
        If you are running locally, see the docstring for the myLocalCluster
        class below. If you are running on the Janelia LSF cluster, see
        the docstring for the janeliaLSFCluster class below.

    temporary_directory : string (default: None)
        Temporary files are created during segmentation. The temporary files
        will be in their own folder within the temporary_directory. The default
        is the current directory. Temporary files are removed if the function
        completes successfully.


    Returns
    -------
    A reference to the zarr array on disk containing the stitched cellpose
    segments for your entire image


    Some additional help on cluster parameterization
    ------------------------------------------------
    This function runs on a Dask cluster. There are basically three things you
    need to know about: (1) workers, (2) the task graph, and (3) the scheduler.
    You won't interact with any of these components directly, but in order
    to parameterize this function properly, you need to know a little about how
    they work.

    (1) Workers
    A "Dask worker" is a set of computational resources that have been set aside
    to execute tasks. Let's say you're working on a nice workstation computer with
    16 cpus and 128 GB of RAM. A worker might be 2 cpus and 16 GB of RAM. If that's
    the case then you could have up to 8 workers on that machine. What resources
    a worker has access to and how many workers you want are both configurable
    parameters, and different choices should be made depending on whether you're
    working on a workstation, a cluster, or a laptop.

    (2) The task graph
    Dask code is "lazy" meaning individual lines of dask code don't execute a
    computation right away; instead they just record what you want to do and
    then you specify later when you actually want to execute the computation.
    Say A and B are both Dask arrays of data. Then:
    C = A + B
    D = 2 * C
    won't actually compute anything right away. Instead, you have just
    constructed a "task graph" which contains all the instructions needed
    to compute D at a later time. When you actually want D computed you
    can write:
    D_result = D.compute()
    This will send the task graph for D to the scheduler, which we'll learn
    about next.

    (3) The Scheduler
    When you execute a Dask task graph it is sent to the scheduler. The
    scheduler also knows which workers are available and what resources
    they have. The scheduler will analyze the task graph and determine which
    individual tasks should be mapped to which workers and in what order.
    The scheduler runs in the same Python process as this function.

    The cluster_kwargs argument to this function is how you specify what
    resources workers will have and how many workers you will allow. These
    choices are different if you're on a cluster or a workstation.
    If you are working on the Janelia cluster, then you can see which options
    to put in the cluster_kwargs dictionary by running the following code:
    ```
    from ClusterWrap.clusters import janelia_lsf_cluster
    janelia_lsf_cluster
    ```
    If you are working on a workstation, then you can see which options
    to put in the cluster_kwargs dictionary by running the following code:
    ```
    from ClusterWrap.clusters import local_cluster
    local_cluster?
    ```
    """
    # TODO update dask stuff in docstring above

    # set default values
    if 'diameter' not in eval_kwargs.keys():
        eval_kwargs['diameter'] = 30
    overlap = eval_kwargs['diameter'] * 2
    blocksize = np.array(blocksize)
    if mask is not None:
        ratio = np.array(mask.shape) / input_zarr.shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    # get all foreground block coordinates
    nblocks = np.ceil(np.array(input_zarr.shape) / blocksize).astype(np.int16)
    block_coords = []
    for index in np.ndindex(*nblocks):
        start = blocksize * index - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(input_zarr.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        # check foreground
        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_coords = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_coords]): foreground = False
        if foreground:
            block_coords.append( (index, coords) )

    # construct zarr file for output
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    output_zarr_path = temporary_directory.name + '/segmentation_unstitched.zarr'
    output_zarr = ut.create_zarr(
        output_zarr_path,
        input_zarr.shape,
        blocksize,
        np.uint32,
    )


    # submit all segmentations, scale down cluster when complete
    futures = cluster.client.map(
        process_block,
        block_coords,
        input_zarr=input_zarr,
        preprocessing_steps=preprocessing_steps,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        overlap=overlap,
        blocksize=blocksize,
        nblocks=nblocks,
        output_zarr=output_zarr,
    )
    results = cluster.client.gather(futures)
    cluster.cluster.scale(0)


    # begin the local reduction step (merging label IDs)
    print('REDUCE STEP, SHOULD TAKE 5-10 MINUTES')

    # reformat to dict[block_index] = (faces, boxes, box_ids)
    results = {a:b for a, b in results}

    # find face adjacency pairs
    faces, boxes, box_ids = [], [], []
    for block_index, result in results.items():
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = result[0][2*ax + 1]
                b = results[neighbor_index][0][2*ax]
                faces.append( np.concatenate((a, b), axis=ax) )
            except KeyError:
                continue
        boxes += result[1]
        box_ids += result[2]

    # determine mergers
    label_range = np.max(box_ids)
    label_groups = _block_face_adjacency_graph(faces, label_range)
    new_labeling = connected_components(label_groups, directed=False)[1]
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.

    # map unused label ids to zero and remap remaining ids to [0, 1, 2, ...]
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[box_ids] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
    print('REDUCTION COMPLETE')


    # reformat worker properties for relabeling step
    cluster.change_worker_attributes(
        min_workers=10,
        max_workers=400,
        ncpus=1,
        memory="15GB",
        mem=int(15e9),
        queue=None,
        job_extra=[],
    )

    # define relabeling function
    np.save(temporary_directory.name + '/new_labeling.npy', new_labeling)
    def relabel_block(block):
        new_labeling = np.load(temporary_directory.name + '/new_labeling.npy')
        return new_labeling[block]

    # execute mergers and relabeling, write result to disk
    segmentation_da = da.from_zarr(output_zarr)
    relabeled = da.map_blocks(
        relabel_block,
        segmentation_da,
        dtype=np.uint32,
        chunks=segmentation_da.chunks,
    )
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


#----------------------- component functions ---------------------------------#
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



######################## Cluster related functions ############################

#----------------------- config stuff ----------------------------------------#
DEFAULT_CONFIG_FILENAME = 'distributed_cellpose_dask_config.yaml'

def _config_path(config_name):
    """Add config directory path to config filename"""
    return str(Path.home()) + '/.config/dask/' + config_name


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
    if os.path.exists(config_path): os.remove(config_path)


#----------------------- clusters --------------------------------------------#
class myLocalCluster(LocalCluster):
    """
    This is a thin wrapper extending dask.distributed.LocalCluster to set
    configs before the cluster or workers are initialized.

    For a list of full arguments (how to specify your worker resources) see:
    https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster
    You need to know how many cpu cores and how much RAM your machine has.

    Most users will only need to specify:
    n_workers
    memory_limit (which is the limit per worker)
    threads_per_workers (for most workflows this should be 1)

    You can also modify any dask configuration option through the
    config argument (first argument to constructor).
    """

    def __init__(
        self,
        config={},
        config_name=DEFAULT_CONFIG_FILENAME,
        persist_config=False,
        **kwargs,
    ):
        self.config_name = config_name
        self.persist_config = persist_config
        _modify_dask_config(config, config_name)
        if "host" not in kwargs: kwargs["host"] = ""
        self.cluster = super().__init__(**kwargs)
        self.client = Client(self.cluster)
        print("Cluster dashboard link: ", self.cluster.dashboard_link)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.persist_config:
            _remove_config_file(config_name)
        self.client.close()
        self.cluster.__exit__(exc_type, exc_value, traceback)


class janeliaLSFCluster(LSFCluster):
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
        # config
        self.config_name = config_name
        self.persist_config = persist_config
        scratch_dir = f"/scratch/{os.environ['USER']}/"
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
            Path(log_dir).mkdir(parents=False, exist_ok=True)
            kwargs["log_directory"] = log_dir

        # graceful exit for lsf jobs (adds -d flag)
        class quietLSFJob(LSFJob):
            cancel_command = "bkill -d"

        # create cluster
        self.cluster = super().__init__(
            ncpus=ncpus,
            processes=1,
            cores=1,
            memory=str(15*ncpus)+'GB',
            mem=int(15e9*ncpus),
            job_script_prologue=job_script_prologue,
            job_cls=quietLSFJob,
            **kwargs,
        )
        self.client = Client(self.cluster)
        print("Cluster dashboard link: ", self.cluster.dashboard_link)

        # set adaptive cluster bounds
        self.adapt_cluster(min_workers, max_workers)


    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.persist_config:
            _remove_config_file(config_name)
        self.client.close()
        self.cluster.__exit__(exc_type, exc_value, traceback)


    def adapt_cluster(self, min_workers, max_workers):
        _ = self.cluster.adapt(
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
        self.cluster.scale(0)
        for k, v in kwargs.items():
            self.cluster.new_spec['options'][k] = v
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
        assert kwargs['cluster'] or kwargs['cluster_kwargs'], \
        "Either cluster or cluster_kwargs must be defined"
        if kwargs['cluster'] is None:
            cluster_constructor = myLocalCluster
            F = lambda x: x in kwargs['cluster_kwargs']
            if F('ncpus') and F('min_workers') and F('max_workers'):
                cluster_constructor = janeliaLSFCluster
            with cluster_constructor(**kwargs['cluster_kwargs']) as cluster:
                kwargs['cluster'] = cluster
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return create_or_pass_cluster


