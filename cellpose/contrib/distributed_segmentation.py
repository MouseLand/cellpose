# stdlib imports
import contextlib
import os, getpass, datetime, pathlib, tempfile, functools, glob

# non-stdlib core dependencies
import numpy as np
import scipy
import cellpose.io
import cellpose.models
import tifffile
import imagecodecs

# distributed dependencies
import dask
import distributed
import dask_image.ndmeasure
import yaml
import zarr
import dask_jobqueue




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
    if os.path.exists(config_path): os.remove(config_path)


#----------------------- helpers ---------------------------------------------#
def _parse_memory_mb(memory):
    """Accept memory as int (MB) or str like '125000', '125GB', '125 GiB'
    and return an integer in megabytes. Used by the SLURM cluster wrapper
    to pass the right format to both dask (string with unit) and to
    SLURM's --mem (integer MB)."""
    if isinstance(memory, (int, np.integer)):
        return int(memory)
    s = str(memory).strip().lower().replace(" ", "")
    # strip optional 'b' (so '125gb' and '125g' both work)
    if s.endswith("ib"):
        s = s[:-2]
        binary = True
    elif s.endswith("b"):
        s = s[:-1]
        binary = False
    else:
        binary = False
    units = {"k": 1, "m": 1, "g": 1000, "t": 1000_000}
    units_bin = {"k": 1, "m": 1, "g": 1024, "t": 1024 * 1024}
    if s and s[-1] in "kmgt":
        n = float(s[:-1])
        mul = (units_bin if binary else units)[s[-1]]
        return int(round(n * mul))
    # plain number → already MB
    return int(round(float(s)))


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
        if "host" not in kwargs: kwargs["host"] = ""
        super().__init__(**kwargs)
        self.client = distributed.Client(self)

        # set environment variables for workers (threading)
        environment_vars = {
            'MKL_NUM_THREADS':str(2*ncpus),
            'NUM_MKL_THREADS':str(2*ncpus),
            'OPENBLAS_NUM_THREADS':str(2*ncpus),
            'OPENMP_NUM_THREADS':str(2*ncpus),
            'OMP_NUM_THREADS':str(2*ncpus),
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


class slurmCluster(dask_jobqueue.SLURMCluster):
    """
    Thin wrapper around dask_jobqueue.SLURMCluster, which in turn extends
    dask.distributed.SpecCluster. This wrapper sets configs before the
    cluster or workers are initialized. This is an adaptive cluster and
    will scale the number of workers, between user-specified limits, based
    on the number of pending tasks.

    Defaults are tuned for GPU jobs on a typical SLURM cluster (one worker
    per submitted job, one GPU per worker). Tested on the MPCDF Raven
    cluster; should work on any SLURM cluster by adjusting partition,
    account, and the GPU directives.

    For a full list of arguments see
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html

    Most users only need to specify:
        ncpus       (CPU cores per worker, sets ``--cpus-per-task``)
        min_workers (lower bound for adaptive scaling)
        max_workers (upper bound for adaptive scaling)
        walltime    (e.g. ``"24:00:00"``)
        memory      (e.g. ``"125GB"``)

    Cluster-specific extras such as ``--constraint=gpu`` and
    ``--gres=gpu:a100:1`` can be passed via ``job_extra_directives``. A
    ``partition`` and/or ``account`` may be required by your cluster.

    Example (MPCDF Raven, 1 A100 GPU per worker)::

        cluster_kwargs = {
            'cluster_type': 'slurm',
            'ncpus': 18,
            'min_workers': 1,
            'max_workers': 4,
            'walltime': '24:00:00',
            'memory': '125GB',
            'job_extra_directives': [
                '--constraint=gpu',
                '--gres=gpu:a100:1',
            ],
        }

    Example (MPCDF Raven, 4 A100 GPUs per SLURM job, 4 dask workers per
    job via dask-cuda — useful when the per-user concurrent-job cap (8)
    bottlenecks total GPUs)::

        cluster_kwargs = {
            'cluster_type': 'slurm',
            'ncpus': 18,           # CPUs per GPU/worker; the wrapper multiplies by gpus_per_job
            'memory': '125000',    # MB per GPU/worker; multiplied by gpus_per_job
            'gpus_per_job': 4,
            'min_workers': 1,
            'max_workers': 8,      # 8 jobs * 4 GPUs = 32 effective workers
            'walltime': '24:00:00',
            'job_extra_directives': [
                '--constraint=gpu',
                '--gres=gpu:a100:4',
            ],
        }
    """

    def __init__(
        self,
        ncpus,
        min_workers,
        max_workers,
        walltime='24:00:00',
        memory=None,
        partition=None,
        account=None,
        gpus_per_job=1,
        config={},
        config_name=DEFAULT_CONFIG_FILENAME,
        persist_config=False,
        **kwargs,
    ):
        """``gpus_per_job`` controls multi-GPU mode. The default of 1
        keeps single-GPU behavior. Setting it to >1 makes each submitted
        SLURM job hold multiple GPUs and starts that many dask workers
        per job (one per GPU) using ``dask-cuda-worker`` for per-process
        GPU isolation. Requires ``dask-cuda`` (and a ``dask_cuda_worker_entry``
        importable shim that calls dask_cuda.cli.worker) to be installed
        in the worker env.

        With multi-GPU mode the caller still passes ``ncpus`` as the
        per-GPU CPU count (e.g. 18 on Raven). The wrapper multiplies that
        by ``gpus_per_job`` for the SLURM job_cpu/job_mem directives, so
        a 4-GPU job on Raven gets cpus=72, mem=500000.
        """

        # store all args in case needed later
        self.locals_store = {**locals()}
        self.gpus_per_job = int(gpus_per_job)

        # config
        self.config_name = config_name
        self.persist_config = persist_config
        # /ptmp/<user> is the recommended fast scratch on Raven; fall back
        # to a per-user dir under /tmp on systems that don't have /ptmp.
        ptmp = pathlib.Path(f"/ptmp/{getpass.getuser()}")
        scratch_dir = (str(ptmp) + "/") if ptmp.is_dir() \
            else f"/tmp/{getpass.getuser()}/"
        config_defaults = {
            'temporary-directory': scratch_dir,
            'distributed.comm.timeouts.connect': '180s',
            'distributed.comm.timeouts.tcp': '360s',
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
        # allow caller to extend (rather than replace) the prologue
        user_prologue = kwargs.pop('job_script_prologue', [])
        job_script_prologue = job_script_prologue + list(user_prologue)

        # set scratch and log directories
        if "local_directory" not in kwargs:
            kwargs["local_directory"] = scratch_dir
        if "log_directory" not in kwargs:
            log_dir = f"{os.getcwd()}/dask_worker_logs_{os.getpid()}/"
            pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)
            kwargs["log_directory"] = log_dir

        # default per-worker memory: ~15 GB per requested CPU
        if memory is None:
            memory_mb = 15 * 1000 * ncpus
        else:
            memory_mb = _parse_memory_mb(memory)

        # In multi-GPU mode the SLURM job holds gpus_per_job GPUs and runs
        # gpus_per_job dask worker processes (one per GPU). Scale the SLURM
        # cpu and memory totals accordingly. ncpus and memory remain the
        # *per-GPU* (per-worker) numbers as the caller passed them.
        n_dask_workers_per_job = self.gpus_per_job
        slurm_cpus = ncpus * self.gpus_per_job
        slurm_mem_mb = memory_mb * self.gpus_per_job
        # dask's per-worker memory budget — *per worker process*, so just memory_mb.
        dask_worker_memory = f"{memory_mb}MB"
        # cores= controls how many dask workers run in this SLURM job.
        # processes= splits cores across that many worker processes.
        # For dask-cuda mode we set both = gpus_per_job and let dask-cuda-worker
        # pin each to a different CUDA_VISIBLE_DEVICES.
        # SLURM --mem expects an integer of MB (the safest cross-cluster
        # form; G/GiB/GB suffixes get interpreted differently by different
        # SLURM versions and by some site rules — Raven for example treats
        # "G" as GiB, which silently exceeds the 1/4-of-node share cap on
        # GPU jobs). Dask, on the other hand, parses its `memory` argument
        # as a human-readable string and treats a bare number as BYTES,
        # which would set a microscopic per-worker memory budget. So we
        # pass the SLURM directive as a plain int and the dask budget as
        # a unit-suffixed string.
        super_kwargs = dict(
            cores=n_dask_workers_per_job,
            processes=n_dask_workers_per_job,
            memory=f"{slurm_mem_mb}MB",
            walltime=walltime,
            queue=partition,
            account=account,
            job_cpu=slurm_cpus,
            job_mem=str(slurm_mem_mb),
            job_script_prologue=job_script_prologue,
        )
        if self.gpus_per_job > 1:
            # Use dask-cuda's worker so each of the N dask processes in this
            # SLURM job binds to exactly one GPU. We point dask-jobqueue at
            # a small shim module that calls dask_cuda.cli.worker, since
            # the package only ships a console script (no `__main__.py`).
            super_kwargs["worker_command"] = "dask_cuda_worker_entry"
        super().__init__(
            **super_kwargs,
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
        """Drop existing workers and respawn with updated kwargs.

        WARNING: dangerous if you don't know what you're doing.

        Used by ``distributed_eval`` to release GPUs and shrink workers
        for the cheap stitching phase. Updates the canonical
        ``self._job_kwargs`` store; ``self.new_spec['options']`` is the
        same dict (per dask_jobqueue.JobQueueCluster.__init__) so it
        picks up the change. ``_dummy_job`` is a property and recomputes
        the SLURM header on next access. The job_script preview below
        is logged so future runs can verify the new directives reach
        the queued jobs."""
        self.scale(0)
        # Block until existing workers actually leave; otherwise adapt()
        # below sees them and skips the respawn.
        try:
            self.sync(self._correct_state)
        except Exception:
            pass
        # SpecCluster shallow-copied the worker dict at __init__, so
        # ``self.new_spec['options']`` and ``self._job_kwargs`` are the
        # same dict object. We update ``_job_kwargs`` (the canonical
        # store) and assert the alias still holds.
        self._job_kwargs.update(kwargs)
        assert self.new_spec['options'] is self._job_kwargs, (
            "internal invariant broken: new_spec['options'] is no longer "
            "self._job_kwargs; dask_jobqueue API changed?"
        )
        # Log the freshly-rendered job script so the SBATCH directives
        # are visible in the driver log — makes it obvious whether the
        # new kwargs propagated. Truncated to the header.
        try:
            preview = "\n".join(
                ln for ln in self._dummy_job.job_script().split("\n")
                if ln.startswith("#") or "dask" in ln.lower()
            )
            print("change_worker_attributes -> next job script header:")
            print(preview)
        except Exception as exc:  # pragma: no cover
            print(f"change_worker_attributes: could not preview job script: {exc}")
        self.adapt_cluster(min_workers, max_workers)


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
CLUSTER_CONSTRUCTORS = {
    'local': myLocalCluster,
    'lsf': janeliaLSFCluster,
    'slurm': slurmCluster,
}


def cluster(func):
    """
    This decorator ensures a function will run inside a cluster
    as a context manager. The decorated function, "func", must
    accept "cluster" and "cluster_kwargs" as parameters. If
    "cluster" is not None then the user has provided an existing
    cluster and we just run func. If "cluster" is None then
    "cluster_kwargs" are used to construct a new cluster, and
    the function is run inside that cluster context.

    The cluster type is selected by the ``cluster_type`` key in
    ``cluster_kwargs`` ("local", "lsf", or "slurm"). For backward
    compatibility, when ``cluster_type`` is not given the choice
    defaults to LSF if ``ncpus``, ``min_workers``, and ``max_workers``
    are all present (the original LSF signature) and to local otherwise.
    """
    @functools.wraps(func)
    def create_or_pass_cluster(*args, **kwargs):
        # TODO: this only checks if args are explicitly present in function call
        #       it does not check if they are set correctly in any way
        assert 'cluster' in kwargs or 'cluster_kwargs' in kwargs, \
        "Either cluster or cluster_kwargs must be defined"
        if not 'cluster' in kwargs:
            cluster_kwargs = dict(kwargs['cluster_kwargs'])  # copy; we pop
            cluster_type = cluster_kwargs.pop('cluster_type', None)
            if cluster_type is None:
                F = lambda x: x in cluster_kwargs
                if F('ncpus') and F('min_workers') and F('max_workers'):
                    cluster_type = 'lsf'
                else:
                    cluster_type = 'local'
            if cluster_type not in CLUSTER_CONSTRUCTORS:
                raise ValueError(
                    f"Unknown cluster_type {cluster_type!r}. "
                    f"Choose from {list(CLUSTER_CONSTRUCTORS)}."
                )
            cluster_constructor = CLUSTER_CONSTRUCTORS[cluster_type]
            with cluster_constructor(**cluster_kwargs) as cluster:
                kwargs['cluster'] = cluster
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return create_or_pass_cluster




######################## the function to run on each block ####################

#----------------------- The main function -----------------------------------#
def process_block(
    block_index,
    crop,
    input_zarr,
    model_kwargs,
    eval_kwargs,
    blocksize,
    overlap,
    output_zarr,
    preprocessing_steps=[],
    worker_logs_directory=None,
    test_mode=False,
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

    A user may want to test this function on one block before running
    the distributed function. When test_mode=True, steps (5) and (6)
    are omitted and replaced with:

    (5) return remapped segments as a numpy array, boxes, and box_ids

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid

    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    input_zarr : zarr.core.Array
        The image data we want to segment

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

    model_kwargs : dict
        Arguments passed to cellpose.models.Cellpose
        This is how you select and parameterize a model.

    eval_kwargs : dict
        Arguments passed to the eval function of the Cellpose model
        This is how you parameterize model evaluation.

    blocksize : iterable (list, tuple, np.ndarray)
        The number of voxels (the shape) of blocks without overlaps

    overlap : int
        The number of voxels added to the blocksize to provide context
        at the edges

    output_zarr : zarr.core.Array
        A location where segments can be stored temporarily before
        merger is complete

    worker_logs_directory : string (default: None)
        A directory path where log files for each worker can be created
        The directory must exist

    test_mode : bool (default: False)
        The primary use case of this function is to be called by
        distributed_eval (defined later in this same module). However
        you may want to call this function manually to test what
        happens to an individual block; this is a good idea before
        ramping up to process big data and also useful for debugging.

        When test_mode is False (default) this function stores
        the segments and returns objects needed for merging between
        blocks.

        When test_mode is True this function does not store the
        segments, and instead returns them to the caller as a numpy
        array. The boxes and box IDs are also returned. When test_mode
        is True, you can supply dummy values for many of the inputs,
        such as:

        block_index = (0, 0, 0)
        output_zarr=None

    Returns
    -------
    If test_mode == False (the default), three things are returned:
        faces : a list of numpy arrays - the faces of the block segments
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes

    If test_mode == True, three things are returned:
        segments : np.ndarray containing the segments with globally unique IDs
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes
    """
    print('RUNNING BLOCK: ', block_index, '\tREGION: ', crop, flush=True)
    segmentation = read_preprocess_and_segment(
        input_zarr, crop, preprocessing_steps, model_kwargs, eval_kwargs,
        worker_logs_directory,
    )
    segmentation, crop = remove_overlaps(
        segmentation, crop, overlap, blocksize,
    )
    boxes = bounding_boxes_in_global_coordinates(segmentation, crop)
    nblocks = get_nblocks(input_zarr.shape, blocksize)
    segmentation, remap = global_segment_ids(segmentation, block_index, nblocks)
    if remap[0] == 0: remap = remap[1:]

    if test_mode: return segmentation, boxes, remap
    output_zarr[tuple(crop)] = segmentation
    faces = block_faces(segmentation)
    return faces, boxes, remap


#----------------------- component functions ---------------------------------#
def read_preprocess_and_segment(
    input_zarr,
    crop,
    preprocessing_steps,
    model_kwargs,
    eval_kwargs,
    worker_logs_directory,
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""
    image = input_zarr[crop]
    for pp_step in preprocessing_steps:
        pp_step[1]['crop'] = crop
        image = pp_step[0](image, **pp_step[1])
    log_file=None
    if worker_logs_directory is not None:
        log_file = f'dask_worker_{distributed.get_worker().name}.log'
        log_file = pathlib.Path(worker_logs_directory).joinpath(log_file)
    cellpose.io.logger_setup(stdout_file_replacement=log_file)
    model = cellpose.models.CellposeModel(**model_kwargs)
    return model.eval(image, **eval_kwargs)[0].astype(np.uint32)


def remove_overlaps(array, crop, overlap, blocksize):
    """overlaps only there to provide context for boundary voxels
       and can be removed after segmentation is complete
       reslice array to remove the overlaps"""
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        if crop[axis].start != 0:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(overlap, None)
            array = array[tuple(slc)]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlap, b)
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(None, blocksize[axis])
            array = array[tuple(slc)]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed


def bounding_boxes_in_global_coordinates(segmentation, crop):
    """bounding boxes (tuples of slices) are super useful later
       best to compute them now while things are distributed"""
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]
    translate = lambda a, b: slice(a.start+b.start, a.start+b.stop)
    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(translate(a, b) for a, b in zip(crop, box))
    return boxes


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def global_segment_ids(segmentation, block_index, nblocks):
    """pack the block index into the segment IDs so they are
       globally unique. Everything gets remapped to [1..N] later.
       A uint32 is split into 5 digits on left and 5 digits on right.
       This creates limits: 42950 maximum number of blocks and
       99999 maximum number of segments per block"""
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
    resume_dir=None,
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask to ignore background regions in image.
    Either cluster or cluster_kwargs parameter must be set to a
    non-default value; please read these parameter descriptions below.

    Three cluster types are supported out of the box, selectable via the
    ``cluster_type`` key in ``cluster_kwargs``:

    - ``"local"`` -> :class:`myLocalCluster` (single workstation)
    - ``"lsf"``   -> :class:`janeliaLSFCluster` (Janelia LSF cluster)
    - ``"slurm"`` -> :class:`slurmCluster` (any SLURM cluster, defaults
      tuned for MPCDF Raven)

    Running on a different cluster manager will require implementing your
    own dask cluster class; the existing classes are good starting points,
    and the dask_jobqueue library covers most schedulers. PRs with a
    solid start are welcome.

    If running on a workstation, please read the docstring for the
    LocalCluster class defined in this module. For LSF or SLURM, see the
    docstrings for the corresponding cluster classes.

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
        class in this module. If you are running on the Janelia LSF cluster, see
        the docstring for the janeliaLSFCluster class in this module. If you are
        running on a different institute cluster, you may need to implement
        a dask cluster object that conforms to the requirements of your cluster.

    temporary_directory : string (default: None)
        Temporary files are created during segmentation. The temporary files
        will be in their own folder within the temporary_directory. The default
        is the current directory. Temporary files are removed if the function
        completes successfully.

    resume_dir : string (default: None)
        If set, this exact path is used as the (persistent) tempdir instead
        of a randomly named one under ``temporary_directory``. The directory
        is created if missing and is NOT auto-deleted, so a subsequent call
        with the same ``resume_dir`` will detect blocks that were already
        segmented in a previous run and skip them.

        A block is considered "already done" if its corresponding chunk
        file in the unstitched temp zarr exists on disk (the temp zarr is
        chunked at exactly ``blocksize`` so each block maps to one chunk
        file). For each already-done block the per-block return values
        ``(faces, boxes, remap)`` are recomputed from the saved
        segmentation, and stitching proceeds on the union of recomputed
        and freshly-computed results.

        Use this to recover from a SLURM walltime kill: pass the original
        run's tempdir as ``resume_dir`` to the new run; only un-segmented
        blocks will actually be processed.

    Returns
    -------
    Two values are returned:
    (1) A reference to the zarr array on disk containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """

    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    worker_logs_dirname = f'dask_worker_logs_{timestamp}'
    worker_logs_dir = pathlib.Path().absolute().joinpath(worker_logs_dirname)
    worker_logs_dir.mkdir()

    if 'diameter' not in eval_kwargs.keys():
        eval_kwargs['diameter'] = 30
    # diameter may be a float; overlap is used to build zarr slices and
    # must be an int.
    overlap = int(eval_kwargs['diameter'] * 2)
    block_indices, block_crops = get_block_crops(
        input_zarr.shape, blocksize, overlap, mask,
    )

    # tempdir context: random-and-auto-deleted by default, or persistent
    # at `resume_dir` if the caller wants resume capability.
    if resume_dir is not None:
        pathlib.Path(resume_dir).mkdir(parents=True, exist_ok=True)
        tempdir_ctx = contextlib.nullcontext(resume_dir)
    else:
        tempdir_ctx = tempfile.TemporaryDirectory(
            prefix='.', suffix='_distributed_cellpose_tempdir',
            dir=temporary_directory or os.getcwd(),
        )
    with tempdir_ctx as temporary_directory:

        temp_zarr_path = temporary_directory + '/segmentation_unstitched.zarr'
        # Open existing temp zarr (resume) or create fresh.
        if resume_dir is not None and pathlib.Path(temp_zarr_path).exists():
            temp_zarr = zarr.open(temp_zarr_path, mode='a')
            assert temp_zarr.shape == tuple(input_zarr.shape), (
                f"resume_dir {resume_dir} has temp zarr of shape {temp_zarr.shape} "
                f"but input is {tuple(input_zarr.shape)}"
            )
        else:
            temp_zarr = zarr.open(
                temp_zarr_path, 'w',
                shape=input_zarr.shape,
                chunks=blocksize,
                dtype=np.uint32,
            )

        # Resume: classify blocks into todo (need to run) vs done (chunk
        # file already on disk in the temp zarr) and reconstruct the done
        # blocks' return values from the saved segmentation. We freeze
        # done_indices BEFORE workers start so the ordering is stable
        # (workers will write chunks for new blocks during the run, which
        # would otherwise confuse a re-classification at gather time).
        done_indices, done_crops = [], []
        precomputed_results = []
        todo_indices, todo_crops = list(block_indices), list(block_crops)
        if resume_dir is not None:
            done_indices, done_crops = [], []
            todo_indices, todo_crops = [], []
            for idx, crop in zip(block_indices, block_crops):
                if block_chunk_path(temp_zarr_path, idx).exists():
                    done_indices.append(idx)
                    done_crops.append(crop)
                else:
                    todo_indices.append(idx)
                    todo_crops.append(crop)
            print(
                f"resume: {len(done_indices)} blocks already done, "
                f"{len(todo_indices)} blocks to process",
                flush=True,
            )
            for k, (idx, crop) in enumerate(zip(done_indices, done_crops), start=1):
                precomputed_results.append(
                    recompute_block_results(
                        temp_zarr, idx, crop, overlap, blocksize,
                    )
                )
                if k % 100 == 0 or k == len(done_indices):
                    print(
                        f"  recomputed {k}/{len(done_indices)} done blocks",
                        flush=True,
                    )

        futures = cluster.client.map(
            process_block,
            todo_indices,
            todo_crops,
            input_zarr=input_zarr,
            preprocessing_steps=preprocessing_steps,
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            blocksize=blocksize,
            overlap=overlap,
            output_zarr=temp_zarr,
            worker_logs_directory=str(worker_logs_dir),
        )
        new_results = cluster.client.gather(futures)
        if isinstance(cluster, dask_jobqueue.core.JobQueueCluster):
            cluster.scale(0)

        # Combine results: precomputed (done) first, then freshly-computed.
        # block_indices is rewritten to match this ordering so downstream
        # stitching code keeps the parallel-list invariant with results.
        results = list(precomputed_results) + list(new_results)
        block_indices = list(done_indices) + list(todo_indices)

        faces, boxes_, box_ids_ = list(zip(*results))
        boxes = [box for sublist in boxes_ for box in sublist]
        box_ids = np.concatenate(box_ids_).astype(int)  # unsure how but without cast these are float64
        new_labeling = determine_merge_relabeling(block_indices, faces, box_ids)
        debug_unique = np.unique(new_labeling)
        new_labeling_path = temporary_directory + '/new_labeling.npy'
        np.save(new_labeling_path, new_labeling)

        # stitching step is cheap, we should release gpus and use small workers
        if isinstance(cluster, janeliaLSFCluster):
            cluster.change_worker_attributes(
                min_workers=cluster.locals_store['min_workers'],
                max_workers=cluster.locals_store['max_workers'],
                ncpus=1,
                memory="15GB",
                mem=int(15e9),
                queue=None,
                job_extra_directives=[],
            )
        elif isinstance(cluster, slurmCluster):
            cluster.change_worker_attributes(
                min_workers=cluster.locals_store['min_workers'],
                max_workers=cluster.locals_store['max_workers'],
                cores=1,
                processes=1,
                memory="15GB",
                job_cpu=1,
                job_mem="15GB",
                # drop GPU constraints / gres so stitching workers run on CPU nodes
                job_extra_directives=[],
            )
    
        segmentation_da = dask.array.from_zarr(temp_zarr)
        relabeled = dask.array.map_blocks(
            lambda block: np.load(new_labeling_path)[block],
            segmentation_da,
            dtype=np.uint32,
            chunks=segmentation_da.chunks,
        )
        dask.array.to_zarr(relabeled, write_path, overwrite=True)
        merged_boxes = merge_all_boxes(boxes, new_labeling[box_ids])
        return zarr.open(write_path, mode='r'), merged_boxes


#----------------------- component functions ---------------------------------#
def get_block_crops(shape, blocksize, overlap, mask):
    """Given a voxel grid shape, blocksize, and overlap size, construct
       tuples of slices for every block; optionally only include blocks
       that contain foreground in the mask. Returns parallel lists,
       the block indices and the slice tuples."""
    blocksize = np.array(blocksize)
    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]): foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)
    return indices, crops


def determine_merge_relabeling(block_indices, faces, used_labels):
    """Determine boundary segment mergers, remap all label IDs to merge
       and put all label IDs in range [1..N] for N global segments found"""
    faces = adjacent_faces(block_indices, faces)
    # FIX float parameters
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    used_labels = used_labels.astype(int)
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    label_range = int(np.max(used_labels))

    label_groups = block_face_adjacency_graph(faces, label_range)
    new_labeling = scipy.sparse.csgraph.connected_components(
        label_groups, directed=False)[1]
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
    return new_labeling


def adjacent_faces(block_indices, faces):
    """Find faces which touch and pair them together in new data structure"""
    face_pairs = []
    faces_index_lookup = {a:b for a, b in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2*ax + 1]
                b = faces_index_lookup[neighbor_index][2*ax]
                face_pairs.append( np.concatenate((a, b), axis=ax) )
            except KeyError:
                continue
    return face_pairs


def block_face_adjacency_graph(faces, nlabels):
    """Shrink labels in face plane, then find which labels touch across the
    face boundary"""
    # FIX float parameters
    # print("Initial nlabels:", nlabels, "Type:", type(nlabels))
    nlabels = int(nlabels)
    # print("Final nlabels:", nlabels, "Type:", type(nlabels))

    all_mappings = []
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    for face in faces:
        sl0 = tuple(slice(0, 1) if d==2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d==2 else slice(None) for d in face.shape)
        a = shrink_labels(face[sl0], 1.0)
        b = shrink_labels(face[sl1], 1.0)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))
        mapped = dask_image.ndmeasure._utils._label._across_block_label_grouping(face, structure)
        all_mappings.append(mapped)
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    return scipy.sparse.coo_matrix((v, (i, j)), shape=(nlabels+1, nlabels+1)).tocsr()


def shrink_labels(plane, threshold):
    """Shrink labels in plane by some distance from their boundary"""
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def compute_trimmed_crop(crop, overlap, blocksize):
    """Pure-function reproduction of remove_overlaps' crop adjustment, for
    use when we want to know where a block lives in the temp zarr without
    having the segmentation array on hand. Mirrors remove_overlaps logic
    exactly: trim left by overlap if the crop didn't start at zero on
    that axis, then trim right to blocksize if there's still room."""
    crop_trimmed = list(crop)
    for axis in range(len(crop)):
        length = crop[axis].stop - crop[axis].start
        if crop[axis].start != 0:
            crop_trimmed[axis] = slice(
                crop[axis].start + overlap, crop[axis].stop
            )
            length -= overlap
        if length > blocksize[axis]:
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return tuple(crop_trimmed)


def recompute_block_results(temp_zarr, block_index, crop, overlap, blocksize):
    """Reconstruct (faces, boxes, remap) for an already-completed block by
    reading its segmentation from the temp zarr. Used in resume mode where
    we don't have the original return values from process_block.

    The saved segmentation is already globally-remapped (labels packed
    with block index, max can be ~10^9). Calling
    ``scipy.ndimage.find_objects`` on it directly allocates a list of
    None entries up to ``max(label)`` — gigabytes per call, OOMing the
    driver after a few hundred blocks. Instead we relabel locally to
    1..N (sequential) via searchsorted, run find_objects on that small
    label space, and translate the resulting per-label boxes back into
    global coordinates."""
    crop_trimmed = compute_trimmed_crop(crop, overlap, blocksize)
    segmentation = np.asarray(temp_zarr[crop_trimmed])

    # Drop 0 (background). The unique non-zero labels in this block are
    # both the block's "remap" (parallel to its boxes) and the lookup
    # table for our local relabeling.
    remap_arr = np.unique(segmentation)
    if remap_arr.size > 0 and remap_arr[0] == 0:
        remap_arr = remap_arr[1:]

    if remap_arr.size == 0:
        boxes = []
    else:
        # Build a local-label image: 0 stays 0, label remap_arr[k] becomes k+1.
        # searchsorted on a sorted unique array gives a 0-based index, +1
        # so background stays at 0.
        local_seg = np.zeros_like(segmentation, dtype=np.int32)
        nonzero = segmentation != 0
        local_seg[nonzero] = (
            np.searchsorted(remap_arr, segmentation[nonzero]).astype(np.int32) + 1
        )
        local_boxes = scipy.ndimage.find_objects(local_seg)
        # find_objects returns a list of length max(local_seg)==len(remap_arr).
        # Each entry is either a slice tuple or None.
        translate = lambda a, b: slice(a.start + b.start, a.start + b.stop)
        boxes = []
        for box in local_boxes:
            if box is None:
                continue
            boxes.append(tuple(translate(a, b) for a, b in zip(crop_trimmed, box)))

    # block_faces returns numpy views into `segmentation`. Without a copy
    # the views keep the full 64 MB block alive in `precomputed_results`
    # for the rest of the run; for ~900 blocks that's 50+ GB resident.
    faces = [f.copy() for f in block_faces(segmentation)]
    return faces, boxes, list(remap_arr)


def block_chunk_path(temp_zarr_path, block_index, dimension_separator='.'):
    """Path of the zarr v2 chunk file for the given block. The temp zarr is
    chunked at blocksize so block_index == chunk index."""
    name = dimension_separator.join(str(int(i)) for i in block_index)
    return pathlib.Path(temp_zarr_path) / name


def merge_all_boxes(boxes, box_ids):
    """Merge all boxes that map to the same box_ids.

    Returns one merged box per unique id, in sorted-id order (matching
    the legacy ``np.unique(box_ids)`` iteration order).

    Vectorized via ``argsort`` + ``np.minimum/maximum.reduceat`` to keep
    runtime at O(N log N * ndim). The previous per-id ``argwhere`` loop
    was O(N) per group; on volumes with ~10^7 unique labels the
    quadratic blow-up made the stitching tail wedge for hours."""
    box_ids = np.asarray(box_ids).astype(int)
    n = len(boxes)
    if n == 0:
        return []

    # Pull starts/stops out of the slice tuples into dense (N, ndim) arrays.
    ndim = len(boxes[0])
    starts = np.empty((n, ndim), dtype=np.int64)
    stops = np.empty((n, ndim), dtype=np.int64)
    for i, box in enumerate(boxes):
        for d, s in enumerate(box):
            starts[i, d] = s.start
            stops[i, d] = s.stop

    # Sort rows by id so same-id rows are contiguous, then reduce per group.
    order = np.argsort(box_ids, kind="stable")
    box_ids_sorted = box_ids[order]
    starts_sorted = starts[order]
    stops_sorted = stops[order]

    # `return_index=True` on a sorted array gives the start of each run,
    # which is exactly what reduceat needs.
    unique_ids, group_starts = np.unique(box_ids_sorted, return_index=True)
    merged_starts = np.minimum.reduceat(starts_sorted, group_starts, axis=0)
    merged_stops = np.maximum.reduceat(stops_sorted, group_starts, axis=0)

    return [
        tuple(slice(int(merged_starts[i, d]), int(merged_stops[i, d]))
              for d in range(ndim))
        for i in range(len(unique_ids))
    ]


def merge_boxes(boxes):
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


