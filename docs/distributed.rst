Big Data
------------------------------------------------

Distributed Cellpose for larger-than-memory data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cellpose.contrib.distributed_cellpose`` module is intended to help run cellpose on 3D datasets
that are too large to fit in system memory. The dataset is divided into overlapping blocks and
each block is segmented separately. Results are stitched back together into a seamless segmentation
of the whole dataset.

Built to run on workstations or clusters. Blocks can be run in parallel, in series, or both. 
Compute resources (GPUs, CPUs, and RAM) can be arbitrarily partitioned for parallel computing.
Currently workstations and LSF clusters are supported. SLURM clusters are
an easy addition - if you need this to run on a SLURM cluster `please post a feature request issue
to the github repository <https://github.com/MouseLand/cellpose/issues>`_ and tag @GFleishman.

The input data format must be a `zarr array <https://zarr.readthedocs.io/en/stable/>`_.
Some functions are provided in the module to help convert your data to a zarr array, but
not all formats or situations are covered. These are good opportunities to submit pull requests.
Currently, the module must be run via the Python API, but making it available in the GUI
is another good PR or feature request.

Many images contain large volumes of background - i.e. parts of the image that do not contain
sample. It would be a waste of resources to try and segment in these areas. Distributed Cellpose can
take a foreground mask and will only process blocks that contain foreground. The mask does not have
to be the same sampling rate (resolution) as the input data. It is only assumed that the input data
and mask have the same field of view. So you can give a huge image and a small mask as long as the
physical length of the axes (number of voxels times the voxel size in physical units) is the same.

Preprocessing (for example Gaussian smoothing) can sometimes improve Cellpose performance. But
preprocessing big data can be inconvenient. Distributed Cellpose can take a list of
preprocessing steps, which is a list of functions and their arguments, and it will run those
functions on the blocks before running Cellpose. This distributes any preprocessing steps you
like along with Cellpose itself. This can be used in very creative ways. For example,
currently to perform multi-channel segmentation, you must use preprocessing_steps to provide
the second channel. See the examples below to learn how to do multi-channel segmentation.

All user facing functions in the module have verbose docstrings that explain inputs and outputs.
You can access these docstrings like this:

.. code-block:: python

    from cellpose.contrib.distributed_segmentation import distributed_eval
    distributed_eval?

Examples
~~~~~~~~

Run distributed Cellpose on half the resources of a workstation that has 16 cpus, 1 gpu,
and 128GB system memory:

.. code-block:: python

   from cellpose.contrib.distributed_segmentation import distributed_eval

    # parameterize cellpose however you like
    model_kwargs = {'gpu':True, 'model_type':'cyto3'}  # can also use 'pretrained_model'
    eval_kwargs = {'diameter':30,
                   'z_axis':0,
                   'channels':[0,0],
                   'do_3D':True,
    }
    
    # define compute resources for local workstation
    cluster_kwargs = {
        'n_workers':1,    # if you only have 1 gpu, then 1 worker is the right choice
        'ncpus':8,
        'memory_limit':'64GB',
        'threads_per_worker':1,
    }
    
    # run segmentation
    # outputs:
    #     segments: zarr array containing labels
    #     boxes: list of bounding boxes around all labels (very useful for navigating big data)
    segments, boxes = distributed_eval(
        input_zarr=large_zarr_array,
        blocksize=(256, 256, 256),
        write_path='/where/zarr/array/containing/results/will/be/written.zarr',
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster_kwargs=cluster_kwargs,
    )


Test run a single block before distributing the whole dataset (always a good idea):

.. code-block:: python

    from cellpose.contrib.distributed_segmentation import process_block

    # parameterize cellpose however you like
    model_kwargs = {'gpu':True, 'model_type':'cyto3'}
    eval_kwargs = {'diameter':30,
                   'z_axis':0,
                   'channels':[0,0],
                   'do_3D':True,
    }
    
    # define a crop as the distributed function would
    starts = (128, 128, 128)
    blocksize = (256, 256, 256)
    overlap = 60
    crop = tuple(slice(s-overlap, s+b+overlap) for s, b in zip(starts, blocksize))
    
    # call the segmentation
    segments, boxes, box_ids = process_block(
        block_index=(0, 0, 0),  # when test_mode=True this is just a dummy value
        crop=crop,
        input_zarr=my_zarr_array,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        blocksize=blocksize,
        overlap=overlap,
        output_zarr=None,
        test_mode=True,
    )


Convert a single large (but still smaller than system memory) tiff image to a zarr array:

.. code-block:: python

    # Note full image will be loaded in system memory
    import tifffile
    from cellpose.contrib.distributed_segmentation import numpy_array_to_zarr

    data_numpy = tifffile.imread('/path/to/image.tiff')
    data_zarr = numpy_array_to_zarr('/path/to/output.zarr', data_numpy, chunks=(256, 256, 256))
    del data_numpy  # assumption is data is large, don't keep in memory copy around


Wrap a folder of tiff images/tiles into a single zarr array without duplicating any data:

.. code-block:: python

    # Note tiff filenames must indicate the position of each file in the overall tile grid
    from cellpose.contrib.distributed_segmentation import wrap_folder_of_tiffs

    reconstructed_virtual_zarr_array = wrap_folder_of_tiffs(
        filname_pattern='/path/to/folder/of/*.tiff',
        block_index_pattern=r'_(Z)(\d+)(Y)(\d+)(X)(\d+)',
    )


Run distributed Cellpose on an LSF cluster with 128 GPUs (e.g. Janelia cluster):

.. code-block:: python

    from cellpose.contrib.distributed_segmentation import distributed_eval
    
    # parameterize cellpose however you like
    model_kwargs = {'gpu':True, 'model_type':'cyto3'}
    eval_kwargs = {'diameter':30,
                   'z_axis':0,
                   'channels':[0,0],
                   'do_3D':True,
    }
    
    # define LSFCluster parameters
    cluster_kwargs = {
        'ncpus':2,                # cpus per worker
        'min_workers':8,          # cluster adapts number of workers based on number of blocks
        'max_workers':128,
        'queue':'gpu_l4',         # flags required to specify a gpu job may differ between clusters
        'job_extra_directives':['-gpu "num=1"'],
    }
    
    # run segmentation
    # outputs:
    #     segments: zarr array containing labels
    #     boxes: list of bounding boxes around all labels (very useful for navigating big data)
    segments, boxes = distributed_eval(
        input_zarr=large_zarr_array,
        blocksize=(256, 256, 256),
        write_path='/where/zarr/array/containing/results/will/be/written.zarr',
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster_kwargs=cluster_kwargs,
    )


Use preprocessing_steps and a mask:

.. code-block:: python

   from scipy.ndimage import gaussian_filter
   from cellpose.contrib.distributed_segmentation import distributed_eval

    # parameterize cellpose however you like
    model_kwargs = {'gpu':True, 'model_type':'cyto3'}  # can also use 'pretrained_model'
    eval_kwargs = {'diameter':30,
                   'z_axis':0,
                   'channels':[0,0],
                   'do_3D':True,
    }
    
    # define compute resources for local workstation
    cluster_kwargs = {
        'n_workers':1,    # if you only have 1 gpu, then 1 worker is the right choice
        'ncpus':8,
        'memory_limit':'64GB',
        'threads_per_worker':1,
    }

    # create preprocessing_steps
    # Note : for any pp step, the first parameter must be image and the last must be crop
    #        you can have any number of other parameters in between them
    def pp_step_one(image, sigma, crop):
        return gaussian_filter(image, sigma)

    # You can sneak other big datasets into the distribution through pp steps
    # the crop parameter contains the slices you need to get the correct block
    def pp_step_two(image, crop):
        return image - background_channel_zarr[crop] # make sure the other dataset is also zarr

    # finally, put all preprocessing steps together
    preprocessing_steps = [(pp_step_one, {'sigma':2.0}), (pp_step_two, {}),]
    
    # run segmentation
    # outputs:
    #     segments: zarr array containing labels
    #     boxes: list of bounding boxes around all labels (very useful for navigating big data)
    segments, boxes = distributed_eval(
        input_zarr=large_zarr_array,
        blocksize=(256, 256, 256),
        write_path='/where/zarr/array/containing/results/will/be/written.zarr',
        preprocessing_steps=preprocessing_steps,
        mask=mask,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster_kwargs=cluster_kwargs,
    )


Multi-channel segmentation using preprocessing_steps:

.. code-block:: python

   from cellpose.contrib.distributed_segmentation import distributed_eval

    # parameterize cellpose however you like
    model_kwargs = {'gpu':True, 'model_type':'cyto3'}  # can also use 'pretrained_model'
    eval_kwargs = {'diameter':30,
                   'z_axis':0,
                   'channels':[2,1],  # two channels along first axis
                   'do_3D':True,
    }

    # define compute resources for local workstation
    cluster_kwargs = {
        'n_workers':1,    # if you only have 1 gpu, then 1 worker is the right choice
        'ncpus':8,
        'memory_limit':'64GB',
        'threads_per_worker':1,
    }

    # preprocessing step to stack second channel onto first
    def stack_channels(image, crop):
        return np.stack((image, second_channel_zarr[crop]), axis=1) # second channel is also a zarr array
    preprocessing_steps = [(stack_channels, {}),]

    # run segmentation
    # outputs:
    #     segments: zarr array containing labels
    #     boxes: list of bounding boxes around all labels (very useful for navigating big data)
    segments, boxes = distributed_eval(
        input_zarr=large_zarr_array,
        blocksize=(256, 256, 256),
        write_path='/where/zarr/array/containing/results/will/be/written.zarr',
        preprocessing_steps=preprocessing_steps,  # sneaky multi-channel segmentation
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster_kwargs=cluster_kwargs,
    )

