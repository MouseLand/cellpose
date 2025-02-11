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
Currently workstations (your own machine) and LSF clusters are supported. SLURM clusters are
an easy addition - if you need this to run on a SLURM cluster `please post a feature request issue
to the github repository<https://github.com/MouseLand/cellpose/issues>`_ and tag @GFleishman.

The input data format must be a zarr array. Some functions are provided in the module to help
convert your data to a zarr array, but not all formats or situations are covered. These are
good opportunities to submit pull requests. Currently, the module must be run via the Python API,
but making it available in the GUI is another good PR or feature request.

Examples
~~~~~~~~

Run distributed Cellpose on half the resources of a workstation with 16 cpus, 1 gpu, and 128GB system memory:
............................

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


