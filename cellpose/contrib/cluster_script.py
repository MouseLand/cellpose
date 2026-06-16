from pathlib import Path
import subprocess
import pickle

import zarr
from tifffile import imread
from pooch import retrieve
from cellpose.contrib.distributed_segmentation import numpy_array_to_zarr

from cellpose.contrib.distributed_segmentation import distributed_eval
from cellpose.contrib.distributed_segmentation import SlurmCluster, janeliaLSFCluster

def main():
    ## PARAMETERS
    # Compute node-accessible directory for input zarr dataset and outputs 
    output_dir = Path() / 'outputs'
    input_zarr_path = output_dir / 'input.zarr'
    output_zarr_path = output_dir / 'segmentation.zarr'
    output_bbox_pkl = output_dir / 'bboxes.pkl'

    # Cluster parameters (example: https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html)
    cluster_kwargs = {
        'job_cpu': 2,               # number of CPUs per GPU worker
        'ncpus':1,                  # threads requested per GPU worker
        'min_workers':1,            # min number of workers based on expected workload
        'max_workers':16,           # max number of workers based on expected workload 
        'walltime': '1:00:00',      # available runtime for each GPU worker for cluster scheduler (Slurm, LSF)
        'queue': 'apu',             # queue/ partition name for single GPU worker *
        'interface': 'ib0',         # interface name for compute-node communication *
        'local_directory': '/tmp',  # compute node local temporary directory *
        'job_extra_directives': [   # extra directives for scheduler (here: Slurm) *
            '--constraint apu',
            '--gres gpu:1',
        ],
    }
    # * Ask your cluster support staff for assistance

    # Cellpose parameters
    model_kwargs = {'gpu':True}
    eval_kwargs = {
        'z_axis':0,
        'do_3D':True,
    }

    # Optional: Crop data to reduce runtime for this test case
    crop = (slice(0, 221), slice(1024,2048), slice(1024,2048))


    ## DATA PREPARATION
    # here: DAPI-stained human gastruloid by Zhiyuan Yu (https://zenodo.org/records/17590053)
    if not input_zarr_path.exists():
        print('Download test data')
        fname = retrieve(
            url="https://zenodo.org/records/17590053/files/2d_gastruloid.tif?download=1",
            known_hash="8ac2d944882268fbaebdfae5f7c18e4d20fdab024db2f9f02f4f45134b936872",
            path = Path.home() / '.cellpose' / 'data',
            progressbar=True,
        )       
        data_numpy = imread(fname)[crop]

        print(f'Convert to {data_numpy.shape} zarr array')
        data_zarr = numpy_array_to_zarr(input_zarr_path, data_numpy, chunks=(256, 256, 256))
        print(f'Input stored in {input_zarr_path}')
        del data_numpy
    else:
        print(f'Read input data from {input_zarr_path}')
        data_zarr = zarr.open(input_zarr_path)

    ## EVALUATION
    # Guess cluster type by checking for cluster submission commands 
    if subprocess.getstatusoutput('sbatch -h')[0] == 0:
        print('Slurm sbatch command detected -> use SlurmCluster')
        cluster = SlurmCluster(**cluster_kwargs)
    elif subprocess.getstatusoutput('bsub -h')[0] == 0:
        print('LSF bsub command detected -> use janeliaLSFCLuster')
        cluster = janeliaLSFCluster(**cluster_kwargs)
    else:
        cluster = None
        ## Note in case you want to test without a cluster scheduler use:
        #from cellpose.contrib.distributed_segmentation import myLocalCluster
        #cluster = myLocalCluster(**{
        #    'n_workers': 1,            # if you only have 1 gpu, then 1 worker is the right choice
        #    'ncpus': 8,                
        #    'memory_limit':'64GB',
        #    'threads_per_worker':1,
        #})

    if cluster is None:
        raise Exception(
            "Neither SLURM nor LFS cluster detected. "
            "Currently, this script only supports SLURM or LSF cluster scheduler. "
            "You have two options:"
            "\n * Either use `distributed_eval` without the `cluster` but with the `cluster_kwargs` argument to start a local cluster on your machine"
            "\n * or raise a feature request at https://github.com/MouseLand/cellpose/issues."
        )

    # Start computation
    segments, boxes = distributed_eval(
        input_zarr = data_zarr,
        blocksize = (256, 256, 256),
        write_path = str(output_zarr_path),
        model_kwargs = model_kwargs,
        eval_kwargs = eval_kwargs,
        cluster = cluster,
    )

    # Save bounding boxes on disk
    with open(output_bbox_pkl, 'wb') as f:
        pickle.dump(boxes, f)

    print(f'Segmentation saved in {str(output_zarr_path)}')
    print(f'Object bounding boxes saved in {str(output_bbox_pkl)}')

if __name__ == '__main__':
    main()