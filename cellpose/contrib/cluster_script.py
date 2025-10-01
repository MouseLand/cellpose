from pathlib import Path
import subprocess
import pickle

import imageio
import zarr
import numpy
from tifffile import imwrite
from cellpose.contrib.distributed_segmentation import numpy_array_to_zarr

from cellpose.contrib.distributed_segmentation import distributed_eval
from cellpose.contrib.distributed_segmentation import SlurmCluster, janeliaLSFCluster

## PARAMETERS
# Compute node accessible directory for test input zarr dataset and outputs 
output_dir = Path.home() / 'link_scratch'

# Cluster parameters (here: https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html)
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

input_zarr_path = output_dir / 'input.zarr'
output_zarr_path = output_dir / 'segmentation.zarr'
output_bbox_pkl = output_dir / 'bboxes.pkl'

if not input_zarr_path.exists():
    print('Download (1024 x 1024 x 1024) test data')
    crop = (slice(0,1), slice(2048,3072), slice(2048,3072), slice(0,1024))
    data_numpy = zarr.open("https://webknossos-data.mpinb.mpg.de/data/zarr/653bd498010000ae005914a1/color/16-16-2", mode='r')[crop]

    print('Save as 3D local zarr array')
    data_zarr = numpy_array_to_zarr(input_zarr_path, data_numpy.squeeze(0), chunks=(256, 256, 256))
    del data_numpy
else:
    data_zarr = zarr.open(input_zarr_path)

# parameterize cellpose however you like
model_kwargs = {'gpu':True}
eval_kwargs = {
    'z_axis':0,
    'do_3D':True,
}

# Guess cluster type by checking for cluster submission commands 
if subprocess.getstatusoutput('sbatch -h')[0] == 0:
    print('Slurm sbatch command detected -> use SlurmCluster')
    cluster = SlurmCluster(**cluster_kwargs)
elif subprocess.getstatusoutput('bsub -h')[0] == 0:
    print('LSF bsub command detected -> use janeliaLSFCLuster')
    cluster = janeliaLSFCluster(**cluster_kwargs)
else:
    cluster = None

if cluster is None:
    raise Exception(
        "Neither SLURM nor LFS cluster detected. "
        "Currently, this script only supports SLURM or LSF cluster scheduler. "
        "You have two options:"
        "\n * Either use `distributed_eval` without the `cluster` but with the `cluster_kwargs` argument to start a local cluster on your machine"
        "\n * or raise a feature request at https://github.com/MouseLand/cellpose/issues."
    )

# Start evaluation
segments, boxes = distributed_eval(
    input_zarr=data_zarr,
    blocksize=(256, 256, 256),
    write_path=str(output_zarr_path),
    model_kwargs=model_kwargs,
    eval_kwargs=eval_kwargs,
    cluster = cluster,
)

# Save boxes on disk
with open(output_bbox_pkl, 'wb') as f:
    pickle.dump(boxes, f)

print(f'Segmentation saved in {str(output_zarr_path)}')
print(f'Object boxes saved in {str(output_bbox_pkl)}')
