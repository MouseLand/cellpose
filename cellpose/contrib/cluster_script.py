from pathlib import Path
import subprocess
import pickle

import imageio
import zarr
import numpy
from tifffile import imwrite
from cellpose.contrib.distributed_segmentation import numpy_array_to_zarr

from cellpose.contrib.distributed_segmentation import distributed_eval
from cellpose.contrib.distributed_segmentation import SlurmCluster, JaneliaLSFCluster

## Parameters needs to be modified based 
output_dir = Path.home() / 'link_scratch'

# define cluster parameters
cluster_kwargs = {
    'job_cpu': 2
    'ncpus':1,                  # cpus requested per worker
    'min_workers':1,            # cluster adapts number of workers based on number of blocks
    'max_workers':16,
    'walltime': '1:00:00',      # TODO: find realistic wall time
    'queue': 'GPU',             # Queue name for running (single) GPU jobs -> Ask your local HPC support
    'interface': 'ib0',         # Interface name for compute-node communication -> 
    'local_directory': '/tmp',  # worker local temporary directory -> Ask you local HPC support 
    'job_extra_directives': [
        '--gres gpu:1'
    ],
}

input_zarr_path = output_dir / 'input.zarr'
output_zarr_path = output_dir / 'segmentation.zarr'
output_bbox_pkl = output_dir / 'bboxes.pkl'

if not input_zarr_path.exists():
    print('Download test data (requires internet access)')
    crop = (slice(0,1), slice(2048,3072), slice(2048,3072), slice(0:1024))
    data_numpy = zarr.open("https://webknossos-data.mpinb.mpg.de/data/zarr/653bd498010000ae005914a1/color/16-16-2", mode='r')[crop]

    print('Save as 3D local zarr array')
    data_zarr = numpy_array_to_zarr(input_zarr_path, data_numpy, chunks=(256, 256, 256))
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
    print('sbatch command detected -> use SlurmCluster')
    cluster = SlurmCluster(**cluster_kwargs)
elif subprocess.getstatusoutput('bsub -h')[0] == 0:
    print('bsub command detected -> use JaneliaLSFCLuster')
    cluster = JaneliaLSFCluster(**cluster_kwargs)
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
