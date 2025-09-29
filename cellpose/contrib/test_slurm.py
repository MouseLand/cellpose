from pathlib import Path

import imageio
import zarr
import numpy
from tifffile import imwrite
from cellpose.contrib.distributed_segmentation import numpy_array_to_zarr

input_tif_path = Path('volumedata.tif')
input_zarr_path = Path('volumedata.zarr')
output_tif_path = Path('output.tif')

scratch_dir = Path.home() / 'link_scratch'
output_zarr_path = scratch_dir / 'output.zarr'


if not input_zarr_path.exists():
    print('Test zarr data not found')
    if not input_tif_path.is_file():
        print('Download test data')
        # Just get image data from somewhere. Note that cellpose is probably the best choice for segmenting it
        data_numpy = imageio.imread('https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/volumedata.tif')
        imageio.imwrite(input_tif_path, data_numpy)
    else:
        print('Load test data from disk')
        data_numpy = imageio.imread(input_tif_path)

    print('Convert data to zarr')
    data_zarr = numpy_array_to_zarr(input_zarr_path, data_numpy, chunks=(256, 256, 256))
    del data_numpy
else:
    data_zarr = zarr.open(input_zarr_path)


# Test a single block first
from cellpose.contrib.distributed_segmentation import process_block

# parameterize cellpose however you like
model_kwargs = {'gpu':True}
eval_kwargs = {
    'z_axis':0,
    'do_3D':True,
}

if small_test := False:
    # define a crop as the distributed function would
    starts = (128, 128, 128)
    blocksize = (256, 256, 256)
    overlap = 60
    crop = tuple(slice(s-overlap, s+b+overlap) for s, b in zip(starts, blocksize))

    # call the segmentation
    segments, boxes, box_ids = process_block(
        block_index=(0, 0, 0),  # when test_mode=True this is just a dummy value
        crop=crop,
        input_zarr=data_zarr,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        blocksize=blocksize,
        overlap=overlap,
        output_zarr=None,
        test_mode=True,
    )

    imwrite(output_tif_path, segments, compression ='zlib')

else:
    from cellpose.contrib.distributed_segmentation import distributed_eval, SlurmCluster

    # define cluster parameters
    cluster_kwargs = {
        'ncpus':2,                # cpus per worker
        'min_workers':1,          # cluster adapts number of workers based on number of blocks
        'max_workers':16,
        'queue': 'apu',
        'interface': 'ib0',
        'scratch_dir': str(scratch_dir),
        'walltime': '1:00:00', # TODO: find realistic wall time
        'job_extra_directives':['--constraint apu', '--gres gpu:1'],
        'worker_extra_args': ["--lifetime", "55m", "--lifetime-stagger", "4m"],
        'job_script_prologue' : [
            "module load condainer",
            f"cd {str(Path.home())}/src/cellpose/env",
            "source activate",
            f"cd {str(Path.home())}/src/cellpose",
        ],
    }

    slurm_cluster = SlurmCluster(
        **cluster_kwargs
    )

    # run segmentation
    # outputs:
    #     segments: zarr array containing labels
    #     boxes: list of bounding boxes around all labels (very useful for navigating big data)
    segments, boxes = distributed_eval(
        input_zarr=data_zarr,
        blocksize=(256, 256, 256),
        write_path=str(output_zarr_path),
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster = slurm_cluster,
    )

