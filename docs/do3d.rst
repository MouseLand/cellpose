.. _do3d:

3D segmentation
------------------------------------

Input format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tiffs with multiple planes and multiple channels are supported in the GUI (can 
drag-and-drop tiffs) and supported when running in a notebook.
To open the GUI with z-stack support, use ``python -m cellpose --Zstack``. 
Multiplane images should be of shape nplanes x channels x nY x nX or as 
nplanes x nY x nX. You can test this by running in python 

::

    import tifffile
    data = tifffile.imread('img.tif')
    print(data.shape)

If drag-and-drop of the tiff into 
the GUI does not work correctly, then it's likely that the shape of the tiff is 
incorrect. If drag-and-drop works (you can see a tiff with multiple planes), 
then the GUI will automatically run 3D segmentation and display it in the GUI. Watch 
the command line for progress. It is recommended to use a GPU to speed up processing.

In the CLI/notebook, you need to specify the ``z_axis`` and the ``channel_axis``
parameters to specify the axis (0-based) of the image which corresponds to the image channels and to the z axis. 
For example an image with 2 channels of shape (1024,1024,2,105,1) can be 
specified with ``channel_axis=2`` and ``z_axis=3``. These parameters can be specified using the command line 
with ``--channel_axis`` or ``--z_axis`` or as inputs to ``model.eval`` for 
the ``CellposeModel`` model.

Volumetric stacks do not always have the same sampling in XY as they do in Z. 
Therefore you can set an ``anisotropy`` parameter in CLI/notebook to allow for differences in 
sampling, e.g. set to 2.0 if Z is sampled half as dense as X or Y, and then in the algorithm 
Z is upsampled by 2x.

Segmentation settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default segmentation in the GUI is 2.5D segmentation, where the flows are computed 
on each YX, ZY and ZX slice and then averaged, and then the dynamics are run in 3D.
Specify this segmentation format in the notebook with ``do_3D=True`` or in the CLI with ``--do_3D``
(with the CLI it will segment all tiffs in the folder as 3D tiffs if possible).

If you see many cells that are fragmented, you can smooth the flows before the dynamics 
are run in 3D using the ``flow3D_smooth`` parameter, which specifies the standard deviation of 
a Gaussian for smoothing the flows. The default is 0.0, which means no smoothing. Alternatively/additionally,
you may want to train a model on 2D slices from your 3D data to improve the segmentation (see below).

The network can rescale images using the user diameter and the model ``diam_mean`` (30),
so for example if you input a diameter of 90, 
then the image will be downsampled by a factor of 3, which will increase run speed.
However, the new Cellpose-SAM model is invariant to diameter, so this is optional.

3D segmentation ignores the ``flow_threshold`` because we did not find that
it helped to filter out false positives in our test 3D cell volume. Instead, 
we found that setting ``min_size`` is a good way to remove false positives. 
Note that ``min_size`` applies per slice when ``stitch_threshold`` is used, 
you will need to remove masks afterwards if you have a 3D minimum size to apply.

There may be additional differences in YZ and XZ slices 
that make them unable to be used for 3D segmentation. 
I'd recommend viewing the volume in those dimensions if 
the segmentation is failing, using the orthoviews (activate in the bottom left of the GUI). 
In those instances, you may want to turn off 
3D segmentation (``do_3D=False``) and run instead with ``stitch_threshold>0``. 
Cellpose will create ROIs in 2D on each XY slice and then stitch them across 
slices if the IoU between the mask on the current slice and the next slice is 
greater than or equal to the ``stitch_threshold``.


Training for 3D segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create image crops from z-stacks (in YX, YZ and XZ) using the script ``cellpose/gui/make_train.py``. 
If you have anisotropic volumes, then set the ``--anisotropy`` flag to the ratio between pixel size in Z and in YX, 
e.g. set ``--anisotropy 5`` for pixel size of 1.0 um in YX and 5.0 um in Z. Now you can 
drag-and-drop an image from the folder into the GUI and start to re-train a model 
by labeling your crops and using the ``Train`` option in the GUI (see the 
Cellpose2 tutorial for more advice). 

See the help message for more information:

::
    
    python cellpose\gui\make_train.py --help
    usage: make_train.py [-h] [--dir DIR] [--image_path IMAGE_PATH] [--look_one_level_down] [--img_filter IMG_FILTER]
                        [--channel_axis CHANNEL_AXIS] [--z_axis Z_AXIS] [--chan CHAN] [--chan2 CHAN2] [--invert]
                        [--all_channels] [--anisotropy ANISOTROPY] [--sharpen_radius SHARPEN_RADIUS]
                        [--tile_norm TILE_NORM] [--nimg_per_tif NIMG_PER_TIF] [--crop_size CROP_SIZE]

    cellpose parameters

    options:
    -h, --help            show this help message and exit

    input image arguments:
    --dir DIR             folder containing data to run or train on.
    --image_path IMAGE_PATH
                            if given and --dir not given, run on single image instead of folder (cannot train with this
                            option)
    --look_one_level_down
                            run processing on all subdirectories of current folder
    --img_filter IMG_FILTER
                            end string for images to run on
    --channel_axis CHANNEL_AXIS
                            axis of image which corresponds to image channels
    --z_axis Z_AXIS       axis of image which corresponds to Z dimension
    --chan CHAN           channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: 0
    --chan2 CHAN2         nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: 0
    --invert              invert grayscale channel
    --all_channels        use all channels in image if using own model and images with special channels
    --anisotropy ANISOTROPY
                            anisotropy of volume in 3D

    algorithm arguments:
    --sharpen_radius SHARPEN_RADIUS
                            high-pass filtering radius. Default: 0.0
    --tile_norm TILE_NORM
                            tile normalization block size. Default: 0
    --nimg_per_tif NIMG_PER_TIF
                            number of crops in XY to save per tiff. Default: 10
    --crop_size CROP_SIZE
                            size of random crop to save. Default: 512