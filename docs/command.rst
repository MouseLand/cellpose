Command line
------------------------

Input settings
~~~~~~~~~~~~~~~~~~~~~

    * dir: (string)
        directory of images 

    * img_filter: (string)
        (optional) ending of filenames (excluding extension) for processing

Run settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the same :ref:`settings`, but set up for the command line, e.g.
`channels = [chan, chan2]`.

    * chan: (int) channel to segment, ones-based because zero is gray (average of all channels)
        0 = grayscale; 1 = red; 2 = green; 3 = blue 

    * chan2: (int) nuclear or other channel, ones-based because zero means set to all zeros
        (optional); 0 = None (will be set to zero); 1 = red; 2 = green; 3 = blue

    * pretrained_model: (string)
        cyto = cellpose cytoplasm model; nuclei = cellpose nucleus model; can also specify absolute path to model file

    * diameter: (float)
        average diameter of objects in image, if 0 cellpose will estimate for each image, default is 30

    * use_gpu: (bool)
        run network on GPU

    * save_png: FLAG
        save masks as png and outlines as text file for ImageJ

    * save_tif: FLAG
        save masks as tif and outlines as text file for ImageJ

    * no_npy: FLAG 
        turn off saving of _seg.npy file 
    
    * batch_size: (int, optional 8)
        batch size to run tiles of size 224 x 224

Command line examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run ``python -m cellpose`` and specify parameters as below. For instance
to run on a folder with images where cytoplasm is green and nucleus is
blue and save the output as a png (using default diameter 30):

::

   python -m cellpose --dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 3 --save_png

You can specify the diameter for all the images or set to 0 if you want
the algorithm to estimate it on an image by image basis. Here is how to
run on nuclear data (grayscale) where the diameter is automatically
estimated:

::

   python -m cellpose --dir ~/images_nuclei/test/ --pretrained_model nuclei --diameter 0. --save_png

.. warning:: 
    The path given to ``--dir`` must be an absolute path.


Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run the help string and see all the options:

::
    usage: __main__.py [-h] [--use_gpu] [--check_mkl] [--dir DIR]
                   [--look_one_level_down] [--img_filter IMG_FILTER]
                   [--channel_axis CHANNEL_AXIS] [--z_axis Z_AXIS]
                   [--chan CHAN] [--chan2 CHAN2] [--invert] [--all_channels]
                   [--pretrained_model PRETRAINED_MODEL] [--unet]
                   [--nclasses NCLASSES] [--no_resample] [--net_avg]
                   [--no_interp] [--do_3D] [--diameter DIAMETER]
                   [--stitch_threshold STITCH_THRESHOLD] [--fast_mode]
                   [--flow_threshold FLOW_THRESHOLD]
                   [--cellprob_threshold CELLPROB_THRESHOLD]
                   [--anisotropy ANISOTROPY] [--exclude_on_edges] [--save_png]
                   [--save_tif] [--no_npy] [--savedir SAVEDIR] [--dir_above]
                   [--in_folders] [--save_flows] [--save_outlines]
                   [--save_ncolor] [--save_txt] [--train] [--train_size]
                   [--test_dir TEST_DIR] [--mask_filter MASK_FILTER]
                   [--diam_mean DIAM_MEAN] [--learning_rate LEARNING_RATE]
                   [--weight_decay WEIGHT_DECAY] [--n_epochs N_EPOCHS]
                   [--batch_size BATCH_SIZE]
                   [--min_train_masks MIN_TRAIN_MASKS]
                   [--residual_on RESIDUAL_ON] [--style_on STYLE_ON]
                   [--concatenation CONCATENATION] [--save_every SAVE_EVERY]
                   [--save_each] [--verbose]

    cellpose parameters

    optional arguments:
    -h, --help            show this help message and exit
    --verbose             show information about running and settings and save
                            to log

    hardware arguments:
    --use_gpu             use gpu if torch with cuda installed
    --check_mkl           check if mkl working

    input image arguments:
    --dir DIR             folder containing data to run or train on.
    --look_one_level_down
                            run processing on all subdirectories of current folder
    --img_filter IMG_FILTER
                            end string for images to run on
    --channel_axis CHANNEL_AXIS
                            axis of image which corresponds to image channels
    --z_axis Z_AXIS       axis of image which corresponds to Z dimension
    --chan CHAN           channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3:
                            BLUE. Default: 0
    --chan2 CHAN2         nuclear channel (if cyto, optional); 0: NONE, 1: RED,
                            2: GREEN, 3: BLUE. Default: 0
    --invert              invert grayscale channel
    --all_channels        use all channels in image if using own model and
                            images with special channels

    model arguments:
    --pretrained_model PRETRAINED_MODEL
                            model to use for running or starting training
    --unet                run standard unet instead of cellpose flow output
    --nclasses NCLASSES   if running unet, choose 2 or 3; cellpose always uses 3

    algorithm arguments:
    --no_resample         disable dynamics on full image (makes algorithm faster
                            for images with large diameters)
    --net_avg             run 4 networks instead of 1 and average results
    --no_interp           do not interpolate when running dynamics (was default)
    --do_3D               process images as 3D stacks of images (nplanes x nchan
                            x Ly x Lx
    --diameter DIAMETER   cell diameter, if 0 will use the diameter of the
                            training labels used in the model, or with built-in
                            model will estimate diameter for each image
    --stitch_threshold STITCH_THRESHOLD
                            compute masks in 2D then stitch together masks with
                            IoU>0.9 across planes
    --fast_mode           now equivalent to --no_resample; make code run faster
                            by turning off resampling
    --flow_threshold FLOW_THRESHOLD
                            flow error threshold, 0 turns off this optional QC
                            step. Default: 0.4
    --cellprob_threshold CELLPROB_THRESHOLD
                            cellprob threshold, default is 0, decrease to find
                            more and larger masks
    --anisotropy ANISOTROPY
                            anisotropy of volume in 3D
    --exclude_on_edges    discard masks which touch edges of image

    output arguments:
    --save_png            save masks as png and outlines as text file for ImageJ
    --save_tif            save masks as tif and outlines as text file for ImageJ
    --no_npy              suppress saving of npy
    --savedir SAVEDIR     folder to which segmentation results will be saved
                            (defaults to input image directory)
    --dir_above           save output folders adjacent to image folder instead
                            of inside it (off by default)
    --in_folders          flag to save output in folders (off by default)
    --save_flows          whether or not to save RGB images of flows when masks
                            are saved (disabled by default)
    --save_outlines       whether or not to save RGB outline images when masks
                            are saved (disabled by default)
    --save_ncolor         whether or not to save minimal "n-color" masks
                            (disabled by default
    --save_txt            flag to enable txt outlines for ImageJ (disabled by
                            default)

    training arguments:
    --train               train network using images in dir
    --train_size          train size network at end of training
    --test_dir TEST_DIR   folder containing test data (optional)
    --mask_filter MASK_FILTER
                            end string for masks to run on. Default: _masks
    --diam_mean DIAM_MEAN
                            mean diameter to resize cells to during training -- if
                            starting from pretrained models it cannot be changed
                            from 30.0
    --learning_rate LEARNING_RATE
                            learning rate. Default: 0.2
    --weight_decay WEIGHT_DECAY
                            weight decay. Default: 1e-05
    --n_epochs N_EPOCHS   number of epochs. Default: 500
    --batch_size BATCH_SIZE
                            batch size. Default: 8
    --min_train_masks MIN_TRAIN_MASKS
                            minimum number of masks a training image must have to
                            be used. Default: 5
    --residual_on RESIDUAL_ON
                            use residual connections
    --style_on STYLE_ON   use style vector
    --concatenation CONCATENATION
                            concatenate downsampled layers with upsampled layers
                            (off by default which means they are added)
    --save_every SAVE_EVERY
                            number of epochs to skip between saves. Default: 100
    --save_each           save the model under a different filename per
                            --save_every epoch for later comparsion