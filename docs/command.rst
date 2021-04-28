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

    * chan: (int)
        0 = grayscale; 1 = red; 2 = green; 3 = blue 

    * chan2: (int)
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

    * fast_mode: FLAG
        make code run faster by turning off augmentations and 4 network averaging

    * all_channels: FLAG 
        run cellpose on all image channels (use for custom models ONLY)

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
        
    usage: __main__.py [-h] [--check_mkl] [--mkldnn] [--train] [--dir DIR]
                   [--img_filter IMG_FILTER] [--use_gpu] [--do_3D]
                   [--pretrained_model PRETRAINED_MODEL] [--chan CHAN]
                   [--chan2 CHAN2] [--all_channels] [--diameter DIAMETER]
                   [--flow_threshold FLOW_THRESHOLD]
                   [--cellprob_threshold CELLPROB_THRESHOLD] [--save_png]
                   [--save_tif] [--fast_mode] [--no_npy]
                   [--mask_filter MASK_FILTER] [--test_dir TEST_DIR]
                   [--learning_rate LEARNING_RATE] [--n_epochs N_EPOCHS]
                   [--batch_size BATCH_SIZE]

    cellpose parameters

    optional arguments:
    -h, --help            show this help message and exit
    --check_mkl           check if mkl working
    --mkldnn              force MXNET_SUBGRAPH_BACKEND = "MKLDNN"
    --train               train network using images in dir (not yet
                            implemented)
    --dir DIR             folder containing data to run or train on
    --img_filter IMG_FILTER
                            end string for images to run on
    --use_gpu             use gpu if mxnet with cuda installed
    --do_3D               process images as 3D stacks of images (nplanes x nchan
                            x Ly x Lx
    --pretrained_model PRETRAINED_MODEL
                            model to use
    --chan CHAN           channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE
    --chan2 CHAN2         nuclear channel (if cyto, optional); 0: NONE, 1: RED,
                            2: GREEN, 3: BLUE
    --all_channels        use all channels in image if using own model and
                            images with special channels
    --diameter DIAMETER   cell diameter, if 0 cellpose will estimate for each
                            image
    --flow_threshold FLOW_THRESHOLD
                            flow error threshold, 0 turns off this optional QC
                            step
    --cellprob_threshold CELLPROB_THRESHOLD
                            cell probability threshold, centered at 0.0
    --save_png            save masks as png and outlines as text file for ImageJ
    --save_tif            save masks as tif and outlines as text file for ImageJ
    --fast_mode           make code run faster by turning off augmentations and
                            4 network averaging
    --no_npy              suppress saving of npy
    --mask_filter MASK_FILTER
                            end string for masks to run on
    --test_dir TEST_DIR   folder containing test data (optional)
    --learning_rate LEARNING_RATE
                            learning rate
    --n_epochs N_EPOCHS   number of epochs
    --batch_size BATCH_SIZE
                            batch size