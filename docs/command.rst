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
        cyto = cytoplasm; nuclei = nucleus

    * diameter: (float)
        average diameter of objects in image, if 0 cellpose will estimate for each image, default is 30

    * use_gpu: (bool)
        run network on GPU

    * save_png: FLAG
        save masks as png

    * all_channels: FLAG 
        run cellpose on all image channels (use for custom models ONLY)

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

You can run the help string and see all the options:

::
        
    usage: __main__.py [-h] [--train] [--dir DIR] [--img_filter IMG_FILTER]
                    [--use_gpu] [--pretrained_model PRETRAINED_MODEL]
                    [--chan CHAN] [--chan2 CHAN2] [--all_channels]
                    [--diameter DIAMETER] [--save_png]
                    [--mask_filter MASK_FILTER] [--test_dir TEST_DIR]
                    [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]