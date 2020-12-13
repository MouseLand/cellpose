Training
---------------------------

At the beginning of training, cellpose computes the flow field representation for each 
mask image (``dynamics.labels_to_flows``).

The cellpose pretrained models are trained using resized images so that the cells have the same median diameter across all images.
If you choose to use a pretrained model, then this fixed median diameter is used.

If you choose to train from scratch, you can set the median diameter you want to use for rescaling with the ``--diameter`` flag, or set it to 0 to disable rescaling. 
We trained the `cyto` model with a diameter of 30 pixels and the `nuclei` model with a diameter of 17 pixels.

When you rescale everything to 30. pixel diameter, if you have images with varying diameters
you may also want to learn a `SizeModel` that predicts the diameter from the styles that the 
network outputs. Add the flag ``--train_size`` and this model will be trained and saved as an 
``*.npy`` file.

The same channel settings apply for training models (see all Command line `options
<http://www.cellpose.org/static/docs/command.html>`_). 

Note Cellpose expects the labelled masks (0=no mask, 1,2...=masks) in a separate file, e.g:

::

    wells_000.tif
    wells_000_masks.tif

If you use the --img_filter option (``--img_filter img`` in this case):

::

    wells_000_img.tif
    wells_000_masks.tif

.. warning:: 
    The path given to ``--dir`` and ``--test_dir`` must be an absolute path.

Training-specific options

::

    --test_dir TEST_DIR       folder containing test data (optional)
    --n_epochs N_EPOCHS       number of epochs (default: 500)
  
To train on cytoplasmic images (green cyto and red nuclei) starting with a pretrained model from cellpose (cyto or nuclei):

::
    
    python -m cellpose --train --dir ~/images_cyto/train/ --test_dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 1

You can train from scratch as well:

::

    python -m cellpose --train --dir ~/images_nuclei/train/ --pretrained_model None

To train the cyto model from scratch using the same parameters we did, download the dataset and run

::

    python -m cellpose --train --train_size --use_gpu --dir ~/cellpose_dataset/train/ --test_dir ~/cellpose_dataset/test/ --img_filter _img --pretrained_model None --chan 2 --chan2 1


You can also specify the full path to a pretrained model to use:

::

    python -m cellpose --dir ~/images_cyto/test/ --pretrained_model ~/images_cyto/test/model/cellpose_35_0 --save_png

