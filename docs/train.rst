Training
---------------------------

At the beginning of training, cellpose computes the flow field representation for each 
mask image (``dynamics.labels_to_flows``).

The cellpose pretrained models are trained using resized images so that the cells have the same median diameter across all images.
If you choose to use a pretrained model, then this fixed median diameter is used.

If you choose to train from scratch, you can set the median diameter you want to use for rescaling with the ``--diam_mean`` flag.
We trained all model zoo models with a diameter of 30.0 pixels, except the `nuclei` model which used a diameter of 17 pixels, 
so if you want to start with a pretrained model, it will default to those values.

The models will be saved in the image directory (``--dir``) in a folder called ``models/``.

The same channel settings apply for training models. 

Note Cellpose expects the labelled masks (0=no mask, 1,2...=masks) in a separate file, e.g:

::

    wells_000.tif
    wells_000_masks.tif

You can use a different ending from ``_masks`` with the ``--mask_filter`` option, e.g. ``--mask_filter _masks_2022``.

Also, you can train a model using the labels from the GUI (``_seg.npy``) by using the following option ``--mask_filter _seg.npy``.

If you use the --img_filter option (``--img_filter _img`` in this case):

::

    wells_000_img.tif
    wells_000_masks.tif

.. warning:: 
    The path given to ``--dir`` and ``--test_dir`` should be an absolute path.

  
To train on cytoplasmic images (green cyto and red nuclei) starting with a pretrained model from cellpose (one of the model zoo models), 
we also have included the recommended training parameters in the command below:

::
    
    python -m cellpose --train --dir ~/images_cyto/train/ --test_dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 1 --learning_rate 0.1 --weight_decay 0.0001 --n_epochs 100

You can train from scratch as well:

::

    python -m cellpose --train --dir ~/images_nuclei/train/ --pretrained_model None

To train the cyto model from scratch using the same parameters we did, download the dataset and run

::

    python -m cellpose --train --train_size --use_gpu --dir ~/cellpose_dataset/train/ --test_dir ~/cellpose_dataset/test/ --img_filter _img --pretrained_model None --chan 2 --chan2 1


You can also specify the full path to a pretrained model to use:

::

    python -m cellpose --dir ~/images_cyto/test/ --pretrained_model ~/images_cyto/test/model/cellpose_35_0 --save_png


Training arguments

::

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

