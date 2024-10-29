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

In a notebook, you can train with the `train_seg` function:
::
    from cellpose import io, models, train
    io.logger_setup()
    
    output = io.load_train_test_data(train_dir, test_dir, image_filter="_img",
                                    mask_filter="_masks", look_one_level_down=False)
    images, labels, image_names, test_images, test_labels, image_names_test = output

    # e.g. retrain a Cellpose model
    model = models.CellposeModel(model_type="cyto3")
    
    model_path, train_losses, test_losses = train.train_seg(model.net, 
                                train_data=images, train_labels=labels,
                                channels=[1,2], normalize=True,
                                test_data=test_images, test_labels=test_labels,
                                weight_decay=1e-4, SGD=True, learning_rate=0.1,
                                n_epochs=100, model_name="my_new_model")


CLI training options
~~~~~~~~~~~~~~~~~~~~

::

    --train               train network using images in dir
    --train_size          train size network at end of training
    --test_dir TEST_DIR   folder containing test data (optional)
    --mask_filter MASK_FILTER
                            end string for masks to run on. use '_seg.npy' for
                            manual annotations from the GUI. Default: _masks
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
    --SGD SGD             use SGD
    --save_every SAVE_EVERY
                            number of epochs to skip between saves. Default: 100
    --model_name_out MODEL_NAME_OUT
                            Name of model to save as, defaults to name describing
                            model architecture. Model is saved in the folder
                            specified by --dir in models subfolder.


Re-training a model 
~~~~~~~~~~~~~~~~~~~

We find that for re-training, using SGD generally works better, and it is the default in the GUI. 
The options in the code above are the default options for retraining in the GUI and in the Cellpose 2.0 paper
``(weight_decay=1e-4, SGD=True, learning_rate=0.1, n_epochs=100)``, 
although in the paper we often use 300 epochs instead of 100 epochs, and it may help to use more epochs, 
especially when you have more training data.

When re-training, keep in mind that the normalization happens per image that you train on, and often these are image crops from full images. 
These crops may look different after normalization than the full images. To approximate per-crop normalization on the full images, we have the option for 
tile normalization that can be set in ``model.eval``: ``normalize={"tile_norm_blocksize": 128}``. Alternatively/additionally, you may want to change 
the overall normalization scaling on the full images, e.g. ``normalize={"percentile": [3, 98]``. You can visualize how the normalization looks in 
a notebook for example with ``from cellpose import transforms; plt.imshow(transforms.normalize99(img, lower=3, upper=98))``. The default 
that will be used for training on the image crops is ``[1, 99]``. 

See :ref:`do3d` for info on training on 3D data.