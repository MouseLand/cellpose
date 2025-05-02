Training
---------------------------

At the beginning of training, cellpose computes the flow field representation for each 
mask image (``dynamics.labels_to_flows``).

.. warning::
    
    You should only start training with the built-in cpsam model, which is the default. 
    When you start training from a built-in model, then you are training 
    the network on all the previously labelled images in the folder and weighting them equally in 
    your training set. 

    If you restart from a previous retraining, you are biasing the network towards the earlier 
    images it has already been trained on. Conversely, if you have created a custom model 
    with different images, and you retrain that model, then you are downweighting the images 
    that you have already trained on and excluded from your new training set. Therefore, we recommend having all images 
    that you want to be trained for the same model in the same folder so they are all used.

By default, models are trained with the images and ROIs not resized, and expects that 
the testing images will have a similar diameter distribution as the training data.

The models will be saved in the image directory (``--dir``) in a folder called ``models/``.

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

  
Here is the recommended training setup for fine-tuning the Cellpose-SAM model:

::
    
    python -m cellpose --train --dir ~/images/train/ --test_dir ~/images/test/ --learning_rate 0.00001 --weight_decay 0.1 --n_epochs 100 --train_batch_size 1



In a notebook, you can train with the `train_seg` function:
::
    from cellpose import io, models, train
    io.logger_setup()
    
    output = io.load_train_test_data(train_dir, test_dir, image_filter="_img",
                                    mask_filter="_masks", look_one_level_down=False)
    images, labels, image_names, test_images, test_labels, image_names_test = output

    model = models.CellposeModel(gpu=True)
    
    model_path, train_losses, test_losses = train.train_seg(model.net, 
                                train_data=images, train_labels=labels,
                                test_data=test_images, test_labels=test_labels,
                                weight_decay=0.1, learning_rate=1e-5,
                                n_epochs=100, model_name="my_new_model")


CLI training options
~~~~~~~~~~~~~~~~~~~~

::

    --train               train network using images in dir
    --test_dir TEST_DIR   folder containing test data (optional)
    --mask_filter MASK_FILTER
                            end string for masks to run on. use '_seg.npy' for
                            manual annotations from the GUI. Default: _masks
    
    --learning_rate LEARNING_RATE
                            learning rate. Default: 1e-5
    --weight_decay WEIGHT_DECAY
                            weight decay. Default: 0.1
    --n_epochs N_EPOCHS   number of epochs. Default: 100
    --train_batch_size TRAIN_BATCH_SIZE
                            batch size for training. Default: 1
    --min_train_masks MIN_TRAIN_MASKS
                            minimum number of masks a training image must have to
                            be used. Default: 5
    --save_every SAVE_EVERY
                            number of epochs to skip between saves. Default: 100
    --model_name_out MODEL_NAME_OUT
                            Name of model to save as, defaults to name describing
                            model architecture. Model is saved in the folder
                            specified by --dir in models subfolder.


Re-training a model 
~~~~~~~~~~~~~~~~~~~

When re-training, keep in mind that the normalization happens per image that you train on, and often these are image crops from full images. 
These crops may look different after normalization than the full images. To approximate per-crop normalization on the full images, we have the option for 
tile normalization that can be set in ``model.eval``: ``normalize={"tile_norm_blocksize": 128}``. Alternatively/additionally, you may want to change 
the overall normalization scaling on the full images, e.g. ``normalize={"percentile": [3, 98]``. You can visualize how the normalization looks in 
a notebook for example with ``from cellpose import transforms; plt.imshow(transforms.normalize99(img, lower=3, upper=98))``. The default 
that will be used for training on the image crops is ``[1, 99]``. 

See :ref:`do3d` for info on training on 3D data.