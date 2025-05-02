.. _image_restoration:

Image Restoration
=================

Image restoration is only implemented in Cellpose3, not available in Cellpose-SAM. To install,

::

    pip install cellpose==3.1.1.2


We introduce image restoration in the `Cellpose3 paper <https://www.biorxiv.org/content/10.1101/2024.02.10.579780v2>`_. 
The image restoration module ``denoise`` provides functions for restoring degraded images. 
There are two main classes, ``DenoiseModel`` for image restoration only, and 
``CellposeDenoiseModel`` for image restoration and then segmentation. There are four types 
of image restoration provided: denoising, deblurring, upsampling and one-click (trained on 
all degradation types). For each of these 
there are three models: one trained on the full ``cyto3`` training set, one trained on the 
``cyto2`` training set, and one trained on the ``nuclei`` training set. Each of these 
models are available on the website as ``https://www.cellpose.org/models/MODEL_NAME``, or will be 
automatically downloaded when you first run the model in the notebook, CLI or GUI: 
``'denoise_cyto3'``, ``'deblur_cyto3'``, ``'upsample_cyto3'``, ``'oneclick_cyto3'``,
``'denoise_cyto2'``, ``'deblur_cyto2'``, ``'upsample_cyto2'``, ``'oneclick_cyto2'``,
``'denoise_nuclei'``, ``'deblur_nuclei'``, ``'upsample_nuclei'``, ``'oneclick_nuclei'``.

Each of the models above were trained with the segmentation loss and perceptual loss. We also make available
the models trained with different loss functions for verifying the results of the paper, 
e.g. ``'denoise_rec_cyto2'`` is the denoising model trained with the reconstruction loss function 
on the ``cyto2`` training set.

DenoiseModel
--------------

Initialize a DenoiseModel with the model_type:

:: 

    from cellpose import denoise
    dn = denoise.DenoiseModel(model_type="denoise_cyto3", gpu=True)

Now you can apply this denoising model to specified channels in your images, 
using the Cellpose channel format (e.g. ``channels=[1,2]``), or leave 
``channels=None`` to apply the model to all channels. Make sure to set the diameter to 
the size of the objects in your image.

::

    imgs_dn = dn.eval(imgs, channels=None, diameter=50.)

If you have two channels, and the second is a nuclear channel, you can specify to use 
the nuclei restoration models on the second channel, with ``chan2=True``:

:: 

    from cellpose import denoise
    dn = denoise.DenoiseModel(model_type="denoise_cyto3", gpu=True, chan2=True)
    imgs_dn = dn.eval(imgs, channels=[1,2], diameter=50.)

The upsampling model ``'upsample_cyto3'`` enables upsampling to diameter of 30., and the 
upsampling model ``'upsample_nuclei'`` enables upsampling to diameter of 17. If you have 
images, for example, in which the objects are of diameter 10, specify that in the 
function call, and then the model will upsample the image to 30 or 17:

:: 

    from cellpose import denoise
    dn = denoise.DenoiseModel(model_type="upsample_cyto3", gpu=True, chan2=True)
    imgs_up = dn.eval(imgs, channels=[1,2], diameter=10.)

For more details refer to the API section.

CellposeDenoiseModel
----------------------

The ``CellposeDenoiseModel`` wraps the CellposeModel and DenoiseModel into one class to 
ensure the channels and diameters are handled properly. See example:

::
    
    from cellpose import denoise
    model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",
                 restore_type="denoise_cyto3", chan2_restore=True)
    masks, flows, styles, imgs_dn = model.eval(imgs, channels=[1,2], diameter=50.)             

For more details refer to the API section.

Command line usage 
---------------------

These models can be used on the command line with model_type input using ``--restore_type`` 
and add flag ``--chan2_restore`` for restoring the optional nuclear channel, e.g.:

::

    python -m cellpose --dir /path/to/images --model_type cyto3 --restore_type denoise_cyto3 --diameter 25 --chan2_restore --chan 2 --chan2 1

Training your own models
--------------------------

It is also possible to train your own models for image restoration using the 
``cellpose.denoise`` module. For example, to train a denoising (Poisson noise) 
model with the cyto2 segmentation model with train_data and train_labels 
(images and ``_flows.tif``):

::

    from cellpose import denoise
    model = denoise.DenoiseModel(gpu=True, nchan=1)

    io.logger_setup()
    model_path = model.train(train_data, train_labels, test_data=None, test_labels=None, 
                            save_path=save_path, iso=True, 
                            blur=0., downsample=0., poisson=0.8, 
                            n_epochs=2000, learning_rate=0.001,
                            seg_model_type="/home/carsen/.cellpose/models/cyto2torch_0")


This training can also be performed on the command line:

::

    python cellpose/denoise.py --dir /path/to/images --noise_type poisson --seg_model_type cyto2 --diam_mean 30.