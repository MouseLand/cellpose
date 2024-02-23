.. _image_restoration:

Image Restoration
=================

The image restoration module ``denoise`` provides functions for restoring degraded images. 
There are two main classes, ``DenoiseModel`` for image restoration only, and 
``CellposeDenoiseModel`` for image restoration and then segmentation. There are three types 
of image restoration provided, denoising, deblurring, and upsampling, and for each of these 
there are two models, one trained on the full ``cyto3`` training set and one trained on 
the ``nuclei`` training set: ``'denoise_cyto3'``, ``'deblur_cyto3'``, ``'upsample_cyto3'``,
``'denoise_nuclei'``, ``'deblur_nuclei'``, ``'upsample_nuclei'``.

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

These models can be used on the command line with input ``--restore_type`` and flag
``--chan2_restore``.
