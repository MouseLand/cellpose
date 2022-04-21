Inputs
-------------------------------

You can use tiffs or PNGs or JPEGs. We use the image loader from scikit-image. 
Single plane images can read into data as nY x nX x channels or channels x nY x nX. 
Then the `channels <settings.html#channels>`__ settings will take care of reshaping 
the input appropriately for the network. Note the model also rescales the input for 
each channel so that 0 = 1st percentile of image values and 1 = 99th percentile.

If you want to run multiple images in a directory, use the command line or a jupyter notebook to run cellpose.

3D segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tiffs with multiple planes and multiple channels are supported in the GUI (can 
drag-and-drop tiffs) and supported when running in a notebook.
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

When running cellpose in a notebook, set ``do_3D=True`` to enable 3D processing.
You can give a list of 3D inputs, or a single 3D/4D stack.
When running on the command line, add the flag ``--do_3D`` (it will run all tiffs 
in the folder as 3D tiffs if possible). 

If the 3D segmentation is not working well and there is inhomogeneity in Z, try stitching 
masks in Z instead of running ``do_3D=True``. See details for this option here: 
`stitch_threshold <settings.html#d-settings>`__.

If drag-and-drop doesn't work because of the shape of your tiff, 
you need to transpose the tiff and resave to use the GUI, or 
use the napari plugin for cellpose, or run CLI/notebook and 
specify the ``channel_axis`` and/or ``z_axis``
parameters:

  ``channel_axis`` and ``z_axis`` can be used to specify the axis (0-based) 
  of the image which corresponds to the image channels and to the z axis. 
  For example an image with 2 channels of shape (1024,1024,2,105,1) can be 
  specified with ``channel_axis=2`` and ``z_axis=3``. If ``channel_axis=None`` 
  cellpose will try to automatically determine the channel axis by choosing 
  the dimension with the minimal size after squeezing. If ``z_axis=None`` 
  cellpose will automatically select the first non-channel axis of the image 
  to be the Z axis. These parameters can be specified using the command line 
  with ``--channel_axis`` or ``--z_axis`` or as inputs to ``model.eval`` for 
  the ``Cellpose`` or ``CellposeModel`` model.

