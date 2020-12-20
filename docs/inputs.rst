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
Multiplane images should read into data as nplanes x channels x nY x nX or as 
nplanes x nY x nX. You can test this by running in python 

::

    import skimage.io
    data = skimage.io.imread('img.tif')
    print(data.shape)

and ensuring that the shape is as described above. If drag-and-drop of the tiff into 
the GUI does not work correctly, then it's likely that the shape of the tiff is 
incorrect. If drag-and-drop works (you can see a tiff with multiple planes), 
then the GUI will automatically run 3D segmentation and display it in the GUI. Watch 
the command line for progress. It is recommended to use a GPU to speed up processing.

When running cellpose in a notebook, set ``do_3D=True`` to enable 3D processing.
You can give a list of 3D inputs, or a single 3D/4D stack.
When running on the command line, add the flag ``--do_3D`` (it will run all tiffs 
in the folder as 3D tiffs). 

If the 3D segmentation is not working well and there is inhomogeneity in Z, try stitching 
masks in Z instead of running ``do_3D=True``. See details for this option here: 
`stitch_threshold <settings.html#d-settings>`__.
