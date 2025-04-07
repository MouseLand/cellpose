Inputs
-------------------------------

You can use tiffs or PNGs or JPEGs. We use the image loader from ``tifffile`` or 
``cv2``. Single plane images can be formatted as nY x nX x channels or channels x nY x nX. 
Then the `channels <settings.html#channels>`__ settings will take care of reshaping 
the input appropriately for the network. Note the model also normalizes, by default, the input for 
each channel so that 0 = 1st percentile of image values and 1 = 99th percentile.

If you want to run multiple images in a directory, use the command line or a jupyter notebook to run cellpose.

If you have multiple images of the same size, it can be faster to input them into the 
Cellpose `model.eval` function as an array rather than a list, and running with a large 
batch size. This is because the model can process tiles from multiple images in single batches 
on the GPU if the images are fed in as an array. You can specify the ``channel_axis`` and 
``z_axis`` parameters to specify the axis of the image which corresponds to the image channels
and to the z axis, zero-based. You can also speed this up by increasing the ``batch_size``.

For info about 3D data, see :ref:`do3d`.