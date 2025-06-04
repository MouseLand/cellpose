.. _Settings:

Settings
--------------------------

The important settings are described on this page. 
See the :ref:`cpmclass` for all run options.

.. warning:: 
    Cellpose 3 used ``models.Cellpose`` class which has been removed in Cellpose 4. Users should
    now only use the ``models.CellposeModel`` class. 

Here is an example of calling the ``CellposeModel`` class and
running a list of images for reference:

::

    from cellpose import models
    from cellpose.io import imread

    model = models.CellposeModel(gpu=True)

    files = ['img0.tif', 'img1.tif']
    imgs = [imread(f) for f in files]
    masks, flows, styles, diams = model.eval(imgs, flow_threshold=0.4, cellprob_threshold=0.0)

Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Channels are no longer an input to Cellpose-SAM - Cellpose-SAM has been trained to be invariant to the order of the channels in your image.
Cellpose-SAM will use the first 3 channels of your image, truncating the rest. It has been trained with three 
different channels for H&E images, and for cellular images it has been trained with the cytoplasm and nuclear channels in any order, 
with the other channel set to zero.

So, if you have two channels, cytoplasm and nuclei, you can put them in in any order. If you have a third channel in fluorescent imaging, 
you will want to omit it from the input, or combine it with the cytoplasm channel, or train a new model with all three inputs, e.g.

::

    from cellpose import models
    from cellpose.io import imread

    model = models.CellposeModel(gpu=True)

    img = imread("img.tif") # tiff is n x 100 x 100 
    
    # if nuclei and cytoplasm are in first two channels 
    img_cp = img[:2] # keep first two channels

    # if nuclei and cytoplasm are in different channels from first two 
    img_cp = img[[1, 3]] # keep 1 and 3 (2nd and 4th channels)

    # if you want to combine two stains to create your "cytoplasm" channel 
    # in this example indices 0 and 2 (1st and 3rd) have two cellular stains 
    # and nuclei are in index 1 (2nd channel)
    img_cp = np.stack((img[[0,2]].sum(axis=0), img[1]), axis=0)

    masks, flows, styles = model.eval(img_cp)
    
.. _diameter:

Diameter 
~~~~~~~~~~~~~~~~~~~~~~~~

Cellpose-SAM been trained on images with ROI diameters from size 7.5 to 120, with a mean diameter of 30 pixels.
Thus the model has a good amount of size-invariance, meaning that specifying the diameter is optional. 
However, if your cells are very big, you may want to use the diameter input to downsample them. For example if you input a diameter of 90, 
then the image will be downsampled by a factor of 3, which will increase run speed.

Flow threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note there is nothing keeping the neural network from predicting 
horizontal and vertical flows that do not correspond to any real 
shapes at all. In practice, most predicted flows are consistent with 
real shapes, because the network was only trained on image flows 
that are consistent with real shapes, but sometimes when the network 
is uncertain it may output inconsistent flows. To check that the 
recovered shapes after the flow dynamics step are consistent with 
real ROIs, we recompute the flow gradients for these putative 
predicted ROIs, and compute the mean squared error between them and
the flows predicted by the network. 

The ``flow_threshold`` parameter is the maximum allowed error of the flows 
for each mask. The default is ``flow_threshold=0.4``. Increase this threshold 
if cellpose is not returning as many ROIs as you'd expect. 
Similarly, decrease this threshold if cellpose is returning too many 
ill-shaped ROIs.

Cellprob threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The network predicts 3 outputs: flows in X, flows in Y, and cell "probability". 
The predictions the network makes of the probability are the inputs to a sigmoid 
centered at zero (1 / (1 + e^-x)), 
so they vary from around -6 to +6. The pixels greater than the 
``cellprob_threshold`` are used to run dynamics and determine ROIs. The default 
is ``cellprob_threshold=0.0``. Decrease this threshold if cellpose is not returning 
as many ROIs as you'd expect. Similarly, increase this threshold if cellpose is 
returning too ROIs particularly from dim areas.

Number of iterations niter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flows from the network are used to simulate a dynamical system governing the 
movements of the pixels. We simulate the dynamics for ``niter`` iterations. 
The pixels that converge to the same position make up a single ROI. The default ``niter=None`` 
or ``niter=0`` sets the number of iterations to be proportional to the ROI diameter.
For longer ROIs, more iterations might be needed, for example ``niter=2000``, for convergence.

For info about 3D data, see :ref:`do3d`.


