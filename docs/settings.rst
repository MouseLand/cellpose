Settings
--------------------------

The important settings are described on this page. 
See the :ref:`cpclass` for all run options.

Here is an example of calling the Cellpose class and
running a list of images for reference:
::
    from cellpose import models
    from cellpose.io import imread

    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=False, model_type='cyto')

    files = ['img0.tif', 'img1.tif']
    imgs = [imread(f) for f in files]
    masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0], 
                                             flow_threshold=0.4, do_3D=False)

You can make lists of channels/diameter for each image, or set the same channels/diameter for all images
as shown in the example above.

Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two channels inputs. The first channel is the channel you want to segment. 
The second channel is an optional channel that is helpful in models trained with images 
with a nucleus channel. See more details in the models page.

1. 0=grayscale, 1=red, 2=green, 3=blue 
2. 0=None (will set to zero), 1=red, 2=green, 3=blue

Set channels to a list with each of these elements, e.g.
``channels = [0,0]`` if you want to segment cells in grayscale or for single channel images, or
``channels = [2,3]`` if you green cells with blue nuclei.

On the command line the above would be ``--chan 0 --chan2 0`` or ``--chan 2 --chan2 3``.

Diameter 
~~~~~~~~~~~~~~~~~~~~~~~~

The cellpose models have been trained on images which were rescaled 
to all have the same diameter (30 pixels in the case of the `cyto` 
model and 17 pixels in the case of the `nuclei` model). Therefore, 
cellpose needs a user-defined cell diameter (in pixels) as input, or to estimate 
the object size of an image-by-image basis.

The automated estimation of the diameter is a two-step process using the `style` vector 
from the network, a 64-dimensional summary of the input image. We trained a 
linear regression model to predict the size of objects from these style vectors 
on the training data. On a new image the procedure is as follows.

1. Run the image through the cellpose network and obtain the style vector. Predict the size using the linear regression model from the style vector.
2. Resize the image based on the predicted size and run cellpose again, and produce ROIs. Take the final estimated size as the median diameter of the predicted ROIs.

For automated estimation set ``diameter = None``. 
However, if this estimate is incorrect please set the diameter by hand.

Changing the diameter will change the results that the algorithm 
outputs. When the diameter is set smaller than the true size 
then cellpose may over-split cells. Similarly, if the diameter 
is set too big then cellpose may over-merge cells.

Resample
~~~~~~~~~~~~~~~~~~~~~~~~

The cellpose network is run on your rescaled image -- where the rescaling factor is determined 
by the diameter you input (or determined automatically as above). For instance, if you have 
an image with 60 pixel diameter cells, the rescaling factor is 30./60. = 0.5. After determining 
the flows (dX, dY, cellprob), the model runs the dynamics. The dynamics can be run at the rescaled 
size (``resample=False``), or the dynamics can be run on the resampled, interpolated flows 
at the true image size (``resample=True``). ``resample=True`` will create smoother ROIs when the 
cells are large but will be slower in case; ``resample=False`` will find more ROIs when the cells 
are small but will be slower in this case. By default in versions >=1.0 ``resample=True``.

The nuclear model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
always set to an array of zeros. Therefore set the first channel as 
0=grayscale, 1=red, 2=green, 3=blue; and set the second channel to zero, e.g.
``channels = [0,0]`` if you want to segment nuclei in grayscale or for single channel images, or 
``channels = [3,0]`` if you want to segment blue nuclei.

If the nuclear model isn't working well, try the cytoplasmic model.

Flow threshold (aka model fit threshold in GUI)
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

Mask threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The network predicts 3 outputs: flows in X, flows in Y, and cell "probability". 
The predictions the network makes of the probability are the inputs to a sigmoid 
centered at zero (1 / (1 + e^-x)), 
so they vary from around -6 to +6. The pixels greater than the 
``cellprob_threshold`` are used to run dynamics and determine ROIs. The default 
is ``cellprob_threshold=0.0``. Decrease this threshold if cellpose is not returning 
as many ROIs as you'd expect. Similarly, increase this threshold if cellpose is 
returning too ROIs particularly from dim areas.

3D settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Volumetric stacks do not always have the same sampling in XY as they do in Z. 
Therefore you can set an ``anisotropy`` parameter to allow for differences in 
sampling, e.g. set to 2.0 if Z is sampled half as dense as X or Y. 

There may be additional differences in YZ and XZ slices 
that make them unable to be used for 3D segmentation. 
I'd recommend viewing the volume in those dimensions if 
the segmentation is failing. In those instances, you may want to turn off 
3D segmentation (``do_3D=False``) and run instead with ``stitch_threshold>0``. 
Cellpose will create ROIs in 2D on each XY slice and then stitch them across 
slices if the IoU between the mask on the current slice and the next slice is 
greater than or equal to the ``stitch_threshold``. 

3D segmentation ignores the ``flow_threshold`` because we did not find that
it helped to filter out false positives in our test 3D cell volume. Instead, 
we found that setting ``min_size`` is a good way to remove false positives.





