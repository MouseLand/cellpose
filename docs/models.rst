Model
-------------------------------

``from cellpose import models``

The cpsam model weights will be downloaded automatically to your ``models.MODELS_DIR`` (see
Installation instructions for more details on MODELS_DIR). You can also directly
download the model by going to the URL, e.g.:

``https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam``

This model was trained on images with a range of diameters from 7.5 to 120 pixels. 
If your images have even larger diameters, you may want to specify the diameter parameter, 
e.g. specifying ``diameter=60`` with downsample the image by a factor of 2 
(downsampling is respect to 30).


User-trained models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, models are trained with the images and ROIs not resized, and expects that 
the testing images will have a similar diameter distribution as the training data.

These models can be loaded and used in the notebook with e.g.
``models.CellposeModel(pretrained_model='name_in_gui')``  or with the full path
``models.CellposeModel(pretrained_model='/full/path/to/model')`` . If
you trained in the command line, you can add the model to the cellpose
path either in the GUI in the Models menu, or using the command line: ``python
-m cellpose --add_model /full/path/to/model``.

Or these models can be used in the command line with ``python -m cellpose
--pretrained_model name_in_gui`` or ``python -m cellpose --pretrained_model
/full/path/to/model``.

Finding models on BioImage.IO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`BioImage.IO <https://bioimage.io/>`_ is a repository for sharing AI models,
datasets and tools for bioimage analysis. You may look for Cellpose models on
BioImage.IO Model Zoo by searching for the tag ``cellpose``. To download a
model, click on the model card, click the download icon, and choose "Download by
Weight Format" - "Pytorch State Dict".

Sharing models on BioImage.IO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also share your trained Cellpose models on the BioImage.IO Model Zoo. To
do this, you need to export your model in the BioImage.IO format using
``cellpose/export.py`` and then upload the packaged model to the Model Zoo.

Detailed steps:

1. Train a Cellpose model and check if it works well on your data.
2. Create an environment ``python -m pip install 'cellpose[bioimageio]'`` or
   ``'cellpose[all]'`` if you haven't already. Note that most users installed
   ``'cellpose[gui]'`` without the bioimageio packages.
3. Export the model using ``export.py`` script. Use ``python export.py --help``
   to see the usage, or check the example in `its docstring
   <https://github.com/MouseLand/cellpose/blob/8bc3f628be732a733e923e93c30c11172e564895/cellpose/export.py#L3-L38>`_.
   In short, you need to name your models, specify if the model runs on
   cytoplasm/nuclei/both, and provide:

   1. a model filepath,
   2. a README.md filepath,
   3. a cover image filepath(s),
   4. a short description string,
   5. a license name like ``MIT``,
   6. a link to your GitHub repo (or Cellpose repo),
   7. information about the authors and what to cite,
   8. tags including ``cellpose``, ``2d`` and ``3d`` (Cellpose models handle
      both).

4. If you are updating a uploaded model, you should also specify the model ID
   and icon. Don't forget to increment the version number.
5. Go to `BioImage.IO <https://bioimage.io/>`_, click "Upload", and follow the
   instructions there.
