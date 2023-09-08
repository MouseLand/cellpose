Models
------------------------------

``from cellpose import models``

Cellpose 2.0 now has a model zoo and options for user model training. 
Each model will be downloaded automatically to your ``models.MODELS_DIR`` 
(see Installation instructions for more details on MODELS_DIR). 
See paper for more details on the model zoo. You can also directly download a 
model by going to the URL, e.g.:

``https://www.cellpose.org/models/MODEL_NAME``

Model Zoo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All built-in models were trained with the ROIs resized to a diameter of 30.0
(``diam_mean = 30``), 
except the `'nuclei'` model which was trained with a diameter of 17.0 
(``diam_mean = 17``). 
The models will internally take care of rescaling the images given a 
user-provided diameter (or with the diameter from 
auto-diameter estimation in full models).

There is a suggestion button below the model zoo in the GUI. This runs a ``general`` model 
that has been trained on Cellpose, TissueNet, and LiveCell to obtain the style 
of the image. It uses this style to suggest which model would be best for the 
given image (see info in Cellpose 2.0 `paper <https://www.biorxiv.org/content/10.1101/2022.04.01.486764v1>`_, 
and runs the suggested model on the image. Make sure the diameter is set to the approximate 
diameter of the ROIs in the image before clicking the button to ensure best performance.


Full built-in models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These models have a size model and 4 different training versions, each trained
starting from 4 different random initial parameter sets. This means you can 
run with ``diameter=0`` or ``--diameter 0`` and the model can estimate the ROI size. Also you can set 
``net_avg=True`` or ``--net_avg`` to average the results of the 4 models.

These models can be loaded and used in the notebook with ``models.Cellpose(model_type='cyto')`` 
or in the command line with ``python -m cellpose --pretrained_model cyto``.

These models' names (to download all the models for a class run with ``--net_avg``): 
* `'cyto'`: ``cytotorch_0``, ``cytotorch_1``, ``cytotorch_2``, ``cytotorch_3``, ``size_cytotorch_0.npy``
* `'nuclei'`: ``nucleitorch_0``, ``nucleitorch_1``, ``nucleitorch_2``, ``nucleitorch_3``, ``size_nucleitorch_0.npy``
* `'cyto2'`: ``cyto2torch_0``, ``cyto2torch_1``, ``cyto2torch_2``, ``cyto2torch_3``, ``size_cyto2torch_0.npy``

Cytoplasm model (`'cyto'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cytoplasm model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
an optional nuclear channel. Here are the options for each:
1. 0=grayscale, 1=red, 2=green, 3=blue 
2. 0=None (will set to zero), 1=red, 2=green, 3=blue

Set channels to a list with each of these elements, e.g.
``channels = [0,0]`` if you want to segment cells in grayscale or for single channel images, or
``channels = [2,3]`` if you green cells with blue nuclei.

Nucleus model (`'nuclei'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nuclear model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
always set to an array of zeros. Therefore set the first channel as 
0=grayscale, 1=red, 2=green, 3=blue; and set the second channel to zero, e.g.
``channels = [0,0]`` if you want to segment nuclei in grayscale or for single channel images, or 
``channels = [3,0]`` if you want to segment blue nuclei.

Cytoplasm 2.0 model (`'cyto2'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cytoplasm 2.0 model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
an optional nuclear channel, as the cytoplasm model.

In addition to the training data in our dataset, it was 
trained with user-submitted images.


Other built-in models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These models do not have a size model and 4 different training versions.
If the diameter is set to 0.0, then the model uses the default ``diam_mean`` for the
diameter (``30.0``).

These models can be loaded and used in the notebook with e.g. 
``models.CellposeModel(model_type='tissuenet')`` or ``models.CellposeModel(model_type='LC2')``, 
or in the command line with ``python -m cellpose --pretrained_model tissuenet``.

These models' names are the same as their strings in the GUI.

TissueNet models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``'tissuenet'`` model was trained on all training images from the 
`tissuenet dataset <https://datasets.deepcell.org/>`_. 
These images have a cytoplasm channel and a nuclear channel. The 
other tissuenet models (``'TN1'``, ``'TN2'``, and ``'TN3'``) were trained on subsets 
of the tissuenet dataset that had similar characteristics.

LiveCell models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``'livecell'`` model was trained on all training images from the 
`livecell dataset <https://sartorius-research.github.io/LIVECell/>`_. 
These images only have a cytoplasm channel. The 
other livecell models (``'LC1'``, ``'LC2'``, ``'LC3'``, and ``'LC4'``) were trained on subsets 
of the livecell dataset that had similar characteristics.


User-trained models 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, models are trained with the ROIs resized to a diameter of 30.0
(``diam_mean = 30``) -- this is necessary if you want to start from a pretrained 
cellpose model. If you want to use a different diameter and use pretraining,
we recommend performing training yourself on the cellpose dataset with that diameter so the 
model learns objects at that size. All user-trained models will save the 
``diam_mean`` so it will be loaded automatically along with the model weights.

Each model also saves the ``diam_labels`` which is the mean diameter of the 
ROIs in the training images. This value is auto-loaded into the GUI for use 
with the model, or will be used if the diameter is 0 
(``diameter=0`` or ``--diameter 0``).

These models can be loaded and used in the notebook with e.g. 
``models.CellposeModel(model_type='name_in_gui')``  or with the full path
``models.CellposeModel(pretrained_model='/full/path/to/model')`` . If you trained in the 
GUI, you can automatically use the ``model_type`` argument. If you trained in the 
command line, you need to first add the model to the cellpose path either in the GUI 
in the Models menu, or using the command line:
``python -m cellpose --add_model /full/path/to/model``. 

Or these models can be used in the command line with ``python -m cellpose --pretrained_model name_in_gui`` 
or ``python -m cellpose --pretrained_model /full/path/to/model`` .