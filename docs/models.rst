Models
------------------------------

``from cellpose import models``

Each model will be downloaded automatically to your ``models.MODELS_DIR`` 
(see Installation instructions for more details on MODELS_DIR). 
You can also directly download a model by going to the URL, e.g.:

``https://www.cellpose.org/models/MODEL_NAME``

All built-in models were trained with the ROIs resized to a diameter of 30.0
(``diam_mean = 30``), 
except the `'nuclei'` model which was trained with a diameter of 17.0 
(``diam_mean = 17``). User-trained models will be trained with the same ``diam_mean`` 
as the model they are initalized with.
The models will internally take care of rescaling the images given a 
user-provided diameter (or with the diameter from 
auto-diameter estimation in full models).

Full built-in models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These models have Cellpose model weights and a size model. This means you can 
run with ``diameter=0`` or ``--diameter 0`` and the model can estimate the ROI size. 
However, we recommend that you set the diameter for your ROIs rather than having Cellpose 
guess the diameter.

These models can be loaded and used in the notebook with ``models.Cellpose(model_type='cyto3')`` 
or in the command line with ``python -m cellpose --pretrained_model cyto3``.

We have a ``nuclei`` model and a super-generalist ``cyto3`` model. There are also two 
older models, ``cyto``, which is trained on only the Cellpose training set, and ``cyto2``,
which is also trained on user-submitted images.

FYI we are no longer using the 4 different versions and ``--net_avg`` is deprecated.

Cytoplasm model (``'cyto3'``, ``'cyto2'``, ``'cyto'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cytoplasm models in cellpose are trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
an optional nuclear channel. Here are the options for each:
1. 0=grayscale, 1=red, 2=green, 3=blue 
2. 0=None (will set to zero), 1=red, 2=green, 3=blue

Set channels to a list with each of these elements, e.g.
``channels = [0,0]`` if you want to segment cells in grayscale or for single channel images, or
``channels = [2,3]`` if you green cells with blue nuclei.

The `'cyto3'` model is trained on 9 datasets, see the Cellpose3 paper for more details.

Nucleus model (`'nuclei'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nuclear model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
always set to an array of zeros. Therefore set the first channel as 
0=grayscale, 1=red, 2=green, 3=blue; and set the second channel to zero, e.g.
``channels = [0,0]`` if you want to segment nuclei in grayscale or for single channel images, or 
``channels = [3,0]`` if you want to segment blue nuclei.

Other built-in models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main built-in models are dataset-specific models trained on one of the 9 datasets 
in the Cellpose3 paper. These models do not have a size model.
If the diameter is set to 0.0, then the model uses the default ``diam_mean`` for the
diameter (``30.0``).

These models can be loaded and used in the notebook with e.g. 
``models.CellposeModel(model_type='tissuenet_cp3')`` or ``models.CellposeModel(model_type='livecell_cp3')``, 
or in the command line with ``python -m cellpose --pretrained_model tissuenet_cp3``.

The dataset-specific models were trained on the training images provided in the following datasets: 
    - ``tissuenet_cp3``: `tissuenet dataset <https://datasets.deepcell.org/>`_. 
    - ``livecell_cp3``: `livecell dataset <https://sartorius-research.github.io/LIVECell/>`_
    - ``yeast_PhC_cp3``: `YEAZ dataset <https://www.epfl.ch/labs/lpbs/data-and-software/>`_
    - ``yeast_BF_cp3``: `YEAZ dataset <https://www.epfl.ch/labs/lpbs/data-and-software/>`_
    - ``bact_phase_cp3``: `omnipose dataset <https://osf.io/xmury/>`_
    - ``bact_fluor_cp3``: `omnipose dataset <https://osf.io/xmury/>`_
    - ``deepbacs_cp3``: `deepbacs dataset <https://github.com/HenriquesLab/DeepBacs/wiki/Segmentation>`_
    - ``cyto2_cp3``: `cellpose dataset <http://www.cellpose.org/dataset>`_


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