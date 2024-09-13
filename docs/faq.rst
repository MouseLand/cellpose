FAQ
~~~~~~~~~~~~~~~~~~~~~~~~

**Q: What should I set the** ``--flow_threshold``/``--cellprob_threshold``/``--diameter`` **parameter to?**

    These parameters should be set experimentally by running Cellpose, viewing the results, and tuning the parameters
    to get the best results. The default parameters are set to work well for most images, but may not be optimal
    for your images. See :ref:`Settings` for more information.


**Q: What accuracy is good enough? Is there a quantitative threshold that should be met before implementing a model?**

    Generally speaking you want to meet or exceed the accuracy of a human. You can estimate human accuracy by labeling
    the same image twice and evaluating accuracy metrics. In practice human accuracy is often lower than you would
    expect. You can see our results from this analysis in our Cellpose 2
    `paper <https://www.biorxiv.org/content/10.1101/2022.04.01.486764v1>`_.

    Some additional information on precision and accuracy can be found `here <https://forum.image.sc/t/how-to-interpret-cellposes-average-precision-model-evaluation-value/75231/3>`_.


**Q: How do I download the pretrained models?**

    The models will be downloaded automatically from the `website <https://www.cellpose.org/>`_ when you first run a
    pretrained model in cellpose. If you are having issues with the downloads, you can download them from this
    `google drive zip file <https://drive.google.com/file/d/1zHGFYCqRCTwTPwgEUMNZu0EhQy2zaovg/view?usp=sharing>`_,
    unzip the file and put the models in your home directory under the path ``.cellpose/models/``,
    e.g. on Windows this would be ``C:/Users/YOUR_USERNAME/.cellpose/models/`` or on Linux this would be
    ``/home/YOUR_USERNAME/.cellpose/models/``, so ``/home/YOUR_USERNAME/.cellpose/models/cyto_0`` is the full
    path to one model for example. If you cannot access google drive, the models are also available on
    baidu: https://pan.baidu.com/s/1CARpRGCBHIYaz7KeyoX-fg thanks to @qixinbo!


**Q: How can I use cellpose to recognize different types of cells in the same image?**

    Cellpose does not natively support recognizing different types of cells (aka 'multiclass segmentation').
    However, you can train individual models that are capable of recognizing only a given cell type at a time and run
    Cellpose multiple times on the same image. With sufficient training, the result will be two sets of
    outputs that could be combined in post-processing to identify the different cell types.


**Q: Why does the PNG mask file look dim at the top and light at the bottom? I can't see the cell masks.**

    This is expected and intended behavior, although it is dependent on the image viewer used to view
    the mask file. The mask file is saved with each pixel as background
    (represented by a 0), or as a cell label (represented by the cell label number). The gradient is
    produced because each cell label is unique and monotonically increasing from top to bottom.

    You can use different look up tables (LUTs) in ImageJ to view the resulting masks or threshold everything
    above zero to get everything that cellpose detects. Image post processing is outside the scope
    of cellpose, but you can find additional help at https://forum.image.sc/tag/cellpose.

**Q: The Cellpose GUI is unresponsive/frozen. Is it broken?**

    Cellpose is likely not broken; it is just busy. Currently, the GUI cannot receive input while computing
    segmentation. Cellpose is a fairly computationally intensive program and may take a long time
    to run, depending on computer hardware specifications. Cellpose will take a long time to run on large images.
    Using hardware with a faster CPU and with more available memory will speed up the process. Using a GPU will
    also speed up the process, especially if you are training with a large dataset.
