In a notebook 
-----------------------

See :ref:`Settings` for more information on run settings.

::

    import numpy as np
    import matplotlib.pyplot as plt
    from cellpose import models, io
    from cellpose.io import imread 

    io.logger_setup()

    # model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
    model = models.Cellpose(model_type='cyto3')

    # list of files
    # PUT PATH TO YOUR FILES HERE!
    files = ['/media/carsen/DATA1/TIFFS/onechan.tif']

    imgs = [imread(f) for f in files]
    nimg = len(imgs)

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    channels = [[0,0]]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended) 
    # diameter can be a list or a single number for all images
    
    masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)


    ### or to run one of the other models, or a custom model, specify a CellposeModel 
    model = models.CellposeModel(model_type='livecell_cp3')

    masks, flows, styles = model.eval(imgs, diameter=30, channels=[0,0])

See example notebook at `run_cellpose.ipynb`_. 

.. _run_cellpose.ipynb: https://nbviewer.jupyter.org/github/MouseLand/cellpose/blob/master/notebooks/run_cellpose.ipynb
