In a notebook 
-----------------------

See :ref:`Settings` for more information on run settings.

::

    import numpy as np
    import matplotlib.pyplot as plt
    from cellpose import models, io
    from cellpose.io import imread 

    io.logger_setup()

    model = models.CellposeModel(gpu=True)

    # list of files
    # PUT PATH TO YOUR FILES HERE!
    files = ['/media/carsen/DATA1/TIFFS/onechan.tif']

    imgs = [imread(f) for f in files]
    nimg = len(imgs)

    masks, flows, styles = model.eval(imgs)

See example notebook at `run_Cellpose-SAM.ipynb`_. 

.. _run_cellpose.ipynb: https://github.com/MouseLand/cellpose/blob/main/notebooks/run_Cellpose-SAM.ipynb