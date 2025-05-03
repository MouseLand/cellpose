Timing + memory usage
------------------------------------

Training time and memory usage shown across various GPUs, for 2 labeled images and default 100 epochs:

.. image:: https://www.cellpose.org/static/images/benchmark_train.png
    :width: 800

The inference runtime and memory usage increases with the data size. The runtimes 
shown below are for a single image run once on an RTX 4070S or an A100 with a batch_size of 32.
The runtime will be faster if you run many images of the same size 
input as an array into Cellpose with a large batch_size. The runtimes will also be 
slightly faster if you have fewer cells/cell pixels.

Table for 2D (RTX 4070S):

.. image:: https://www.cellpose.org/static/images/benchmark_2D_4070.png
    :width: 800

Table for 2D (A100):

.. image:: https://www.cellpose.org/static/images/benchmark_2D.png
    :width: 800

Table for 3D (A100):

.. image:: https://www.cellpose.org/static/images/benchmark_3D.png
    :width: 800

If you are running out of GPU memory for your images, you can reduce the 
``batch_size`` parameter in the ``model.eval`` function or in the CLI (default is 8).

If you have even larger images than above, you will want to tile them 
before running Cellpose.