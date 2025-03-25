Timing + memory usage
------------------------------------

The algorithm runtime and memory usage increases with the data size. The runtimes 
shown below are for a single image run once on an A100 with a batch_size of 32.
This timing includes warm-up of GPU, thus runtimes will be faster for subsequent images. It will also be faster if you run many images of the same size 
input as an array into Cellpose with a large batch_size. The runtimes will also be 
slightly faster if you have fewer cells/cell pixels.

.. image:: https://www.cellpose.org/static/images/benchmark_plot.png
    :width: 600

Table for 2D:

.. image:: https://www.cellpose.org/static/images/benchmark_2d.png
    :width: 800

Table for 3D:

.. image:: https://www.cellpose.org/static/images/benchmark_3d.png
    :width: 800

If you are running out of GPU memory for your images, you can reduce the 
``batch_size`` parameter in the ``model.eval`` function or in the CLI (default is 8).

If you have even larger images than above, you may want to tile them 
before running Cellpose.