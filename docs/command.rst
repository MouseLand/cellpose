Command line
------------------------

The full list of options and what they do can be found on the Command Line Interface (CLI) documentation
page: :ref:`Cellpose CLI`. A description of the most important settings can be found on the :ref:`Settings` page.

.. _Command line examples:

Command Line Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run ``python -m cellpose`` and specify parameters as below. For instance
to run on a folder with images where cytoplasm is green and nucleus is
blue and save the output as a png (using default diameter 30):

::

   python -m cellpose --dir /home/carsen/images_cyto/test/ --save_png

To run on a single 3D image:

:: 
   
   python -m cellpose --image_path /home/carsen/image3D.tif --do_3D --flow3D_smooth 2 --save_tif


.. warning:: 
    The path given to ``--dir`` is recommended to be an absolute path.
