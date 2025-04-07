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

   python -m cellpose --dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 3 --save_png

You can specify the diameter for all the images or set to 0 if you want
the algorithm to estimate it on an image by image basis. Here is how to
run on nuclear data (grayscale) where the diameter is automatically
estimated:

::

   python -m cellpose --dir ~/images_nuclei/test/ --pretrained_model nuclei --diameter 0. --save_png

.. warning:: 
    The path given to ``--dir`` must be an absolute path.
