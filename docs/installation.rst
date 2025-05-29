Installation
------------------------------

For basic install instructions, look up the main github readme. 

Built-in model directory
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the pretrained cellpose model(s) is downloaded to ``$HOME/.cellpose/models/``.
This path on linux would look like ``/home/USERNAME/.cellpose/``, and on Windows, 
``C:/Users/USERNAME/.cellpose/models/``. These models are downloaded the first time you 
try to use them, either on the command line, in the GUI or in a notebook.

If you'd like to download the models to a different directory, 
and are using the command line or the GUI, before you run ``python -m cellpose ...``, 
you will need to always set the environment variable ``CELLPOSE_LOCAL_MODELS_PATH`` 
(thanks Chris Roat for implementing this!).

To set the environment variable in the command line/Anaconda prompt on windows run the following command modified for your path:
``set CELLPOSE_LOCAL_MODELS_PATH=C:/PATH_FOR_MODELS/``. To set the environment variable in the command line on 
linux, run ``export CELLPOSE_LOCAL_MODELS_PATH=/PATH_FOR_MODELS/``.

To set this environment variable when running cellpose in a jupyter notebook, run 
this code at the beginning of your notebook before you import cellpose:

::
   
   import os 
   os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/PATH_FOR_MODELS/"

M1-M3 Mac installation
~~~~~~~~~~~~~~~~~~~~~~~

Support for M1-M3 should work out-of-the-box with Cellpose now! Please submit an issue if it's not working.

From the command line you can choose the Mac device explicitly with

::

   python -m cellpose --dir path --gpu_device mps --use_gpu

AMD GPU ROCm installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

As an alternative to the CUDA acceleration for NVIDIA GPUs, you can use the ROCm acceleration for AMD GPUs.
This is not yet supported on Windows, but is supported on Linux. Installation instructions are `available here
<https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.5/page/Introduction_to_ROCm_Installation_Guide_for_Linux.html>`_.
Just like the NVIDIA CUDA installation, you will need to install the ROCm drivers first and then install Cellpose.
Be warned that the ROCm project is significantly less mature than CUDA, and you may run into issues.

.. warning::
   The ROCm acceleration is not yet supported on Windows, and is only supported on Linux.
   If you are on Windows, you will need to use CUDA acceleration.

.. warning::
   ROCm is significantly less mature than the CUDA acceleration, and you may run into issues.


Common issues
~~~~~~~~~~~~~~~~~~~~~~~

If you receive an issue with Qt "xcb", you may need to install xcb libraries, e.g.:

:: 

   sudo apt install libxcb-cursor0
   sudo apt install libxcb-xinerama0

There is also more advice here: https://github.com/NVlabs/instant-ngp/discussions/300.


If you are having issues with CUDA on Windows, or want to use 
Cuda Toolkit 10, please follow these `instructions <https://github.com/MouseLand/cellpose/issues/481#issuecomment-1080137885>`_:

::
   
   conda create -n cellpose pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
   conda activate cellpose
   pip install cellpose

If you receive the error: ``No module named PyQt5.sip``, then try
uninstalling and reinstalling pyqt5

::

   pip uninstall pyqt5 pyqt5-tools
   pip install pyqt5 pyqt5-tools pyqt5.sip

If you are having other issues with the graphical interface and QT, see some advice `here <https://github.com/MouseLand/cellpose/issues/564#issuecomment-1268061118>`_ .

If you have errors related to OpenMP and libiomp5, then try 

::
   
   conda install nomkl

If you receive an error associated with **matplotlib**, try upgrading
it:

::

   pip install matplotlib --upgrade

If you receive the error: ``ImportError: _arpack DLL load failed``, then try uninstalling and reinstalling scipy
::

   pip uninstall scipy
   pip install scipy


If you are on Yosemite Mac OS or earlier, PyQt doesn't work and you won't be able
to use the graphical interface for cellpose. More recent versions of Mac
OS are fine. The software has been heavily tested on Windows 10 and
Ubuntu 18.04, and less well tested on Mac OS. Please post an issue if
you have installation problems.