Installation
------------------------------

We recommend installing an `Anaconda`_ distribution of Python -- Choose
**Python 3.7** and your operating system. Note you might need to use an
anaconda prompt if you did not add anaconda to the path. From your base
environment (or you can make a new environment) in an anaconda
prompt/command prompt, run

::

   pip install cellpose[gui]

If you want to install without the GUI dependencies, run ``pip install cellpose``.

.. note::
    The first time you run cellpose it downloads the latest available trained 
    model weights from the cellpose website. These models are saved in your home 
    directory in a .cellpose folder, not in the code package.

With an environment file 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively you can use the included environment file (if you'd like a
cellpose-specific environment). This environment file includes all the 
dependencies for using the GUI. Using the environment file is 
**recommended** if you have problems with the pip. 
Please follow these instructions:

1. Download the `environment.yml`_ file from the repository. You can
   do this by cloning the repository, or copy-pasting the text from the
   file into a text document on your local computer.
2. Open an anaconda prompt / command prompt with ``conda`` for **python
   3** in the path
3. Change directories to where the ``environment.yml`` is and run
   ``conda env create -f environment.yml``
4. To activate this new environment, run ``conda activate cellpose``
5. You should see ``(cellpose)`` on the left side of the terminal line.
   Now run ``python -m cellpose`` and you're all set.

To upgrade cellpose (package `here`_), run the following in the
environment:

::

   pip install cellpose --upgrade

If you have an older ``cellpose`` environment you can remove it with
``conda env remove -n cellpose`` before creating a new one.

.. note:: 
    Now you will always have to run **conda activate cellpose** before you
    run cellpose. If you want to run jupyter notebooks in this environment,
    then also ``conda install jupyter``.

Common issues
~~~~~~~~~~~~~~~~~~~~~~~

If you receive the error: ``Illegal instruction (core dumped)``, then
likely mxnet does not recognize your MKL version. Please uninstall and
reinstall mxnet without mkl:

::

   pip uninstall mxnet-mkl
   pip uninstall mxnet
   pip install mxnet==1.4.0

If you receive the error: ``No module named PyQt5.sip``, then try
uninstalling and reinstalling pyqt5

::

   pip uninstall pyqt5 pyqt5-tools
   pip install pyqt5 pyqt5-tools pyqt5.sip

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

If you are having issues with the graphical interface, make sure you have **python 3.7** and not python 3.8 installed.

If you are on Yosemite Mac OS or earlier, PyQt doesn't work and you won't be able
to use the graphical interface for cellpose. More recent versions of Mac
OS are fine. The software has been heavily tested on Windows 10 and
Ubuntu 18.04, and less well tested on Mac OS. Please post an issue if
you have installation problems.

CUDA version
~~~~~~~~~~~~~~~~~~~~~~

If you plan on running many images, you may want to install a GPU
version of *mxnet*. I recommend using CUDA 10.0 or greater. Follow the
instructions `here <https://mxnet.apache.org/get_started?>`__.

Before installing the GPU version, remove the CPU version:

::

   pip uninstall mxnet-mkl
   pip uninstall mxnet

When upgrading cellpose, you will want to ignore dependencies (so that
mxnet-mkl does not install):

::

   pip install --no-deps cellpose --upgrade

**Installation of github version**

Follow steps from above to install the dependencies. In the github
repository, run ``pip install -e .`` and the github version will be
installed. If you want to go back to the pip version of cellpose, then
say ``pip install cellpose``.


Dependencies
~~~~~~~~~~~~~~~~~~~~~~

cellpose relies on the following excellent packages (which are
automatically installed with conda/pip if missing):

-  `mxnet_mkl`_
-  `pyqtgraph`_
-  `PyQt5`_
-  `numpy`_ (>=1.16.0)
-  `numba`_
-  `scipy`_
-  `scikit-image`_
-  `natsort`_
-  `matplotlib`_

.. _Anaconda: https://www.anaconda.com/download/
.. _environment.yml: https://github.com/MouseLand/cellpose/blob/master/environment.yml?raw=true
.. _here: https://pypi.org/project/cellpose/

.. _mxnet_mkl: https://mxnet.apache.org/
.. _pyqtgraph: http://pyqtgraph.org/
.. _PyQt5: http://pyqt.sourceforge.net/Docs/PyQt5/
.. _numpy: http://www.numpy.org/
.. _numba: http://numba.pydata.org/numba-doc/latest/user/5minguide.html
.. _scipy: https://www.scipy.org/
.. _scikit-image: https://scikit-image.org/
.. _natsort: https://natsort.readthedocs.io/en/master/
.. _matplotlib: https://matplotlib.org/