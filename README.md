# Cellpose <img src="cellpose/logo/logo.png" width="250" title="cellpose" alt="cellpose" align="right" vspace = "50">

[![Documentation Status](https://readthedocs.org/projects/cellpose/badge/?version=latest)](https://cellpose.readthedocs.io/en/latest/?badge=latest)

A generalist algorithm for cell and nucleus segmentation. 

This code was written by Carsen Stringer and Marius Pachitariu. To learn about Cellpose, read the [paper](https://t.co/4HFsxDezAP?amp=1) or watch the [talk](https://t.co/JChCsTD0SK?amp=1). For support, please open an [issue](https://github.com/MouseLand/cellpose/issues). 

If you want to improve Cellpose for yourself and for everyone else, please consider contributing manual segmentations for a few of your images via the built-in GUI interface (see instructions below). 

### Run cellpose without local installation

You can quickly try out Cellpose on the [website](http://www.cellpose.org) first (some features disabled). 

You can also run Cellpose in google colab with a GPU -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MouseLand/cellpose/blob/master/notebooks/run_cellpose_GPU.ipynb). This is recommended if you have issues with MKL or run speed on your local computer (and are running 3D volumes). Colab does not allow you to run the GUI, but you can save `*_seg.npy` files in colab that you can download and open in the GUI.

### Detailed documentation at [www.cellpose.org/docs](http://www.cellpose.org/docs).

## System requirements

Linux, Windows and Mac OS are supported for running the code. For running the graphical interface you will need a Mac OS later than Yosemite. At least 8GB of RAM is required to run the software. 16GB-32GB may be required for larger images and 3D volumes. The software has been heavily tested on Windows 10 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

## Installation

This process should take less than 5 minutes.

### (Option 1) Standard install in base environment

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.7** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path. 
2. From your base environment in an anaconda prompt/command prompt, run
~~~~
pip install cellpose[gui]
~~~~

If you want to install without the GUI dependencies, run `pip install cellpose`.

### (Option 2) Install in a new environment

Alternatively you can use the included environment file (if you'd like a cellpose-specific environment). This environment file includes all the dependencies for using the GUI. Using the environment file is **recommended** if you have problems with *option 1*. Please follow these instructions:

1. Download the [`environment.yml`](https://github.com/MouseLand/cellpose/blob/master/environment.yml?raw=true) file from the repository. You can do this by cloning the repository, or copy-pasting the text from the file into a text document on your local computer.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Change directories to where the `environment.yml` is and run `conda env create -f environment.yml`
4. To activate this new environment, run `conda activate cellpose`
5. You should see `(cellpose)` on the left side of the terminal line. Now run `python -m cellpose` and you're all set.

To upgrade cellpose (package [here](https://pypi.org/project/cellpose/)), run the following in the environment:
~~~~
pip install cellpose --upgrade
~~~~

If you have an older `cellpose` environment you can remove it with `conda env remove -n cellpose` before creating a new one.

Note you will always have to run **conda activate cellpose** before you run cellpose. If you want to run jupyter notebooks in this environment, then also `conda install jupyter` 
and `pip install matplotlib`.

### Common issues

If you receive the error: `Illegal instruction (core dumped)`, then likely mxnet does not recognize your MKL version. Please uninstall and reinstall mxnet without mkl:
~~~~
pip uninstall mxnet-mkl
pip uninstall mxnet
pip install mxnet==1.4.0
~~~~

**MAC OS ISSUE**: You may have an issue on Mac with the latest *opencv-python* library (package name *cv2*). Downgrade it with the command
~~~~
pip install opencv-python==3.4.5.20
~~~~

If you receive the error: `No module named PyQt5.sip`, then try uninstalling and reinstalling pyqt5
~~~~
pip uninstall pyqt5 pyqt5-tools
pip install pyqt5 pyqt5-tools pyqt5.sip
~~~~

If you receive an error associated with **matplotlib**, try upgrading it:
~~~~
pip install matplotlib --upgrade
~~~~


If you receive the error: `ImportError: _arpack DLL load failed`, then try uninstalling and reinstalling scipy
~~~~
pip uninstall scipy
pip install scipy
~~~~

If you are having issues with the graphical interface, make sure you have **python 3.7** and not python 3.8 installed.

**CUDA version**

If you plan on running many images, you may want to install a GPU version of *mxnet*. I recommend using CUDA 10.0 or greater. Follow the instructions [here](https://mxnet.apache.org/get_started?).

Before installing the GPU version, remove the CPU version:
~~~
pip uninstall mxnet-mkl
pip uninstall mxnet
~~~

When upgrading cellpose, you will want to ignore dependencies (so that mxnet-mkl does not install):
~~~
pip install --no-deps cellpose --upgrade
~~~

### Installation of github version

Follow steps from above to install the dependencies. In the github repository, run `pip install -e .` and the github version will be installed. If you want to go back to the pip version of cellpose, then say `pip install cellpose`.

## Running cellpose

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path:
~~~~
python -m cellpose
~~~~

The first time cellpose runs it downloads the latest available trained model weights from the website.

You can now **drag and drop** any images (*.tif, *.png, *.jpg, *.gif) into the GUI and run Cellpose, and/or manually segment them. When the GUI is processing, you will see the progress bar fill up and during this time you cannot click on anything in the GUI. For more information about what the GUI is doing you can look at the terminal/prompt you opened the GUI with. For example data, see [website](http://www.cellpose.org) or this google drive [folder](https://drive.google.com/open?id=18syVlaix8cIlrnNF20pEWKMWUsKx9R9z). For best accuracy and runtime performance, resize images so cells are less than 100 pixels across. 

### Step-by-step demo

1. Download the google drive [folder](https://drive.google.com/open?id=18syVlaix8cIlrnNF20pEWKMWUsKx9R9z) and unzip it. These are a subset of the test images from the paper.
2. Start the GUI with `python -m cellpose`.
3. Drag an image from the folder into the GUI.
4. Set the model (in demo all are `cyto`) and the channel you want to segment (in demo all are `green`). Optionally set the second channel if you are segmenting `cyto` and have an available nucleus channel.
5. Click the `calibrate` button to estimate the size of the objects in the image. Alternatively you can set the `cell diameter` by hand and press ENTER. You will see the size you set as a red disk at the bottom left of the image.
6. Click the `run segmentation` button. If MASKS ON is checked, you should see masks drawn on the image.
7. Now you can click the LEFT/RIGHT arrow keys to move through the folder and segment another image.

On the demo images each of these steps should run in less than a few seconds on a standard laptop or desktop (with mkl working).

### 3D segmentation

For multi-channel, multi-Z tiff's, the expected format is Z x channels x Ly x Lx.

## Contributing training data

We are very excited about receiving community contributions to the training data and re-training the cytoplasm model to make it better. Please follow these guidelines:

1. Run cellpose on your data to see how well it does. Try varying the diameter, which can change results a little. 
2. If there are relatively few mistakes, it won't help much to contribute labelled data. 
3. If there are consistent mistakes, your data is likely very different from anything in the training set, and you should expect major improvements from contributing even just a few manually segmented images.
4. For images that you contribute, the cells should be at least 10 pixels in diameter, and there should be **at least** several dozens of cells per image, ideally ~100. If your images are too small, consider combining multiple images into a single big one and then manually segmenting that. If they are too big, consider splitting them into smaller crops. 
5. For the manual segmentation, please try to outline the boundaries of the cell, so that everything (membrane, cytoplasm, nucleus) is inside the boundaries. Do not just outline the cytoplasm and exclude the membrane, because that would be inconsistent with our own labelling and we wouldn't be able to use that. 
6. Do not use the results of the algorithm in any way to do contributed manual segmentations. This can reinforce a vicious circle of mistakes, and compromise the dataset for further algorithm development. 

If you are having problems with the nucleus model, please open an issue before contributing data. Nucleus images are generally much less diverse, and we think the current training dataset already covers a very large set of modalities. 


## Using the GUI

The GUI serves two main functions:

1. Running the segmentation algorithm.
2. Manually labelling data.

There is a help window in the GUI that provides more instructions and 
a page in the documentation [here](http://cellpose.readthedocs.io/en/latest/gui.html).
 Also, if you hover over certain words in the GUI, their definitions 
 are revealed as tooltips.

### In a notebook

See [run_cellpose.ipynb](notebooks/run_cellpose.ipynb).

### From the command line

Run `python -m cellpose` and specify parameters as below. For instance to run on a folder with images where cytoplasm is green and nucleus is blue and save the output as a png:
~~~
python -m cellpose --dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 3 --save_png
~~~

You can specify the diameter for all the images or set to 0 if you want the algorithm to estimate it on an image by image basis. Here is how to run on nuclear data (grayscale) where the diameter is automatically estimated:
~~~
python -m cellpose --dir ~/images_nuclei/test/ --pretrained_model nuclei --diameter 0. --save_png
~~~

See the [docs](http://cellpose.readthedocs.io/en/latest/command.html) for more info.

## Outputs

See the [docs](http://cellpose.readthedocs.io/en/latest/outputs.html) for info.

## Dependencies
cellpose relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [mxnet_mkl](https://mxnet.apache.org/)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [scipy](https://www.scipy.org/)
- [natsort](https://natsort.readthedocs.io/en/master/)
