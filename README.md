# <p>  <b>Cellpose </b> </p>
<img src="http://www.cellpose.org/static/images/logo.png?raw=True" width="250" title="cellpose" alt="cellpose" align="right" vspace = "50">

[![Documentation Status](https://readthedocs.org/projects/cellpose/badge/?version=latest)](https://cellpose.readthedocs.io/en/latest/?badge=latest)
![tests](https://github.com/mouseland/cellpose/actions/workflows/test_and_deploy.yml/badge.svg)
[![codecov](https://codecov.io/gh/MouseLand/cellpose/branch/main/graph/badge.svg?token=9FFo4zNtYP)](https://codecov.io/gh/MouseLand/cellpose)
[![PyPI version](https://badge.fury.io/py/cellpose.svg)](https://badge.fury.io/py/cellpose)
[![Downloads](https://pepy.tech/badge/cellpose)](https://pepy.tech/project/cellpose)
[![Downloads](https://pepy.tech/badge/cellpose/month)](https://pepy.tech/project/cellpose)
[![Python version](https://img.shields.io/pypi/pyversions/cellpose)](https://pypistats.org/packages/cellpose)
[![Licence: GPL v3](https://img.shields.io/github/license/MouseLand/cellpose)](https://github.com/MouseLand/cellpose/blob/master/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/MouseLand/cellpose)](https://github.com/MouseLand/cellpose/graphs/contributors)
[![website](https://img.shields.io/website?url=https%3A%2F%2Fwww.cellpose.org)](https://www.cellpose.org)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fcellpose.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/cellpose)
[![repo size](https://img.shields.io/github/repo-size/MouseLand/cellpose)](https://github.com/MouseLand/cellpose/)
[![GitHub stars](https://img.shields.io/github/stars/MouseLand/cellpose?style=social)](https://github.com/MouseLand/cellpose/)
[![GitHub forks](https://img.shields.io/github/forks/MouseLand/cellpose?style=social)](https://github.com/MouseLand/cellpose/)

A generalist algorithm for cell and nucleus segmentation (v1.0) that can be optimized for your own data (v2.0) and (**NEW**) perform image restoration (v3.0).

Cellpose was written by Carsen Stringer and Marius Pachitariu. To learn about Cellpose3 (image restoration), read the [paper](https://www.biorxiv.org/content/10.1101/2024.02.10.579780v1). To learn about Cellpose 2.0 (human-in-the-loop), read the [paper](https://www.nature.com/articles/s41592-022-01663-4) or watch the [talk](https://www.youtube.com/watch?v=3ydtAhfq6H0). To learn about Cellpose 1.0, read the [paper](https://t.co/kBMXmPp3Yn?amp=1) or watch the [talk](https://t.co/JChCsTD0SK?amp=1). For support, please open an [issue](https://github.com/MouseLand/cellpose/issues). 

Please see install instructions [below](README.md/#Installation), and also check out the detailed documentation at [**cellpose.readthedocs.io**](https://cellpose.readthedocs.io/en/latest/) for more information.

### :star2: v3 (Feb 2024) :star2:

Cellpose3 enables image restoration in the GUI, API and CLI (saved to `_seg.npy`). To learn more...
* Check out the [paper](https://www.biorxiv.org/content/10.1101/2024.02.10.579780v1).
* API documentation [here](https://cellpose.readthedocs.io/en/latest/restore.html)
* Example google colab notebook for image restoration: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose3.ipynb), using new `CellposeDenoiseModel`.
* Example google colab notebook with new super-generalist "cyto3" model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cyto3.ipynb). Try out the new `cyto3` super-generalist Cellpose model with `model_type="cyto3"`.

denoising             |  deblurring                  |  upsampling
:-------------------------:|:-------------------------:|:-------------------------:
<img src="http://www.cellpose.org/static/images/poisson_cp3.gif?raw=True" width="350" title="cellpose denoising" alt="cellpose denoising examples"> | <img src="http://www.cellpose.org/static/images/blur_cp3.gif?raw=True" width="350" title="cellpose deblurring" alt="cellpose deblurring examples"> | <img src="http://www.cellpose.org/static/images/downsample_cp3.gif?raw=True" width="350" title="cellpose upsampling" alt="cellpose upsampling examples">

### CITATION

**If you use Cellpose 1, 2 or 3, please cite the Cellpose 1.0 [paper](https://t.co/kBMXmPp3Yn?amp=1):**  
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. <em>Nature methods, 18</em>(1), 100-106.

**If you use the human-in-the-loop training, please also cite the Cellpose 2.0 [paper](https://www.nature.com/articles/s41592-022-01663-4):**  
Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. <em>Nature methods</em>, 1-8.

**If you use the new image restoration models or cyto3, please also cite the Cellpose3 [paper](https://www.biorxiv.org/content/10.1101/2024.02.10.579780v1):**  
Stringer, C. & Pachitariu, M. (2024). Cellpose3: one-click image restoration for improved segmentation. <em>bioRxiv</em>.

:triangular_flag_on_post: All models in Cellpose, except `yeast_BF_cp3`, `yeast_PhC_cp3`, and `deepbacs_cp3`, are trained on some amount of data that is **CC-BY-NC**. The Cellpose annotated dataset is also CC-BY-NC.

### :star2: v2.0 (April 2022) :star2:

Cellpose 2.0 allows human-in-the-loop training of models! To learn more...
* Check out the twitter [thread](https://twitter.com/marius10p/status/1511415409047650307?s=20&t=umTVIG1CFKIWHYMrQqFKyQ) for an overview.
* Check out the [paper](https://www.nature.com/articles/s41592-022-01663-4) for more details on the algorithm and the performance. Also, there's a short review of the paper available [here](https://www.nature.com/articles/s41592-022-01664-3).
* Watch the short overview [talk](https://youtu.be/wB7XYh4QRiI) and watch the longer [tutorial talk](https://youtu.be/5qANHWoubZU) which goes through running Cellpose 2.0 in the GUI and a jupyter notebook.
* Check out the full human-in-the-loop [video](https://youtu.be/3Y1VKcxjNy4). 
* Check out the colab notebook to get cloud access to a GPU to train your models or run your custom models: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb).
* See how to use it yourself in the [docs](https://cellpose.readthedocs.io/en/latest/gui.html#training-your-own-cellpose-model) and also check out the help info in the `Models` menu in the GUI.

Mxnet is no longer supported in cellpose. To use mxnet, please use v1.0.2 (not recommended).

# Installation

You can install cellpose using conda or with native python if you have python3.8+ on your machine. 

## Local installation (< 2 minutes)

### System requirements

Linux, Windows and Mac OS are supported for running the code. For running the graphical interface you will need a Mac OS later than Yosemite. At least 8GB of RAM is required to run the software. 16GB-32GB may be required for larger images and 3D volumes. The software has been heavily tested on Windows 10 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

### Dependencies
cellpose relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [pytorch](https://pytorch.org/)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [scipy](https://www.scipy.org/)
- [natsort](https://natsort.readthedocs.io/en/master/)

### Option 1: Installation Instructions with conda 

If you have an older `cellpose` environment you can remove it with `conda env remove -n cellpose` before creating a new one.

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

1. Install an [Anaconda](https://www.anaconda.com/products/distribution) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt which has `conda` for **python 3** in the path
3. Create a new environment with `conda create --name cellpose python=3.8`. We recommend python 3.8, but python 3.9 and 3.10 will likely work as well.
4. To activate this new environment, run `conda activate cellpose`
5. (option 1) To install cellpose with the GUI, run `python -m pip install cellpose[gui]`.  If you're on a zsh server, you may need to use ' ': `python -m pip install 'cellpose[gui]'`.
6. (option 2) To install cellpose without the GUI, run `python -m pip install cellpose`. 

To upgrade cellpose (package [here](https://pypi.org/project/cellpose/)), run the following in the environment:

~~~sh
python -m pip install cellpose --upgrade
~~~

Note you will always have to run `conda activate cellpose` before you run cellpose. If you want to run jupyter notebooks in this environment, then also `python -m pip install notebook` and `python -m pip install matplotlib`.

You can also try to install cellpose and the GUI dependencies from your base environment using the command

~~~~sh
python -m pip install cellpose[gui]
~~~~

If you have **issues** with installation, see the [docs](https://cellpose.readthedocs.io/en/latest/installation.html) for more details. You can also use the cellpose environment file included in the repository and create a cellpose environment with `conda env create -f environment.yml` which may solve certain dependency issues.

If these suggestions fail, open an issue.

### Option 2: Installation Instructions with python's venv

Venv ([tutorial](https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv), for those interested) is a built-in tool in python for creating virtual environments. It is a good alternative if you don't want to install conda and already have python3 on your machine. The main difference is that you will need to choose where to install the environment and the packages. Cellpose will then live in this environment and not be accessible from other environments. You will need to navigate to the environment directory and activate it each time before running cellpose. The steps are similar to the conda installation:

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

1. Install python3.8 or later from [python.org](https://www.python.org/downloads/). This will be the version of python that will be used in the environment. You can check your python version with `python --version`.
2. Navigate to the directory where you want to create the environment and run `python3 -m venv cellpose` to create a new environment called `cellpose`.
3. Activate the environment with `source cellpose/bin/activate` on Mac/Linux or `cellpose\Scripts\activate` on Windows. A prefix `(cellpose)` should appear in the terminal.
4. Install cellpose into the `cellpose` venv using pip with `python -m pip install cellpose`.
5. Install the cellpose GUI, with `python -m pip install cellpose[gui]`. Depending on your terminal software, you may need to use quotes like this: `python -m pip install 'cellpose[gui]'`.
6. You can now run cellpose from this environment with `python -m cellpose` or `cellpose` if you are in the cellpose directory.
7. To deactivate the environment, run `deactivate`. 

### GPU version (CUDA) on Windows or Linux

If you plan on running many images, you may want to install a GPU version of *torch* (if it isn't already installed).

To use your NVIDIA GPU with python, you will first need to install the NVIDIA driver for your GPU, check out this [website](https://www.nvidia.com/Download/index.aspx?lang=en-us) to download it. You can also install the CUDA toolkit, or use the pytorch cudatoolkit (installed below with conda). If you have trouble with the below install, we recommend installing the CUDA toolkit yourself, choosing one of the 11.x releases [here](https://developer.nvidia.com/cuda-toolkit-archive).

Next we need to remove the CPU version of torch:
~~~
pip uninstall torch
~~~

To install the GPU version of torch, follow the instructions [here](https://pytorch.org/get-started/locally/). The conda install is strongly recommended, and then choose the CUDA version that is supported by your GPU (newer GPUs may need newer CUDA versions > 10.2). For instance this command will install the 11.6 version on Linux and Windows (note the `torchvision` and `torchaudio` commands are removed because cellpose doesn't require them):
~~~
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
~~~

If the latest CUDA versions don't work, try an older version like cuda 11.3:
~~~
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch
~~~~
Info on how to install several older versions is available [here](https://pytorch.org/get-started/previous-versions/). After install you can check `conda list` for `pytorch`, and its version info should have `cuXX.X`, not `cpu`.

### Installation of github version

Follow steps from above to install the dependencies. Then run 
~~~
pip install git+https://www.github.com/mouseland/cellpose.git
~~~

If you want edit ability to the code, in the github repository folder, run `pip install -e .`. If you want to go back to the pip version of cellpose, then say `pip install cellpose`.

## Run cellpose 1.0 without local python installation

You can quickly try out Cellpose on the [website](https://www.cellpose.org) first (many features disabled). 

You can also run Cellpose in google colab with a GPU: 
* a code-based notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_GPU.ipynb)
* a more user-friendly notebook for 2D segmentation written by [@pr4deepr](https://github.com/pr4deepr): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/Cellpose_cell_segmentation_2D_prediction_only.ipynb)
* a user-friendly [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) notebook that includes training cellpose models, written by [@guijacquemet](https://github.com/guijacquemet): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/Beta%20notebooks/Cellpose_2D_ZeroCostDL4Mic.ipynb)

The colab notebooks are recommended if you have issues with MKL or run speed on your local computer (and are running 3D volumes). Colab does not allow you to run the GUI, but you can save `*_seg.npy` files in colab that you can download and open in the GUI.

**Executable file**: You can download an executable file for [*Windows 10*](http://www.cellpose.org/windows) or for [*Mac OS*](http://www.cellpose.org/mac) (High Sierra or greater) that were made using PyInstaller on Intel processors (MKL acceleration works, but no GPU support). Note in both cases it will take a few seconds to open.

* The [*Mac OS*](https://www.cellpose.org/mac) file will download as `cellpose_mac` OR `cellpose_mac.dms`. You will need to make it into an executable file and run it through the terminal:
1. Open a terminal and run `cd ~/Downloads/`.
2. Run `chmod 777 cellpose_mac` OR `chmod 777 cellpose_mac.dms` to make the file executable.
3. Run `./cellpose_mac` OR `./cellpose_mac.dms` to open the cellpose GUI. Messages from cellpose will be printed in the terminal.
4. You can also run using the command line interface, e.g. as `./cellpose_mac --dir ~/Pictures/ --chan 2 --save_png`.

* The [*Windows 10*](https://www.cellpose.org/windows) file is an exe and you can click on it to run the GUI. You can also run using the command line interface, e.g. as `cellpose.exe --dir Pictures/ --chan 2 --save_png`

# Run cellpose locally

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path:
~~~~
python -m cellpose
~~~~

The first time cellpose runs it downloads the latest available trained model weights from the website.

You can now **drag and drop** any images (*.tif, *.png, *.jpg, *.gif) into the GUI and run Cellpose, and/or manually segment them. When the GUI is processing, you will see the progress bar fill up and during this time you cannot click on anything in the GUI. For more information about what the GUI is doing you can look at the terminal/prompt you opened the GUI with. For example data, see [website](http://www.cellpose.org) or this [zip file](https://www.cellpose.org/static/images/demo_images.zip). For best accuracy and runtime performance, resize images so cells are less than 100 pixels across. 

For 3D data, with multi-Z, please use the 3D version of the GUI with:
~~~~
python -m cellpose --Zstack
~~~~


## Step-by-step demo

1. Download this [folder](http://cellpose.org/static/images/demo_images.zip) of images and unzip it. These are a subset of the test images from the paper.
2. Start the GUI with `python -m cellpose`.
3. Drag an image from the folder into the GUI.
4. Set the model (in demo all are `cyto`) and the channel you want to segment (in demo all are `green`). Optionally set the second channel if you are segmenting `cyto` and have an available nucleus channel.
5. Click the `calibrate` button to estimate the size of the objects in the image. Alternatively (RECOMMENDED) you can set the `cell diameter` by hand and press ENTER. You will see the size you set as a red disk at the bottom left of the image.
6. Click the `run segmentation` button. If MASKS ON is checked, you should see masks drawn on the image.
7. Now you can click the LEFT/RIGHT arrow keys to move through the folder and segment another image.

To draw ROIs on the image you can right-click then hover to complete the ROI (do not right-click and drag). To remove ROIs left-click while holding down CTRL. See more details [here](https://cellpose.readthedocs.io/en/latest/gui.html).

On the demo images each of these steps should run in less than a few seconds on a standard laptop or desktop (with mkl working).

### 3D segmentation

For multi-channel, multi-Z tiff's, the expected format is Z x channels x Ly x Lx.

### Download of pretrained models

The models will be downloaded automatically from the [website](https://www.cellpose.org) when you first run a pretrained model in cellpose. If you are having issues with the downloads, you can download them from this [google drive zip file](https://drive.google.com/file/d/1zHGFYCqRCTwTPwgEUMNZu0EhQy2zaovg/view?usp=sharing), unzip the file and put the models in your home directory under the path .cellpose/models/, e.g. on Windows this would be C:/Users/YOUR_USERNAME/.cellpose/models/ or on Linux this would be /home/YOUR_USERNAME/.cellpose/models/, so /home/YOUR_USERNAME/.cellpose/models/cyto_0 is the full path to one model for example. If you cannot access google drive, the models are also available on baidu: Link：https://pan.baidu.com/s/1CARpRGCBHIYaz7KeyoX-fg ; Fetch code：pose ; thanks to @qixinbo!

# Older software releases

### UPDATE v1.0 (Jan 2022)

Cellpose has been relatively stable for a while now. Small bugs will continue to be fixed, but we are now releasing a reference 1.0 version. Larger updates to Cellpose will go towards a new 2.0 candidate version to be released soon.  

This update fixes bugs in GUI and plotting. It also stops model weight reloading to improve speed. `resample=True` is default again as in earlier releases, turn off with `--no_resample`. Now logging is turned off by default. Turn on in CLI with `--verbose` flag or in a script/notebook by
```
from cellpose.io import logger_setup
logger_setup();
```

To install this version please use
```
pip install cellpose==1.0.2
```

### UPDATE v0.7 (Nov 2021)

Check out [Omnipose](https://github.com/kevinjohncutler/omnipose), an extension of Cellpose for long filamentous bacteria. Omnipose was written by Kevin Cutler ([@kevinjohncutler](https://github.com/kevinjohncutler)). To learn about Omnipose, read the [paper](http://biorxiv.org/content/early/2021/11/04/2021.11.03.467199).

### UPDATE v0.6 (Dec 2020)

Pytorch is now the default deep neural network software for cellpose. Mxnet will still be supported. To install mxnet (CPU), run `pip install mxnet-mkl`. To use mxnet in a notebook, declare `torch=False` when creating a model, e.g. `model = models.Cellpose(torch=False)`. To use mxnet on the command line, add the flag `--mxnet`, e.g. `python -m cellpose --dir ~/images/ --mxnet`. The pytorch implementation is 20% faster than the mxnet implementation when running on the GPU and 20% slower when running on the CPU. 

Dynamics are computed using bilinear interpolation by default instead of nearest neighbor interpolation. Set `interp=False` in `model.eval` to turn off. The bilinear interpolation will be slightly slower on the CPU, but it is faster than nearest neighbor if using torch and the GPU is enabled.


### Timing (v0.6)

You can check if cellpose is running the MKL version (if you are using the CPU not the GPU) by adding the flag `--check_mkl`. If you are not using MKL cellpose will be much slower. Here are Cellpose run times divided into the time it takes to run the deep neural network (DNN) and the time for postprocessing (gradient tracking, segmentation, quality control etc.). The DNN runtime is shown using either a GPU (Nvidia GTX 1080Ti) or a CPU (Intel 10-core 7900X), with or without network ensembling (4net vs 1net). The postprocessing runtime is similar regardless of ensembling or CPU/GPU version. Runtime is shown for different image sizes, all with a cell diameter of 30 pixels (the average from our training set).

|   | 256 pix | 512 pix | 1024 pix |
|----|-------|------|----------|
| DNN (1net, GPU) | 0.054 s | 0.12 s | 0.31 s  |
| DNN (1net, CPU) | 0.30 s | 0.65 s | 2.4 s  |
| DNN (4net, GPU) | 0.23 s | 0.41 s | 1.3 s |
| DNN (4net, CPU) | 1.3 s | 2.5 s | 9.1 s  |
|  | |  |  |
| Postprocessing (CPU) | 0.32 s | 1.2 s | 6.1 s  |
