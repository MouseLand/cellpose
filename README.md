# cellpose <img src="cellpose/logo/logo.png" width="250" title="cellpose" alt="cellpose" align="right" vspace = "50">

Anatomical segmentation algorithm for cytoplasm and nuclei
Copyright (C) 2019  Howard Hughes Medical Institute Janelia Research Campus  

cellpose includes the following modules:

* cell size estimation
* anatomical segmentation
* automated removal of bad masks
* visualization GUI

This code was written by Carsen Stringer and Marius Pachitariu.  
For support, please open an [issue](https://github.com/MouseLand/cellpose/issues).

## Installation

Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.x** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path. From your base environment (or you can make a new environment) in an anaconda prompt/command prompt, run
~~~~
pip install cellpose
~~~~

Alternatively you can use the included environment file (if you'd like a cellpose-specific environment). This is also recommended if you're having issues with conflicts with the pip version:

1. Download the [`environment.yml`](https://github.com/MouseLand/cellpose/blob/master/environment.yml) file from the repository. You can do this by cloning the repository, or copy-pasting the text from the file into a text document on your local computer.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Change directories to where the `environment.yml` is and run `conda env create -f environment.yml`
4. To activate this new environment, run `conda activate cellpose`
5. Install cellpose with `pip install -e .` (in the repo folder)
6. You should see `(cellpose)` on the left side of the terminal line. Now run `python -m cellpose` and you're all set.

If you have an older `cellpose` environment you can remove it with `conda env remove -n cellpose` before creating a new one.

Note you will always have to run **conda activate cellpose** before you run cellpose. Conda ensures all the versions of code are playing nice together. If you want to run jupyter notebooks in this environment, then also `conda install jupyter`.

To upgrade cellpose (package [here](https://pypi.org/project/cellpose/)), run the following in the environment:
~~~~
pip install cellpose --upgrade
~~~~

**Common issues**

If when running `python -m cellpose`, you receive the error: `No module named PyQt5.sip`, then try uninstalling and reinstalling pyqt5
~~~~
pip uninstall pyqt5 pyqt5-tools
pip install pyqt5 pyqt5-tools pyqt5.sip
~~~~

If when running `python -m cellpose`, you receive an error associated with **matplotlib**, try upgrading it:
~~~~
pip install matplotlib --upgrade
~~~~

If you are on Yosemite Mac OS, PyQt doesn't work, and you won't be able to use the graphical interface for cellpose. More recent versions of Mac OS are fine.

The software has been heavily tested on Windows 10 and Ubuntu 18.04, and less well tested on Mac OS. Please post an issue if you have installation problems.

## Installation of github version for developers and super-users :)

Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.x** and your operating system.

1. Clone the repository and `cd cellpose` in an anaconda prompt / command prompt. Your `(base)` environment will have most of the dependencies.
2. Now run `pip install -e .` to install the code in the github repository.
3. Run `python -m cellpose` (<-- you can run this command from anywhere now. If you want to go back to the supported version of cellpose, then say `pip install cellpose` and that will grab the pip package of cellpose)

## Example data

See [website](http://www.cellpose.appspot.com).

## Running cellpose

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path:
~~~~
python -m cellpose
~~~~

You can now **drag and drop** any images (*.tif, *.png, *.jpg, *.gif) into the GUI and manually segment them and/or process them in the GUI.

For multi-channel, multi-Z tiff's, the expected format is Z x channels x Ly x Lx.

When the GUI is processing, you will see the progress bar fill up and during this time you cannot click on anything in the GUI. For more information about what the GUI is doing you can look at the terminal/prompt you opened the GUI with.

### In the GUI

The GUI serves two main functions:

1. Running the segmentation algorithm.
2. Manually labelling data.

You can open the "Help" at any time with Ctrl+H for the following information inside the GUI.

Main GUI mouse controls (works in all views):

- Pan  = left-click  + drag  
- Zoom = scroll wheel
- Full view = double left-click
- Select mask = left-click on mask
- Delete mask = Ctrl + left-click
- Start draw mask = right-click
- End draw mask = right-click, or return to circle at beginning

Overlaps in masks are NOT allowed. If you draw a mask on top of another mask, it is cropped so that it doesn't overlap with the old mask. Masks in 2D should be single strokes (single stroke is checked). If you want to draw masks in 3D (experimental), then you can turn this option off and draw a stroke on each plane with the cell and then press ENTER.

!NOTE!: The GUI automatically saves after you draw a mask but NOT after segmentation. Save in the file menu or with Ctrl+S. The output file is in the same folder as the loaded image with `_manual.npy` appended.

| Keyboard shortcuts  | Description                                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| CTRL+Z              | undo previously drawn mask/stroke                              |
| CTRL+0              | clear all masks                                                |
| CTRL+L              | load image (can alternatively drag and drop image)             |
| CTRL+S              | SAVE MASKS IN IMAGE to `_manual.npy` file                      |
| CTRL+P              | load `_manual.npy` file (note: it will load automatically with image if it exists) |
| CTRL+M              | load masks file (must be same size as image with 0 for NO mask, and 1,2,3... for masks)|
| CTRL+N              | load numpy stack (NOT WORKING ATM)                             |
| A/D or LEFT/RIGHT   | cycle through images in current directory                      |
| W/S or UP/DOWN      | change color (RGB/gray/red/green/blue)                         |
| PAGE-UP / PAGE-DOWN | change to flows and cell prob views (if segmentation computed) |
| , / .               | increase / decrease brush size for drawing masks               |
| X                   | turn masks ON or OFF                                           |
| Z                   | toggle outlines ON or OFF                                      |
| C                   | cycle through labels for image type (saved to `_manual.npy`)   |

**Segmentation options**

SIZE: you can manually enter the approximate diameter for your cells, or press "calibrate" to let the model estimate it. The size is represented by a disk at the bottom of the view window (can turn this disk of by unchecking "scale disk on").

use GPU: if you have specially installed the cuda version of mxnet, then you can activate this, but it won't give huge speedups when running single images in the GUI.

MODEL: there is a *cytoplasm* model and a *nuclei* model, choose what you want to segment

CHAN TO SEG: this is the channel in which the cytoplasm or nuclei exist

CHAN2 (OPT): if *cytoplasm* model is chosen, then choose the nuclear channel for this option


### In a notebook

TODO

## Outputs

`*_manual.npy` files have the following fields:

- *filename* : filename of image
- *img* : image with chosen channels (Z x nchan x Ly x Lx)
- *masks* : masks (-1 = NO masks, 0,1,2,... = mask labels)
- *colors* : colors for masks
- *outlines* : outlines of masks (-1 = NO outline, 0,1,2,... = outline labels)
- *chan_choose* : channels that you chose in GUI (0=gray/none, 1=red, 2=green, 3=blue)
- *cell_type* : string with cell_type that you chose
- *cells* : int of cell_type that you chose

~~~~
import numpy as np
from cellpose import plot

dat = np.load('_manual.npy', allow_pickle=True).item()

# plot image with masks overlaid
RGB = plot.mask_overlay(dat['img'][0], dat['masks'][0]+1,
                        colors=np.array(dat['colors']))

# plot image with outlines overlaid in red (can change color of outline)
RGB = plot.outline_overlay(dat['img'][0], dat['outlines'][0]+1,
                           channels=dat['chan_choose'], color=[255,0,0])
~~~~~

## Dependencies
cellpose relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [mxnet_mkl](https://mxnet.apache.org/)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [scipy](https://www.scipy.org/)
- [scikit-image](https://scikit-image.org/)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [matplotlib](https://matplotlib.org/) (not for plotting (only using hsv_to_rgb and colormap function), should not conflict with PyQt5)
