GUI
------------------------------

Starting the GUI 
~~~~~~~~~~~~~~~~~~~~~~~

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path:
::

    python -m cellpose

The first time cellpose runs it downloads the latest available trained model weights from the website.

You can **drag and drop** images (.tif, .png, .jpg, .gif) into the GUI and run Cellpose, and/or manually segment them. When the GUI is processing, you will see the progress bar fill up and during this time you cannot click on anything in the GUI. For more information about what the GUI is doing you can look at the terminal/prompt you opened the GUI with. For example data, See [website](http://www.cellpose.org). For best accuracy and runtime performance, resize images so cells are less than 100 pixels across. 

For multi-channel, multi-Z tiff's, the expected format is Z x channels x Ly x Lx.

Using the GUI 
~~~~~~~~~~~~~~~~~~~~~~~

The GUI serves two main functions:

1. Running the segmentation algorithm.
2. Manually labelling data.
3. (NEW) Fine-tuning a pretrained cellpose model on your own data.

Main GUI mouse controls (works in all views):

-  Pan = left-click + drag
-  Zoom = scroll wheel (or +/= and - buttons)
-  Full view = double left-click
-  Select mask = left-click on mask
-  Delete mask = Ctrl (or Command on Mac) + left-click
-  Merge masks = Alt + left-click (will merge last two)
-  Start draw mask = right-click
-  End draw mask = right-click, or return to circle at beginning

Overlaps in masks are NOT allowed. If you draw a mask on top of another
mask, it is cropped so that it doesn't overlap with the old mask. Masks
in 2D should be single strokes (if *single_stroke* is checked).

If you want to draw masks in 3D, then you can turn *single_stroke*
option off and draw a stroke on each plane with the cell and then press
ENTER. 3D labelling will fill in unlabelled z-planes so that you do not
have to as densely label.

.. note::
    The GUI automatically saves after you draw a mask but NOT after
    segmentation and NOT after 3D mask drawing (too slow). Save in the file
    menu or with Ctrl+S. The output file is in the same folder as the loaded
    image with ``_seg.npy`` appended.

Segmentation options
~~~~~~~~~~~~~~~~~~~~~~~~

SIZE: you can manually enter the approximate diameter for your cells, or
press "calibrate" to let the model estimate it. The size is represented
by a disk at the bottom of the view window (can turn this disk off by
unchecking "scale disk on").

use GPU: if you have installed the cuda version of mxnet, then you can activate this, but it won't give huge speedups when running single images in the GUI.

MODEL: there is a *cytoplasm* model and a *nuclei* model, choose what you want to segment

CHAN TO SEG: this is the channel in which the cytoplasm or nuclei exist

CHAN2 (OPT): if *cytoplasm* model is chosen, then choose the nuclear channel for this option

Training your own cellpose model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out this `video <https://youtu.be/3Y1VKcxjNy4>`_ to watch Marius going through the process.

1. Drag and drop an image from a folder of images with a similar style (like similar cell types).
2. Run the built-in models on the data in the "model zoo" and find the one that works best for your data. Make sure that if you have a nuclear channel you have selected it for CHAN2.
3. Fix the labelling by drawing new ROIs (right-click) and deleting incorrect ones (CTRL+click). The GUI autosaves any manual changes (but does not autosave after running the model, for that click CTRL+S). The segmentation is saved in a ``_seg.npy`` file.
4. Go to the "Models" menu in the File bar at the top and click "Train new model..." or use shortcut CTRL+T.
5. Choose the pretrained model to start the training from (the model you used in #2), and type in the model name that you want to use. The other parameters should work well in general for most data types. Then click OK.
6. The model will train (much faster if you have a GPU) and then auto-run on the next image in the folder. Next you can repeat #3-#5 as many times as is necessary.
7. The trained model is available to use in the future in the GUI in the "custom model" section and is saved in your image folder.

Contributing training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are very excited about receiving community contributions to the training data and re-training the cytoplasm model to make it better. Please follow these guidelines:

1. Run cellpose on your data to see how well it does. Try varying the diameter, which can change results a little. 
2. If there are relatively few mistakes, it won't help much to contribute labelled data. 
3. If there are consistent mistakes, your data is likely very different from anything in the training set, and you should expect major improvements from contributing even just a few manually segmented images.
4. For images that you contribute, the cells should be at least 10 pixels in diameter, and there should be **at least** several dozens of cells per image, ideally ~100. If your images are too small, consider combining multiple images into a single big one and then manually segmenting that. If they are too big, consider splitting them into smaller crops. 
5. For the manual segmentation, please try to outline the boundaries of the cell, so that everything (membrane, cytoplasm, nucleus) is inside the boundaries. Do not just outline the cytoplasm and exclude the membrane, because that would be inconsistent with our own labelling and we wouldn't be able to use that. 
6. Do not use the results of the algorithm in any way to do contributed manual segmentations. This can reinforce a vicious circle of mistakes, and compromise the dataset for further algorithm development. 

If you are having problems with the nucleus model, please open an issue before contributing data. Nucleus images are generally much less diverse, and we think the current training dataset already covers a very large set of modalities. 
Additionally, you can run a non-nuclear model on nuclear data such as cyto.


Keyboard shortcuts 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------+-----------------------------------------------+
| Keyboard shortcuts  | Description                                   |
+=====================+===============================================+
| CTRL+H              | help                                          |
+---------------------+-----------------------------------------------+            
| =/+  // -           | zoom in // zoom out                           |
+---------------------+-----------------------------------------------+
| CTRL+Z              | undo previously drawn mask/stroke             |
+---------------------+-----------------------------------------------+
| CTRL+0              | clear all masks                               |
+---------------------+-----------------------------------------------+
| CTRL+L              | load image (can alternatively drag and drop   |
|                     | image)                                        |
+---------------------+-----------------------------------------------+
| CTRL+S              | SAVE MASKS IN IMAGE to ``_seg.npy`` file      |
+---------------------+-----------------------------------------------+
| CTRL+P              | load ``_seg.npy`` file (note: it will load    |
|                     | automatically with image if it exists)        |
+---------------------+-----------------------------------------------+
| CTRL+M              | load masks file (must be same size as image   |
|                     | with 0 for NO mask, and 1,2,3... for masks)   |
+---------------------+-----------------------------------------------+
| A/D or LEFT/RIGHT   | cycle through images in current directory     |
+---------------------+-----------------------------------------------+
| W/S or UP/DOWN      | change color (RGB/gray/red/green/blue)        |
+---------------------+-----------------------------------------------+
| PAGE-UP / PAGE-DOWN | change to flows and cell prob views (if       |
|                     | segmentation computed)                        |
+---------------------+-----------------------------------------------+
| , / .               | increase / decrease brush size for drawing    |
+---------------------+-----------------------------------------------+
| X                   | turn masks ON or OFF                          |
+---------------------+-----------------------------------------------+
| Z                   | toggle outlines ON or OFF                     |
+---------------------+-----------------------------------------------+
| C                   | cycle through labels for image type (saved to |
|                     | ``_seg.npy``)                                 |
+---------------------+-----------------------------------------------+



