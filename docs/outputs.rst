Outputs
-------------------------

*.npy output 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``*_seg.npy`` files have the following fields:

- *filename* : filename of image
- *img* : image with chosen channels (nchan x Ly x Lx) (if not multiplane)
- *masks* : masks (0 = NO masks; 1,2,... = mask labels)
- *colors* : colors for masks
- *outlines* : outlines of masks (0 = NO outline; 1,2,... = outline labels)
- *chan_choose* : channels that you chose in GUI (0=gray/none, 1=red, 2=green, 3=blue)
- *ismanual* : element *k* = whether or not mask *k* was manually drawn or computed by the cellpose algorithm
- *flows* : flows[0] is XY flow in RGB, flows[1] is the cell probability in range 0-255 instead of 0.0 to 1.0, flows[2] is Z flow in range 0-255 (if it exists)
- *est_diam* : estimated diameter (if run on command line)
- *zdraw* : for each mask, which planes were manually labelled (planes in between manually drawn have interpolated masks)

Here is an example of loading in a ``*_seg.npy`` file and plotting masks and outlines

::
    import numpy as np
    from cellpose import plot
    dat = np.load('_seg.npy', allow_pickle=True).item()

    # plot image with masks overlaid
    mask_RGB = plot.mask_overlay(dat['img'], dat['masks'],
                            colors=np.array(dat['colors']))

    # plot image with outlines embedded in image in red (can change color of outline)
    outline_RGB = plot.outline_overlay(dat['img'], dat['outlines'],
                            channels=dat['chan_choose'], color=[255,0,0])

    # plot image with outlines overlaid in red
    outlines = plot.plot_outlines(dat['masks'])
    plt.imshow(dat['img'])
    for o in outlines:
        plt.plot(o[:,0], o[:,1], color='r')


If you run in a notebook and want to save to a `*_seg.npy` file, run 

::

    from cellpose import io
    io.masks_flows_to_seg(images, masks, flows, diams, channels, file_names)

where each of these inputs is a list (as the output of `model.eval` is)

PNG output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can save masks to PNG in the GUI.

To save masks (and other plots in PNG) using the command line, add the flag ``--save_png``.

Or use the function below if running in a notebook

::

    from cellpose import io
    io.save_to_png(images, masks, flows, image_names)


Plotting functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``plot.py`` there are functions, like ``show_segmentation``:

::

    from cellpose import plot, transforms

    for idx in range(nimg):
        img = transforms.reshape(imgs[idx], channels[idx])
        img = plot.rgb_image(img)
        maski = masks[idx]
        flowi = flows[idx][0]

        fig = plt.figure(figsize=(12,3))
        # can save images (set save_dir=None if not)
        plot.show_segmentation(fig, img, maski, flowi)
        plt.tight_layout()
        plt.show()

.. image:: _static/ex_seg.png
    :width: 600px
    :align: center
    :alt: example segmentation
