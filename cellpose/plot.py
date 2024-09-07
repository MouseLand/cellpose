"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from . import utils, io, transforms

try:
    import matplotlib
    MATPLOTLIB_ENABLED = True
except:
    MATPLOTLIB_ENABLED = False

try:
    from skimage import color
    from skimage.segmentation import find_boundaries
    SKIMAGE_ENABLED = True
except:
    SKIMAGE_ENABLED = False


# modified to use sinebow color
def dx_to_circ(dP, transparency=False, mask=None):
    """Converts the optic flow representation to a circular color representation.

    Args:
        dP (ndarray): Flow field components [dy, dx].
        transparency (bool, optional): Controls the opacity based on the magnitude of flow. Defaults to False.
        mask (ndarray, optional): Multiplies each RGB component to suppress noise.

    Returns:
        ndarray: The circular color representation of the optic flow.

    """
    dP = np.array(dP)
    mag = np.clip(transforms.normalize99(np.sqrt(np.sum(dP**2, axis=0))), 0, 1.)
    angles = np.arctan2(dP[1], dP[0]) + np.pi
    a = 2
    r = ((np.cos(angles) + 1) / a)
    g = ((np.cos(angles + 2 * np.pi / 3) + 1) / a)
    b = ((np.cos(angles + 4 * np.pi / 3) + 1) / a)

    if transparency:
        im = np.stack((r, g, b, mag), axis=-1)
    else:
        im = np.stack((r * mag, g * mag, b * mag), axis=-1)

    if mask is not None and transparency and dP.shape[0] < 3:
        im[:, :, -1] *= mask

    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im


def show_segmentation(fig, img, maski, flowi, channels=[0, 0], file_name=None):
    """Plot segmentation results (like on website).

    Can save each panel of figure with file_name option. Use channels option if
    img input is not an RGB image with 3 channels.

    Args:
        fig (matplotlib.pyplot.figure): Figure in which to make plot.
        img (ndarray): 2D or 3D array. Image input into cellpose.
        maski (int, ndarray): For image k, masks[k] output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flowi (int, ndarray): For image k, flows[k][0] output from Cellpose.eval (RGB of flows).
        channels (list of int, optional): Channels used to run Cellpose, no need to use if image is RGB. Defaults to [0, 0].
        file_name (str, optional): File name of image. If file_name is not None, figure panels are saved. Defaults to None.
        seg_norm (bool, optional): Improve cell visibility under labels. Defaults to False.
    """
    if not MATPLOTLIB_ENABLED:
        raise ImportError(
            "matplotlib not installed, install with 'pip install matplotlib'")
    ax = fig.add_subplot(1, 4, 1)
    img0 = img.copy()

    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if img0.max() <= 50.0:
            img0 = np.uint8(np.clip(img0, 0, 1) * 255)
    ax.imshow(img0)
    ax.set_title("original image")
    ax.axis("off")

    outlines = utils.masks_to_outlines(maski)

    overlay = mask_overlay(img0, maski)

    ax = fig.add_subplot(1, 4, 2)
    outX, outY = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[outX, outY] = np.array([255, 0, 0])  # pure red

    ax.imshow(imgout)
    ax.set_title("predicted outlines")
    ax.axis("off")

    ax = fig.add_subplot(1, 4, 3)
    ax.imshow(overlay)
    ax.set_title("predicted masks")
    ax.axis("off")

    ax = fig.add_subplot(1, 4, 4)
    ax.imshow(flowi)
    ax.set_title("predicted cell pose")
    ax.axis("off")

    if file_name is not None:
        save_path = os.path.splitext(file_name)[0]
        io.imsave(save_path + "_overlay.jpg", overlay)
        io.imsave(save_path + "_outlines.jpg", imgout)
        io.imsave(save_path + "_flows.jpg", flowi)


def mask_rgb(masks, colors=None):
    """Masks in random RGB colors.

    Args:
        masks (int, 2D array): Masks where 0=NO masks; 1,2,...=mask labels.
        colors (int, 2D array, optional): Size [nmasks x 3], each entry is a color in 0-255 range.

    Returns:
        RGB (uint8, 3D array): Array of masks overlaid on grayscale image.
    """
    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)

    HSV = np.zeros((masks.shape[0], masks.shape[1], 3), np.float32)
    HSV[:, :, 2] = 1.0
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            HSV[ipix[0], ipix[1], 0] = np.random.rand()
        else:
            HSV[ipix[0], ipix[1], 0] = colors[n, 0]
        HSV[ipix[0], ipix[1], 1] = np.random.rand() * 0.5 + 0.5
        HSV[ipix[0], ipix[1], 2] = np.random.rand() * 0.5 + 0.5
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB


def mask_overlay(img, masks, colors=None):
    """Overlay masks on image (set image to grayscale).

    Args:
        img (int or float, 2D or 3D array): Image of size [Ly x Lx (x nchan)].
        masks (int, 2D array): Masks where 0=NO masks; 1,2,...=mask labels.
        colors (int, 2D array, optional): Size [nmasks x 3], each entry is a color in 0-255 range.

    Returns:
        RGB (uint8, 3D array): Array of masks overlaid on grayscale image.
    """
    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim > 2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:, :, 2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max() + 1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            HSV[ipix[0], ipix[1], 0] = hues[n]
        else:
            HSV[ipix[0], ipix[1], 0] = colors[n, 0]
        HSV[ipix[0], ipix[1], 1] = 1.0
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB


def image_to_rgb(img0, channels=[0, 0]):
    """Converts image from 2 x Ly x Lx or Ly x Lx x 2 to RGB Ly x Lx x 3.

    Args:
        img0 (ndarray): Input image of shape 2 x Ly x Lx or Ly x Lx x 2.

    Returns:
        ndarray: RGB image of shape Ly x Lx x 3.

    """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim < 3:
        img = img[:, :, np.newaxis]
    if img.shape[0] < 5:
        img = np.transpose(img, (1, 2, 0))
    if channels[0] == 0:
        img = img.mean(axis=-1)[:, :, np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:, :, i]) > 0:
            img[:, :, i] = np.clip(transforms.normalize99(img[:, :, i]), 0, 1)
            img[:, :, i] = np.clip(img[:, :, i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1] == 1:
        RGB = np.tile(img, (1, 1, 3))
    else:
        RGB[:, :, channels[0] - 1] = img[:, :, 0]
        if channels[1] > 0:
            RGB[:, :, channels[1] - 1] = img[:, :, 1]
    return RGB


def interesting_patch(mask, bsize=130):
    """
    Get patch of size bsize x bsize with most masks.

    Args:
        mask (ndarray): Input mask.
        bsize (int): Size of the patch.

    Returns:
        tuple: Patch coordinates (y, x).

    """
    Ly, Lx = mask.shape
    m = np.float32(mask > 0)
    m = gaussian_filter(m, bsize / 2)
    y, x = np.unravel_index(np.argmax(m), m.shape)
    ycent = max(bsize // 2, min(y, Ly - bsize // 2))
    xcent = max(bsize // 2, min(x, Lx - bsize // 2))
    patch = [
        np.arange(ycent - bsize // 2, ycent + bsize // 2, 1, int),
        np.arange(xcent - bsize // 2, xcent + bsize // 2, 1, int)
    ]
    return patch


def disk(med, r, Ly, Lx):
    """Returns the pixels of a disk with a given radius and center.

    Args:
        med (tuple): The center coordinates of the disk.
        r (float): The radius of the disk.
        Ly (int): The height of the image.
        Lx (int): The width of the image.

    Returns:
        tuple: A tuple containing the y and x coordinates of the pixels within the disk.

    """
    yy, xx = np.meshgrid(np.arange(0, Ly, 1, int), np.arange(0, Lx, 1, int),
                         indexing="ij")
    inds = ((yy - med[0])**2 + (xx - med[1])**2)**0.5 <= r
    y = yy[inds].flatten()
    x = xx[inds].flatten()
    return y, x


def outline_view(img0, maski, color=[1, 0, 0], mode="inner"):
    """
    Generates a red outline overlay onto the image.

    Args:
        img0 (numpy.ndarray): The input image.
        maski (numpy.ndarray): The mask representing the region of interest.
        color (list, optional): The color of the outline overlay. Defaults to [1, 0, 0] (red).
        mode (str, optional): The mode for generating the outline. Defaults to "inner".

    Returns:
        numpy.ndarray: The image with the red outline overlay.

    """
    if img0.ndim == 2:
        img0 = np.stack([img0] * 3, axis=-1)
    elif img0.ndim != 3:
        raise ValueError("img0 not right size (must have ndim 2 or 3)")

    if SKIMAGE_ENABLED:
        outlines = find_boundaries(maski, mode=mode)
    else:
        outlines = utils.masks_to_outlines(maski, mode=mode)
    outY, outX = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[outY, outX] = np.array(color)

    return imgout
