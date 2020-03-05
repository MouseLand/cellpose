import cv2
from scipy.ndimage.filters import maximum_filter1d
import skimage
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import os, warnings, time
import matplotlib.pyplot as plt

from cellpose import plot

def masks_flows_to_seg(images, masks, flows, diams, channels, file_names):
    """ save output of model eval to be loaded in GUI """
    nimg = len(masks)
    if channels is None:
        channels = [0,0]
    for n in range(nimg):
        flowi = []
        flowi.append(flows[n][0][np.newaxis,...])
        flowi.append((np.clip(normalize99(flows[n][2]),0,1) * 255).astype(np.uint8)[np.newaxis,...])
        flowi.append((flows[n][1][-1]/10 * 127 + 127).astype(np.uint8)[np.newaxis,...])
        outlines = masks[n] * plot.masks_to_outlines(masks[n])
        base = os.path.splitext(file_names[n])[0]
        if images[n].shape[0]<8:
            np.transpose(images[n], (1,2,0))
        np.save(base+ '_seg.npy',
                    {'outlines': outlines.astype(np.uint16),
                     'masks': masks[n].astype(np.uint16),
                     'chan_choose': channels,
                     'img': images[n],
                     'ismanual': np.zeros(masks[n].max(), np.bool),
                     'filename': file_names[n],
                     'flows': flowi,
                     'est_diam': diams[n]})

def save_to_png(images, masks, flows, file_names):
    nimg = len(images)
    for n in range(nimg):
        img = images[n].copy()
        if img.ndim<3:
            img = img[:,:,np.newaxis]
        elif img.shape[0]<8:
            np.transpose(img, (1,2,0))
        base = os.path.splitext(file_names[n])[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(base+'_cp_masks.png', masks[n].astype(np.uint16))
        maski = masks[n]
        flowi = flows[n][0]
        fig = plt.figure(figsize=(12,3))
        # can save images (set save_dir=None if not)
        plot.show_segmentation(fig, img, maski, flowi)
        fig.savefig(base+'_cp.png', dpi=300)
        plt.close(fig)


def use_gpu(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
        return True
    except mx.MXNetError:
        return False

def taper_mask(bsize=224, sig=7.5):
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask[:, np.newaxis]
    return mask

def diameters(masks):
    unique, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    return md, counts**0.5

def radius_distribution(masks, bins):
    unique, counts = np.unique(masks, return_counts=True)
    counts = counts[unique!=0]
    nb, _ = np.histogram((counts**0.5)*0.5, bins)
    nb = nb.astype(np.float32)
    if nb.sum() > 0:
        nb = nb / nb.sum()
    md = np.median(counts**0.5)*0.5
    if np.isnan(md):
        md = 0
    return nb, md, (counts**0.5)/2

def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def gabors(npix):
    ''' npix - size of gabor patch (should be ODD)'''
    y,x=np.meshgrid(np.arange(npix),np.arange(npix))
    sigma = 1
    f = 0.1
    theta = np.linspace(0, 2*np.pi, 33)[:-1]
    theta = theta[:,np.newaxis,np.newaxis]
    ycent,xcent = y.mean(), x.mean()
    yc = y - ycent
    xc = x - xcent
    ph = np.pi/2

    xc = xc[np.newaxis,:,:]
    yc = yc[np.newaxis,:,:]
    G = np.exp(-(xc**2 + yc**2) / (2*sigma**2)) * np.cos(ph + f * (yc*np.cos(theta) + xc*np.sin(theta)))

    return G

def process_cells(M0, npix=20):
    unq, ic = np.unique(M0, return_counts=True)
    for j in range(len(unq)):
        if ic[j]<npix:
            M0[M0==unq[j]] = 0
    return M0
