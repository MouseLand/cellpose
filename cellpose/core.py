"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import logging
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile
import cv2
from scipy.stats import mode
import fastremap
from . import transforms, dynamics, utils, plot, metrics, resnet_torch

import torch
from torch import nn
from torch.utils import mkldnn as mkldnn_utils

TORCH_ENABLED = True

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)


def use_gpu(gpu_number=0, use_torch=True):
    """ 
    Check if GPU is available for use.

    Args:
        gpu_number (int): The index of the GPU to be used. Default is 0.
        use_torch (bool): Whether to use PyTorch for GPU check. Default is True.

    Returns:
        bool: True if GPU is available, False otherwise.

    Raises:
        ValueError: If use_torch is False, as cellpose only runs with PyTorch now.
    """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError("cellpose only runs with PyTorch now")


def _use_gpu_torch(gpu_number=0):
    """
    Checks if CUDA is available and working with PyTorch.

    Args:
        gpu_number (int): The GPU device number to use (default is 0).

    Returns:
        bool: True if CUDA is available and working, False otherwise.
    """
    try:
        device = torch.device("cuda:" + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        core_logger.info("** TORCH CUDA version installed and working. **")
        return True
    except:
        core_logger.info("TORCH CUDA version not installed/working.")
        return False


def assign_device(use_torch=True, gpu=False, device=0):
    """
    Assigns the device (CPU or GPU or mps) to be used for computation.

    Args:
        use_torch (bool, optional): Whether to use torch for GPU detection. Defaults to True.
        gpu (bool, optional): Whether to use GPU for computation. Defaults to False.
        device (int or str, optional): The device index or name to be used. Defaults to 0.

    Returns:
        torch.device: The assigned device.
        bool: True if GPU is used, False otherwise.
    """
    mac = False
    cpu = True
    if isinstance(device, str):
        if device == "mps":
            mac = True
        else:
            device = int(device)
    if gpu and use_gpu(use_torch=True):
        device = torch.device(f"cuda:{device}")
        gpu = True
        cpu = False
        core_logger.info(">>>> using GPU")
    elif mac:
        try:
            device = torch.device("mps")
            gpu = True
            cpu = False
            core_logger.info(">>>> using GPU")
        except:
            cpu = True
            gpu = False

    if cpu:
        device = torch.device("cpu")
        core_logger.info(">>>> using CPU")
        gpu = False
    return device, gpu


def check_mkl(use_torch=True):
    """
    Checks if MKL-DNN is enabled and working.

    Args:
        use_torch (bool, optional): Whether to use torch. Defaults to True.

    Returns:
        bool: True if MKL-DNN is enabled, False otherwise.
    """
    mkl_enabled = torch.backends.mkldnn.is_available()
    if mkl_enabled:
        mkl_enabled = True
    else:
        core_logger.info(
            "WARNING: MKL version on torch not working/installed - CPU version will be slightly slower."
        )
        core_logger.info(
            "see https://pytorch.org/docs/stable/backends.html?highlight=mkl")
    return mkl_enabled


def _to_device(x, device):
    """
    Converts the input tensor or numpy array to the specified device.

    Args:
        x (torch.Tensor or numpy.ndarray): The input tensor or numpy array.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The converted tensor on the specified device.
    """
    if not isinstance(x, torch.Tensor):
        X = torch.from_numpy(x).float().to(device)
        return X
    else:
        return x


def _from_device(X):
    """
    Converts a PyTorch tensor from the device to a NumPy array on the CPU.

    Args:
        X (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    x = X.detach().cpu().numpy()
    return x


def _forward(net, x):
    """Converts images to torch tensors, runs the network model, and returns numpy arrays.

    Args:
        net (torch.nn.Module): The network model.
        x (numpy.ndarray): The input images.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The output predictions (flows and cellprob) and style features.
    """
    X = _to_device(x, net.device)
    net.eval()
    if net.mkldnn:
        net = mkldnn_utils.to_mkldnn(net)
    with torch.no_grad():
        y, style = net(X)[:2]
    del X
    y = _from_device(y)
    style = _from_device(style)
    return y, style


def run_net(net, imgs, batch_size=8, augment=False, tile=True, tile_overlap=0.1,
            bsize=224):
    """ 
    Run network on image or stack of images.
    
    (faster if augment is False)

    Args:
        net (class): cellpose network (model.net)
        imgs (np.ndarray): The input image or stack of images of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile (bool, optional): Tiles image to ensure GPU/CPU memory usage limited (recommended); cannot be turned off for 3D segmentation. Defaults to True.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.

    Returns:
        y (np.ndarray): output of network, if tiled it is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability.
        style (np.ndarray): 1D array of size 256 summarizing the style of the image, if tiled it is averaged over tiles.
    """
    if imgs.ndim == 4:
        # make image Lz x nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        detranspose = (0, 2, 3, 1)
    elif imgs.ndim == 3:
        # make image nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (2, 0, 1))
        detranspose = (1, 2, 0)
    elif imgs.ndim == 2:
        imgs = imgs[np.newaxis, :, :]
        detranspose = (1, 2, 0)

    # pad image for net so Ly and Lx are divisible by 4
    imgs, ysub, xsub = transforms.pad_image_ND(imgs)
    # slices from padding
    #         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size
    slc = [slice(0, imgs.shape[n] + 1) for n in range(imgs.ndim)]
    slc[-3] = slice(0, 3)
    slc[-2] = slice(ysub[0], ysub[-1] + 1)
    slc[-1] = slice(xsub[0], xsub[-1] + 1)
    slc = tuple(slc)

    # run network
    if tile or augment or imgs.ndim == 4:
        y, style = _run_tiled(net, imgs, augment=augment, bsize=bsize,
                              batch_size=batch_size, tile_overlap=tile_overlap)
    else:
        imgs = np.expand_dims(imgs, axis=0)
        y, style = _forward(net, imgs)
        y, style = y[0], style[0]
    style /= (style**2).sum()**0.5

    # slice out padding
    y = y[slc]
    # transpose so channels axis is last again
    y = np.transpose(y, detranspose)

    return y, style


def _run_tiled(net, imgi, batch_size=8, augment=False, bsize=224, tile_overlap=0.1):
    """ 
    Run network on tiles of size [bsize x bsize]
    
    (faster if augment is False)

    Args:
        imgs (np.ndarray): The input image or stack of images of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.

    Returns:
        y (np.ndarray): output of network, if tiled it is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability.
        style (np.ndarray): 1D array of size 256 summarizing the style of the image, if tiled it is averaged over tiles.
    """
    nout = net.nout
    if imgi.ndim == 4:
        Lz, nchan = imgi.shape[:2]
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize,
                                                        augment=augment,
                                                        tile_overlap=tile_overlap)
        ny, nx, nchan, ly, lx = IMG.shape
        batch_size *= max(4, (bsize**2 // (ly * lx))**0.5)
        yf = np.zeros((Lz, nout, imgi.shape[-2], imgi.shape[-1]), np.float32)
        styles = []
        if ny * nx > batch_size:
            ziterator = trange(Lz, file=tqdm_out)
            for i in ziterator:
                yfi, stylei = _run_tiled(net, imgi[i], augment=augment, bsize=bsize,
                                         tile_overlap=tile_overlap)
                yf[i] = yfi
                styles.append(stylei)
        else:
            # run multiple slices at the same time
            ntiles = ny * nx
            nimgs = max(2, int(np.round(batch_size / ntiles)))
            niter = int(np.ceil(Lz / nimgs))
            ziterator = trange(niter, file=tqdm_out)
            for k in ziterator:
                IMGa = np.zeros((ntiles * nimgs, nchan, ly, lx), np.float32)
                for i in range(min(Lz - k * nimgs, nimgs)):
                    IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(
                        imgi[k * nimgs + i], bsize=bsize, augment=augment,
                        tile_overlap=tile_overlap)
                    IMGa[i * ntiles:(i + 1) * ntiles] = np.reshape(
                        IMG, (ny * nx, nchan, ly, lx))
                ya, stylea = _forward(net, IMGa)
                for i in range(min(Lz - k * nimgs, nimgs)):
                    y = ya[i * ntiles:(i + 1) * ntiles]
                    if augment:
                        y = np.reshape(y, (ny, nx, 3, ly, lx))
                        y = transforms.unaugment_tiles(y)
                        y = np.reshape(y, (-1, 3, ly, lx))
                    yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                    yfi = yfi[:, :imgi.shape[2], :imgi.shape[3]]
                    yf[k * nimgs + i] = yfi
                    stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
                    stylei /= (stylei**2).sum()**0.5
                    styles.append(stylei)
        return yf, np.array(styles)
    else:
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize,
                                                        augment=augment,
                                                        tile_overlap=tile_overlap)
        ny, nx, nchan, ly, lx = IMG.shape
        IMG = np.reshape(IMG, (ny * nx, nchan, ly, lx))
        niter = int(np.ceil(IMG.shape[0] / batch_size))
        y = np.zeros((IMG.shape[0], nout, ly, lx))
        for k in range(niter):
            irange = slice(batch_size * k, min(IMG.shape[0],
                                               batch_size * k + batch_size))
            y0, style = _forward(net, IMG[irange])
            y[irange] = y0.reshape(irange.stop - irange.start, y0.shape[-3],
                                   y0.shape[-2], y0.shape[-1])
            # check size models!
            if k == 0:
                styles = style.sum(axis=0)
            else:
                styles += style.sum(axis=0)
        styles /= IMG.shape[0]
        if augment:
            y = np.reshape(y, (ny, nx, nout, bsize, bsize))
            y = transforms.unaugment_tiles(y)
            y = np.reshape(y, (-1, nout, bsize, bsize))

        yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        yf = yf[:, :imgi.shape[1], :imgi.shape[2]]
        styles /= (styles**2).sum()**0.5
        return yf, styles


def run_3D(net, imgs, batch_size=8, rsz=1.0, anisotropy=None, augment=False, tile=True,
           tile_overlap=0.1, bsize=224, progress=None):
    """ 
    Run network on image z-stack.
    
    (faster if augment is False)

    Args:
        imgs (np.ndarray): The input image stack of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile (bool, optional): Tiles image to ensure GPU/CPU memory usage limited (recommended); cannot be turned off for 3D segmentation. Defaults to True.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.
        progress (QProgressBar, optional): pyqt progress bar. Defaults to None.

    Returns:
        y (np.ndarray): output of network, if tiled it is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability.
        style (np.ndarray): 1D array of size 256 summarizing the style of the image, if tiled it is averaged over tiles.
    """
    sstr = ["YX", "ZY", "ZX"]
    if anisotropy is not None:
        rescaling = [[rsz, rsz], [rsz * anisotropy, rsz], [rsz * anisotropy, rsz]]
    else:
        rescaling = [rsz] * 3
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(3, 0, 1, 2), (3, 1, 0, 2), (3, 1, 2, 0)]
    nout = net.nout
    yf = np.zeros((3, nout, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
    for p in range(3):
        xsl = imgs.copy().transpose(pm[p])
        # rescale image for flow computation
        shape = xsl.shape
        xsl = transforms.resize_image(xsl, rsz=rescaling[p])
        # per image
        core_logger.info("running %s: %d planes of size (%d, %d)" %
                         (sstr[p], shape[0], shape[1], shape[2]))
        y, style = run_net(net, xsl, batch_size=batch_size, augment=augment, tile=tile,
                           bsize=bsize, tile_overlap=tile_overlap)
        y = transforms.resize_image(y, shape[1], shape[2])
        yf[p] = y.transpose(ipm[p])
        if progress is not None:
            progress.setValue(25 + 15 * p)
    return yf, style
