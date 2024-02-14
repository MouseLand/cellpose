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
from . import transforms, dynamics, utils, plot, metrics

import torch
#     from GPUtil import showUtilization as gpu_usage #for gpu memory debugging 
from torch import nn
from torch.utils import mkldnn as mkldnn_utils
from . import resnet_torch
TORCH_ENABLED = True

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)

def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        cp = False
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        cp = True
        nclasses = 3
    else:
        return 3, True, True, False
    
    if 'residual' in model_str and 'style' in model_str and 'concatentation' in model_str:
        ostrs = model_str.split('_')[2::2]
        residual_on = ostrs[0]=='on'
        style_on = ostrs[1]=='on'
        concatenation = ostrs[2]=='on'
        return nclasses, residual_on, style_on, concatenation
    else:
        if cp:
            return 3, True, True, False
        else:
            return nclasses, False, False, True

def use_gpu(gpu_number=0, use_torch=True):
    """ check if gpu works """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError('cellpose only runs with pytorch now')

def _use_gpu_torch(gpu_number=0):
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        core_logger.info('** TORCH CUDA version installed and working. **')
        return True
    except:
        core_logger.info('TORCH CUDA version not installed/working.')
        return False

def assign_device(use_torch=True, gpu=False, device=0):
    mac = False
    cpu = True
    if isinstance(device, str):
        if device=='mps':
            mac = True 
        else:
            device = int(device)
    if gpu and use_gpu(use_torch=True):
        device = torch.device(f'cuda:{device}')
        gpu=True
        cpu=False
        core_logger.info('>>>> using GPU')
    elif mac:
        try:
            device = torch.device('mps')
            gpu=True
            cpu=False
            core_logger.info('>>>> using GPU')
        except:
            cpu = True 
            gpu = False

    if cpu:
        device = torch.device('cpu')
        core_logger.info('>>>> using CPU')
        gpu=False
    return device, gpu

def check_mkl(use_torch=True):
    #core_logger.info('Running test snippet to check if MKL-DNN working')
    mkl_enabled = torch.backends.mkldnn.is_available()
    if mkl_enabled:
        mkl_enabled = True
        #core_logger.info('MKL version working - CPU version is sped up.')
    else:
        core_logger.info('WARNING: MKL version on torch not working/installed - CPU version will be slightly slower.')
        core_logger.info('see https://pytorch.org/docs/stable/backends.html?highlight=mkl')
    return mkl_enabled

def _to_device(x, device):
    if not isinstance(x, torch.Tensor):
        X = torch.from_numpy(x).float().to(device)
        return X
    else:
        return x

def _from_device(X):
    x = X.detach().cpu().numpy()
    return x

def _forward(net, x):
    """ convert imgs to torch and run network model and return numpy """
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

def run_net(net, imgs, augment=False, tile=True, tile_overlap=0.1, bsize=224):
    """ run network on image or stack of images

    (faster if augment is False)

    Parameters
    --------------

    imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    rsz: float (optional, default 1.0)
        resize coefficient(s) for image

    augment: bool (optional, default False)
        tiles image with overlapping tiles and flips overlapped regions to augment

    tile: bool (optional, default True)
        tiles image to ensure GPU/CPU memory usage limited (recommended);
        cannot be turned off for 3D segmentation

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles when computing flows

    bsize: int (optional, default 224)
        size of tiles to use in pixels [bsize x bsize]

    Returns
    ------------------

    y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
        y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

    style: array [64]
        1D array summarizing the style of the image,
        if tiled it is averaged over tiles

    """   
    if imgs.ndim==4:  
        # make image Lz x nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (0,3,1,2)) 
        detranspose = (0,2,3,1)
    elif imgs.ndim==3:
        # make image nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (2,0,1))
        detranspose = (1,2,0)
    elif imgs.ndim==2:
        imgs = imgs[np.newaxis,:,:]
        detranspose = (1,2,0)

    # pad image for net so Ly and Lx are divisible by 4
    imgs, ysub, xsub = transforms.pad_image_ND(imgs)
    # slices from padding
#         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size 
    slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
    slc[-3] = slice(0, 3)
    slc[-2] = slice(ysub[0], ysub[-1]+1)
    slc[-1] = slice(xsub[0], xsub[-1]+1)
    slc = tuple(slc)

    # run network
    if tile or augment or imgs.ndim==4:
        y, style = _run_tiled(net, imgs, augment=augment, bsize=bsize, 
                                    tile_overlap=tile_overlap)
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
    """ run network in tiles of size [bsize x bsize]

    First image is split into overlapping tiles of size [bsize x bsize].
    If augment, tiles have 50% overlap and are flipped at overlaps.
    The average of the network output over tiles is returned.

    Parameters
    --------------

    imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

    augment: bool (optional, default False)
        tiles image with overlapping tiles and flips overlapped regions to augment

    bsize: int (optional, default 224)
        size of tiles to use in pixels [bsize x bsize]
        
    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles when computing flows

    Returns
    ------------------

    yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
        yf is averaged over tiles
        yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

    styles: array [64]
        1D array summarizing the style of the image, averaged over tiles

    """
    nout = net.nout
    if imgi.ndim==4:
        Lz, nchan = imgi.shape[:2]
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize, 
                                                        augment=augment, tile_overlap=tile_overlap)
        ny, nx, nchan, ly, lx = IMG.shape
        batch_size *= max(4, (bsize**2 // (ly*lx))**0.5)
        yf = np.zeros((Lz, nout, imgi.shape[-2], imgi.shape[-1]), np.float32)
        styles = []
        if ny*nx > batch_size:
            ziterator = trange(Lz, file=tqdm_out)
            for i in ziterator:
                yfi, stylei = _run_tiled(net, imgi[i], augment=augment, 
                                                bsize=bsize, tile_overlap=tile_overlap)
                yf[i] = yfi
                styles.append(stylei)
        else:
            # run multiple slices at the same time
            ntiles = ny*nx
            nimgs = max(2, int(np.round(batch_size / ntiles)))
            niter = int(np.ceil(Lz/nimgs))
            ziterator = trange(niter, file=tqdm_out)
            for k in ziterator:
                IMGa = np.zeros((ntiles*nimgs, nchan, ly, lx), np.float32)
                for i in range(min(Lz-k*nimgs, nimgs)):
                    IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[k*nimgs+i], bsize=bsize, 
                                                                    augment=augment, tile_overlap=tile_overlap)
                    IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, ly, lx))
                ya, stylea = _forward(net, IMGa)
                for i in range(min(Lz-k*nimgs, nimgs)):
                    y = ya[i*ntiles:(i+1)*ntiles]
                    if augment:
                        y = np.reshape(y, (ny, nx, 3, ly, lx))
                        y = transforms.unaugment_tiles(y)
                        y = np.reshape(y, (-1, 3, ly, lx))
                    yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                    yfi = yfi[:,:imgi.shape[2],:imgi.shape[3]]
                    yf[k*nimgs+i] = yfi
                    stylei = stylea[i*ntiles:(i+1)*ntiles].sum(axis=0)
                    stylei /= (stylei**2).sum()**0.5
                    styles.append(stylei)
        return yf, np.array(styles)
    else:
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, 
                                                        augment=augment, tile_overlap=tile_overlap)
        ny, nx, nchan, ly, lx = IMG.shape
        IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
        niter = int(np.ceil(IMG.shape[0] / batch_size))
        y = np.zeros((IMG.shape[0], nout, ly, lx))
        for k in range(niter):
            irange = slice(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
            y0, style = _forward(net, IMG[irange])
            y[irange] = y0.reshape(irange.stop-irange.start, y0.shape[-3], y0.shape[-2], y0.shape[-1])
            # check size models!
            if k==0:
                styles = style.sum(axis=0)
            else:
                styles += style.sum(axis=0)
        styles /= IMG.shape[0]
        if augment:
            y = np.reshape(y, (ny, nx, nout, bsize, bsize))
            y = transforms.unaugment_tiles(y)
            y = np.reshape(y, (-1, nout, bsize, bsize))
        
        yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
        styles /= (styles**2).sum()**0.5
        return yf, styles

def run_3D(net, imgs, rsz=1.0, anisotropy=None, 
            augment=False, tile=True, tile_overlap=0.1, 
            bsize=224, progress=None):
    """ run network on stack of images

    (faster if augment is False)

    Parameters
    --------------

    imgs: array [Lz x Ly x Lx x nchan]

    rsz: float (optional, default 1.0)
        resize coefficient(s) for image

    anisotropy: float (optional, default None)
            for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

    augment: bool (optional, default False)
        tiles image with overlapping tiles and flips overlapped regions to augment

    tile: bool (optional, default True)
        tiles image to ensure GPU/CPU memory usage limited (recommended);
        cannot be turned off for 3D segmentation

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles when computing flows

    bsize: int (optional, default 224)
        size of tiles to use in pixels [bsize x bsize]

    progress: pyqt progress bar (optional, default None)
        to return progress bar status to GUI


    Returns
    ------------------

    yf: array [Lz x Ly x Lx x 3]
        y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

    style: array [64]
        1D array summarizing the style of the image,
        if tiled it is averaged over tiles

    """ 
    sstr = ['YX', 'ZY', 'ZX']
    if anisotropy is not None:
        rescaling = [[rsz, rsz],
                        [rsz*anisotropy, rsz],
                        [rsz*anisotropy, rsz]]
    else:
        rescaling = [rsz] * 3
    pm = [(0,1,2,3), (1,0,2,3), (2,0,1,3)]
    ipm = [(3,0,1,2), (3,1,0,2), (3,1,2,0)]
    nout = net.nout
    yf = np.zeros((3, nout, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
    for p in range(3):
        xsl = imgs.copy().transpose(pm[p])
        # rescale image for flow computation
        shape = xsl.shape
        xsl = transforms.resize_image(xsl, rsz=rescaling[p])  
        # per image
        core_logger.info('running %s: %d planes of size (%d, %d)'%(sstr[p], shape[0], shape[1], shape[2]))
        y, style = run_net(net, xsl, augment=augment, tile=tile, 
                                    bsize=bsize, tile_overlap=tile_overlap)
        y = transforms.resize_image(y, shape[1], shape[2])    
        yf[p] = y.transpose(ipm[p])
        if progress is not None:
            progress.setValue(25+15*p)
    return yf, style