"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os, time, datetime
import numpy as np
from scipy.stats import mode
import cv2
import torch
from torch import nn
from torch.nn.functional import conv2d, interpolate
from tqdm import trange
from pathlib import Path

import logging

denoise_logger = logging.getLogger(__name__)

from cellpose import transforms

def deterministic(seed=0):
    """ set random seeds to create test data """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def img_norm(imgi):
    """
    Normalizes the input image by subtracting the 1st percentile and dividing by the difference between the 99th and 1st percentiles.

    Args:
        imgi (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    shape = imgi.shape
    imgi = imgi.reshape(imgi.shape[0], imgi.shape[1], -1)
    perc = torch.quantile(imgi, torch.tensor([0.01, 0.99], device=imgi.device), dim=-1,
                          keepdim=True)
    for k in range(imgi.shape[1]):
        hask = (perc[1, :, k, 0] - perc[0, :, k, 0]) > 1e-3
        imgi[hask, k] -= perc[0, hask, k]
        imgi[hask, k] /= (perc[1, hask, k] - perc[0, hask, k])
    imgi = imgi.reshape(shape)
    return imgi


def add_noise(lbl, alpha=4, beta=0.7, poisson=0.7, blur=0.7, gblur=1.0, downsample=0.7,
              ds_max=7, diams=None, pscale=None, iso=True, sigma0=None, sigma1=None,
              ds=None, uniform_blur=False, partial_blur=False):
    """Adds noise to the input image.

    Args:
        lbl (torch.Tensor): The input image tensor of shape (nimg, nchan, Ly, Lx).
        alpha (float, optional): The shape parameter of the gamma distribution used for generating poisson noise. Defaults to 4.
        beta (float, optional): The rate parameter of the gamma distribution used for generating poisson noise. Defaults to 0.7.
        poisson (float, optional): The probability of adding poisson noise to the image. Defaults to 0.7.
        blur (float, optional): The probability of adding gaussian blur to the image. Defaults to 0.7.
        gblur (float, optional): The scale factor for the gaussian blur. Defaults to 1.0.
        downsample (float, optional): The probability of downsampling the image. Defaults to 0.7.
        ds_max (int, optional): The maximum downsampling factor. Defaults to 7.
        diams (torch.Tensor, optional): The diameter of the objects in the image. Defaults to None.
        pscale (torch.Tensor, optional): The scale factor for the poisson noise, instead of sampling. Defaults to None.
        iso (bool, optional): Whether to use isotropic gaussian blur. Defaults to True.
        sigma0 (torch.Tensor, optional): The standard deviation of the gaussian filter for the Y axis, instead of sampling. Defaults to None.
        sigma1 (torch.Tensor, optional): The standard deviation of the gaussian filter for the X axis, instead of sampling. Defaults to None.
        ds (torch.Tensor, optional): The downsampling factor for each image, instead of sampling. Defaults to None.

    Returns:
        torch.Tensor: The noisy image tensor of the same shape as the input image.
    """
    device = lbl.device
    imgi = torch.zeros_like(lbl)
    Ly, Lx = lbl.shape[-2:]

    diams = diams if diams is not None else 30. * torch.ones(len(lbl), device=device)
    #ds0 = 1 if ds is None else ds.item()
    ds = ds * torch.ones(
        (len(lbl),), device=device, dtype=torch.long) if ds is not None else ds

    # downsample
    ii = []
    idownsample = np.random.rand(len(lbl)) < downsample
    if (ds is None and idownsample.sum() > 0.) or not iso:
        ds = torch.ones(len(lbl), dtype=torch.long, device=device)
        ds[idownsample] = torch.randint(2, ds_max + 1, size=(idownsample.sum(),),
                                        device=device)
        ii = torch.nonzero(ds > 1).flatten()
    elif ds is not None and (ds > 1).sum():
        ii = torch.nonzero(ds > 1).flatten()

    # add gaussian blur
    iblur = torch.rand(len(lbl), device=device) < blur
    iblur[ii] = True
    if iblur.sum() > 0:
        if sigma0 is None:
            if uniform_blur and iso:
                xr = torch.rand(len(lbl), device=device)
                if len(ii) > 0:
                    xr[ii] = ds[ii].float() / 2. / gblur
                sigma0 = diams[iblur] / 30. * gblur * (1 / gblur + (1 - 1 / gblur) * xr[iblur])
                sigma1 = sigma0.clone()
            elif not iso:
                xr = torch.rand(len(lbl), device=device)
                if len(ii) > 0:
                    xr[ii] = (ds[ii].float()) / gblur
                    xr[ii] = xr[ii] + torch.rand(len(ii), device=device) * 0.7 - 0.35
                    xr[ii] = torch.clip(xr[ii], 0.05, 1.5)
                sigma0 = diams[iblur] / 30. * gblur * xr[iblur]
                sigma1 = sigma0.clone() / 10.
            else:
                xrand = np.random.exponential(1, size=iblur.sum())
                xrand = np.clip(xrand * 0.5, 0.1, 1.0)
                xrand *= gblur
                sigma0 = diams[iblur] / 30. * 5. * torch.from_numpy(xrand).float().to(
                    device)
                sigma1 = sigma0.clone()
        else:
            sigma0 = sigma0 * torch.ones((iblur.sum(),), device=device)
            sigma1 = sigma1 * torch.ones((iblur.sum(),), device=device)

        # create gaussian filter
        xr = max(8, sigma0.max().long() * 2)
        gfilt0 = torch.exp(-torch.arange(-xr + 1, xr, device=device)**2 /
                           (2 * sigma0.unsqueeze(-1)**2))
        gfilt0 /= gfilt0.sum(axis=-1, keepdims=True)
        gfilt1 = torch.zeros_like(gfilt0)
        gfilt1[sigma1 == sigma0] = gfilt0[sigma1 == sigma0]
        gfilt1[sigma1 != sigma0] = torch.exp(
            -torch.arange(-xr + 1, xr, device=device)**2 /
            (2 * sigma1[sigma1 != sigma0].unsqueeze(-1)**2))
        gfilt1[sigma1 == 0] = 0.
        gfilt1[sigma1 == 0, xr] = 1.
        gfilt1 /= gfilt1.sum(axis=-1, keepdims=True)
        gfilt = torch.einsum("ck,cl->ckl", gfilt0, gfilt1)
        gfilt /= gfilt.sum(axis=(1, 2), keepdims=True)

        lbl_blur = conv2d(lbl[iblur].transpose(1, 0), gfilt.unsqueeze(1),
                             padding=gfilt.shape[-1] // 2,
                             groups=gfilt.shape[0]).transpose(1, 0)
        if partial_blur:
            #yc, xc = np.random.randint(100, Ly-100), np.random.randint(100, Lx-100)
            imgi[iblur] = lbl[iblur].clone()
            Lxc = int(Lx * 0.85)
            ym, xm = torch.meshgrid(torch.zeros(Ly, dtype=torch.float32), 
                                    torch.arange(0, Lxc, dtype=torch.float32), 
                        indexing="ij")
            mask = torch.exp(-(ym**2 + xm**2) / 2*(0.001**2))
            mask -= mask.min()
            mask /= mask.max()
            lbl_blur_crop = lbl_blur[:, :, :, :Lxc]
            imgi[iblur, :, :, :Lxc] = (lbl_blur_crop * mask + 
                                (1-mask) * imgi[iblur, :, :, :Lxc])
        else:
            imgi[iblur] = lbl_blur

    imgi[~iblur] = lbl[~iblur]

    # apply downsample
    for k in ii:
        i0 = imgi[k:k + 1, :, ::ds[k], ::ds[k]] if iso else imgi[k:k + 1, :, ::ds[k]]
        imgi[k] = interpolate(i0, size=lbl[k].shape[-2:], mode="bilinear")

    # add poisson noise
    ipoisson = np.random.rand(len(lbl)) < poisson
    if ipoisson.sum() > 0:
        if pscale is None:
            pscale = torch.zeros(len(lbl))
            m = torch.distributions.gamma.Gamma(alpha, beta)
            pscale = torch.clamp(m.rsample(sample_shape=(ipoisson.sum(),)), 1.)
            #pscale = torch.clamp(20 * (torch.rand(size=(len(lbl),), device=lbl.device)), 1.5)
            pscale = pscale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        else:
            pscale = pscale * torch.ones((ipoisson.sum(), 1, 1, 1), device=device)
        imgi[ipoisson] = torch.poisson(pscale * imgi[ipoisson].clamp_(min=0.0))
    imgi[~ipoisson] = imgi[~ipoisson]

    # renormalize
    imgi = img_norm(imgi)

    return imgi
