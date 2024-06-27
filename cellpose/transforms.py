"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import numpy as np
import warnings
import cv2
import torch
from torch.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter1d

import logging

transforms_logger = logging.getLogger(__name__)

from . import dynamics, utils


def _taper_mask(ly=224, lx=224, sig=7.5):
    """
    Generate a taper mask.

    Args:
        ly (int): The height of the mask. Default is 224.
        lx (int): The width of the mask. Default is 224.
        sig (float): The sigma value for the tapering function. Default is 7.5.

    Returns:
        numpy.ndarray: The taper mask.

    """
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1 / (1 + np.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize // 2 - ly // 2:bsize // 2 + ly // 2 + ly % 2,
                bsize // 2 - lx // 2:bsize // 2 + lx // 2 + lx % 2]
    return mask


def unaugment_tiles(y):
    """Reverse test-time augmentations for averaging (includes flipping of flowsY and flowsX).

    Args:
        y (float32): Array of shape (ntiles_y, ntiles_x, chan, Ly, Lx) where chan = (flowsY, flowsX, cell prob).

    Returns:
        float32: Array of shape (ntiles_y, ntiles_x, chan, Ly, Lx).

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, :]
                y[j, i, 0] *= -1
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = y[j, i, :, :, ::-1]
                y[j, i, 1] *= -1
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, ::-1]
                y[j, i, 0] *= -1
                y[j, i, 1] *= -1
    return y


def average_tiles(y, ysub, xsub, Ly, Lx):
    """
    Average the results of the network over tiles.

    Args:
        y (float): Output of cellpose network for each tile. Shape: [ntiles x nclasses x bsize x bsize]
        ysub (list): List of arrays with start and end of tiles in Y of length ntiles
        xsub (list): List of arrays with start and end of tiles in X of length ntiles
        Ly (int): Size of pre-tiled image in Y (may be larger than original image if image size is less than bsize)
        Lx (int): Size of pre-tiled image in X (may be larger than original image if image size is less than bsize)

    Returns:
        yf (float32): Network output averaged over tiles. Shape: [nclasses x Ly x Lx]
    """
    Navg = np.zeros((Ly, Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf


def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """Make tiles of image to run at test-time.

    Args:
        imgi (np.ndarray): Array of shape (nchan, Ly, Lx) representing the input image.
        bsize (int, optional): Size of tiles. Defaults to 224.
        augment (bool, optional): Whether to flip tiles and set tile_overlap=2. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles. Defaults to 0.1.

    Returns:
        tuple containing
            - IMG (np.ndarray): Array of shape (ntiles, nchan, bsize, bsize) representing the tiles.
            - ysub (list): List of arrays with start and end of tiles in Y of length ntiles.
            - xsub (list): List of arrays with start and end of tiles in X of length ntiles.
            - Ly (int): Height of the input image.
            - Lx (int): Width of the input image.
    """
    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]

    return IMG, ysub, xsub, Ly, Lx


def normalize99(Y, lower=1, upper=99, copy=True):
    """
    Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.

    Args:
        Y (ndarray): The input image.
        lower (int, optional): The lower percentile. Defaults to 1.
        upper (int, optional): The upper percentile. Defaults to 99.
        copy (bool, optional): Whether to create a copy of the input image. Defaults to True.

    Returns:
        ndarray: The normalized image.
    """
    X = Y.copy() if copy else Y
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X = (X - x01) / (x99 - x01)
    else:
        X[:] = 0
    return X


def normalize99_tile(img, blocksize=100, lower=1., upper=99., tile_overlap=0.1,
                     norm3D=False, smooth3D=1, is3D=False):
    """Compute normalization like normalize99 function but in tiles.

    Args:
        img (numpy.ndarray): Array of shape (Lz x) Ly x Lx (x nchan) containing the image.
        blocksize (float, optional): Size of tiles. Defaults to 100.
        lower (float, optional): Lower percentile for normalization. Defaults to 1.0.
        upper (float, optional): Upper percentile for normalization. Defaults to 99.0.
        tile_overlap (float, optional): Fraction of overlap of tiles. Defaults to 0.1.
        norm3D (bool, optional): Use same tiled normalization for each z-plane. Defaults to False.
        smooth3D (int, optional): Smoothing factor for 3D normalization. Defaults to 1.
        is3D (bool, optional): Set to True if image is a 3D stack. Defaults to False.

    Returns:
        numpy.ndarray: Normalized image array of shape (Lz x) Ly x Lx (x nchan).
    """
    shape = img.shape
    is1c = True if img.ndim == 2 or (is3D and img.ndim == 3) else False
    is3D = True if img.ndim > 3 or (is3D and img.ndim == 3) else False
    img = img[..., np.newaxis] if is1c else img
    img = img[np.newaxis, ...] if img.ndim == 3 else img
    Lz, Ly, Lx, nchan = img.shape

    tile_overlap = min(0.5, max(0.05, tile_overlap))
    blocksizeY, blocksizeX = min(blocksize, Ly), min(blocksize, Lx)
    blocksizeY = np.int32(blocksizeY)
    blocksizeX = np.int32(blocksizeX)
    # tiles overlap by 10% tile size
    ny = 1 if Ly <= blocksize else int(np.ceil(
        (1. + 2 * tile_overlap) * Ly / blocksize))
    nx = 1 if Lx <= blocksize else int(np.ceil(
        (1. + 2 * tile_overlap) * Lx / blocksize))
    ystart = np.linspace(0, Ly - blocksizeY, ny).astype(int)
    xstart = np.linspace(0, Lx - blocksizeX, nx).astype(int)
    ysub = []
    xsub = []
    for j in range(len(ystart)):
        for i in range(len(xstart)):
            ysub.append([ystart[j], ystart[j] + blocksizeY])
            xsub.append([xstart[i], xstart[i] + blocksizeX])

    x01_tiles_z = []
    x99_tiles_z = []
    for z in range(Lz):
        IMG = np.zeros((len(ystart), len(xstart), blocksizeY, blocksizeX, nchan),
                       "float32")
        k = 0
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                IMG[j, i] = img[z, ysub[k][0]:ysub[k][1], xsub[k][0]:xsub[k][1], :]
                k += 1
        x01_tiles = np.percentile(IMG, lower, axis=(-3, -2))
        x99_tiles = np.percentile(IMG, upper, axis=(-3, -2))

        # fill areas with small differences with neighboring squares
        to_fill = np.zeros(x01_tiles.shape[:2], "bool")
        for c in range(nchan):
            to_fill = x99_tiles[:, :, c] - x01_tiles[:, :, c] < +1e-3
            if to_fill.sum() > 0 and to_fill.sum() < x99_tiles[:, :, c].size:
                fill_vals = np.nonzero(to_fill)
                fill_neigh = np.nonzero(~to_fill)
                nearest_neigh = (
                    (fill_vals[0] - fill_neigh[0][:, np.newaxis])**2 +
                    (fill_vals[1] - fill_neigh[1][:, np.newaxis])**2).argmin(axis=0)
                x01_tiles[fill_vals[0], fill_vals[1],
                          c] = x01_tiles[fill_neigh[0][nearest_neigh],
                                         fill_neigh[1][nearest_neigh], c]
                x99_tiles[fill_vals[0], fill_vals[1],
                          c] = x99_tiles[fill_neigh[0][nearest_neigh],
                                         fill_neigh[1][nearest_neigh], c]
            elif to_fill.sum() > 0 and to_fill.sum() == x99_tiles[:, :, c].size:
                x01_tiles[:, :, c] = 0
                x99_tiles[:, :, c] = 1
        x01_tiles_z.append(x01_tiles)
        x99_tiles_z.append(x99_tiles)

    x01_tiles_z = np.array(x01_tiles_z)
    x99_tiles_z = np.array(x99_tiles_z)
    # do not smooth over z-axis if not normalizing separately per plane
    for a in range(2):
        x01_tiles_z = gaussian_filter1d(x01_tiles_z, 1, axis=a)
        x99_tiles_z = gaussian_filter1d(x99_tiles_z, 1, axis=a)
    if norm3D:
        smooth3D = 1 if smooth3D == 0 else smooth3D
        x01_tiles_z = gaussian_filter1d(x01_tiles_z, smooth3D, axis=a)
        x99_tiles_z = gaussian_filter1d(x99_tiles_z, smooth3D, axis=a)

    if not norm3D and Lz > 1:
        x01 = np.zeros((len(x01_tiles_z), Ly, Lx, nchan), "float32")
        x99 = np.zeros((len(x01_tiles_z), Ly, Lx, nchan), "float32")
        for z in range(Lz):
            x01_rsz = cv2.resize(x01_tiles_z[z], (Lx, Ly),
                                 interpolation=cv2.INTER_LINEAR)
            x01[z] = x01_rsz[..., np.newaxis] if nchan == 1 else x01_rsz
            x99_rsz = cv2.resize(x99_tiles_z[z], (Lx, Ly),
                                 interpolation=cv2.INTER_LINEAR)
            x99[z] = x99_rsz[..., np.newaxis] if nchan == 1 else x01_rsz
        if (x99 - x01).min() < 1e-3:
            raise ZeroDivisionError(
                "cannot use norm3D=False with tile_norm, sample is too sparse; set norm3D=True or tile_norm=0"
            )
    else:
        x01 = cv2.resize(x01_tiles_z.mean(axis=0), (Lx, Ly),
                         interpolation=cv2.INTER_LINEAR)
        x99 = cv2.resize(x99_tiles_z.mean(axis=0), (Lx, Ly),
                         interpolation=cv2.INTER_LINEAR)
        if x01.ndim < 3:
            x01 = x01[..., np.newaxis]
            x99 = x99[..., np.newaxis]

    if is1c:
        img, x01, x99 = img.squeeze(), x01.squeeze(), x99.squeeze()
    elif not is3D:
        img, x01, x99 = img[0], x01[0], x99[0]

    return (img - x01) / (x99 - x01)


def gaussian_kernel(sigma, Ly, Lx, device=torch.device("cpu")):
    """
    Generates a 2D Gaussian kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        Ly (int): Number of pixels in the y-axis.
        Lx (int): Number of pixels in the x-axis.
        device (torch.device, optional): Device to store the kernel tensor. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: 2D Gaussian kernel tensor.

    """
    y = torch.linspace(-Ly / 2, Ly / 2 + 1, Ly, device=device)
    x = torch.linspace(-Ly / 2, Ly / 2 + 1, Lx, device=device)
    y, x = torch.meshgrid(y, x, indexing="ij")
    kernel = torch.exp(-(y**2 + x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def smooth_sharpen_img(img, smooth_radius=6, sharpen_radius=12,
                       device=torch.device("cpu"), is3D=False):
    """Sharpen blurry images with surround subtraction and/or smooth noisy images.

    Args:
        img (float32): Array that's (Lz x) Ly x Lx (x nchan).
        smooth_radius (float, optional): Size of gaussian smoothing filter, recommended to be 1/10-1/4 of cell diameter
            (if also sharpening, should be 2-3x smaller than sharpen_radius). Defaults to 6.
        sharpen_radius (float, optional): Size of gaussian surround filter, recommended to be 1/8-1/2 of cell diameter
            (if also smoothing, should be 2-3x larger than smooth_radius). Defaults to 12.
        device (torch.device, optional): Device on which to perform sharpening.
            Will be faster on GPU but need to ensure GPU has RAM for image. Defaults to torch.device("cpu").
        is3D (bool, optional): If image is 3D stack (only necessary to set if img.ndim==3). Defaults to False.

    Returns:
        img_sharpen (float32): Array that's (Lz x) Ly x Lx (x nchan).
    """
    img_sharpen = torch.from_numpy(img.astype("float32")).to(device)
    shape = img_sharpen.shape

    is1c = True if img_sharpen.ndim == 2 or (is3D and img_sharpen.ndim == 3) else False
    is3D = True if img_sharpen.ndim > 3 or (is3D and img_sharpen.ndim == 3) else False
    img_sharpen = img_sharpen.unsqueeze(-1) if is1c else img_sharpen
    img_sharpen = img_sharpen.unsqueeze(0) if img_sharpen.ndim == 3 else img_sharpen
    Lz, Ly, Lx, nchan = img_sharpen.shape

    if smooth_radius > 0:
        kernel = gaussian_kernel(smooth_radius, Ly, Lx, device=device)
        if sharpen_radius > 0:
            kernel += -1 * gaussian_kernel(sharpen_radius, Ly, Lx, device=device)
    elif sharpen_radius > 0:
        kernel = -1 * gaussian_kernel(sharpen_radius, Ly, Lx, device=device)
        kernel[Ly // 2, Lx // 2] = 1

    fhp = fft2(kernel)
    for z in range(Lz):
        for c in range(nchan):
            img_filt = torch.real(ifft2(
                fft2(img_sharpen[z, :, :, c]) * torch.conj(fhp)))
            img_filt = fftshift(img_filt)
            img_sharpen[z, :, :, c] = img_filt

    img_sharpen = img_sharpen.reshape(shape)
    return img_sharpen.cpu().numpy()


def move_axis(img, m_axis=-1, first=True):
    """ move axis m_axis to first or last position """
    if m_axis == -1:
        m_axis = img.ndim - 1
    m_axis = min(img.ndim - 1, m_axis)
    axes = np.arange(0, img.ndim)
    if first:
        axes[1:m_axis + 1] = axes[:m_axis]
        axes[0] = m_axis
    else:
        axes[m_axis:-1] = axes[m_axis + 1:]
        axes[-1] = m_axis
    img = img.transpose(tuple(axes))
    return img


def move_min_dim(img, force=False):
    """Move the minimum dimension last as channels if it is less than 10 or force is True.

    Args:
        img (ndarray): The input image.
        force (bool, optional): If True, the minimum dimension will always be moved. 
            Defaults to False.

    Returns:
        ndarray: The image with the minimum dimension moved to the last axis as channels.
    """
    if len(img.shape) > 2:
        min_dim = min(img.shape)
        if min_dim < 10 or force:
            if img.shape[-1] == min_dim:
                channel_axis = -1
            else:
                channel_axis = (img.shape).index(min_dim)
            img = move_axis(img, m_axis=channel_axis, first=False)
    return img


def update_axis(m_axis, to_squeeze, ndim):
    """
    Squeeze the axis value based on the given parameters.

    Args:
        m_axis (int): The current axis value.
        to_squeeze (numpy.ndarray): An array of indices to squeeze.
        ndim (int): The number of dimensions.

    Returns:
        int or None: The updated axis value.
    """
    if m_axis == -1:
        m_axis = ndim - 1
    if (to_squeeze == m_axis).sum() == 1:
        m_axis = None
    else:
        inds = np.ones(ndim, bool)
        inds[to_squeeze] = False
        m_axis = np.nonzero(np.arange(0, ndim)[inds] == m_axis)[0]
        if len(m_axis) > 0:
            m_axis = m_axis[0]
        else:
            m_axis = None
    return m_axis


def convert_image(x, channels, channel_axis=None, z_axis=None, do_3D=False, nchan=2):
    """Converts the image to have the z-axis first, channels last, and normalized intensities.

    Args:
        x (numpy.ndarray or torch.Tensor): The input image.
        channels (list or None): The list of channels to use (ones-based, 0=gray). If None, all channels are kept.
        channel_axis (int or None): The axis of the channels in the input image. If None, the axis is determined automatically.
        z_axis (int or None): The axis of the z-dimension in the input image. If None, the axis is determined automatically.
        do_3D (bool): Whether to process the image in 3D mode. Defaults to False.
        nchan (int): The number of channels to keep if the input image has more than nchan channels.

    Returns:
        numpy.ndarray: The converted image.

    Raises:
        ValueError: If the input image has less than two channels and channels are not specified.
        ValueError: If the input image is 2D and do_3D is True.
        ValueError: If the input image is 4D and do_3D is False.
    """
    # check if image is a torch array instead of numpy array
    # converts torch to numpy
    if torch.is_tensor(x):
        transforms_logger.warning("torch array used as input, converting to numpy")
        x = x.cpu().numpy()

    # squeeze image, and if channel_axis or z_axis given, transpose image
    if x.ndim > 3:
        to_squeeze = np.array([int(isq) for isq, s in enumerate(x.shape) if s == 1])
        # remove channel axis if number of channels is 1
        if len(to_squeeze) > 0:
            channel_axis = update_axis(
                channel_axis, to_squeeze,
                x.ndim) if channel_axis is not None else channel_axis
            z_axis = update_axis(z_axis, to_squeeze,
                                 x.ndim) if z_axis is not None else z_axis
        x = x.squeeze()

    # put z axis first
    if z_axis is not None and x.ndim > 2 and z_axis != 0:
        x = move_axis(x, m_axis=z_axis, first=True)
        if channel_axis is not None:
            channel_axis += 1
        z_axis = 0

    if z_axis is not None:
        if x.ndim == 3:
            x = x[..., np.newaxis]

    # put channel axis last
    if channel_axis is not None and x.ndim > 2:
        x = move_axis(x, m_axis=channel_axis, first=False)
    elif x.ndim == 2:
        x = x[:, :, np.newaxis]

    if do_3D:
        if x.ndim < 3:
            transforms_logger.critical("ERROR: cannot process 2D images in 3D mode")
            raise ValueError("ERROR: cannot process 2D images in 3D mode")
        elif x.ndim < 4:
            x = x[..., np.newaxis]

    if channel_axis is None:
        x = move_min_dim(x)

    if x.ndim > 3:
        transforms_logger.info(
            "multi-stack tiff read in as having %d planes %d channels" %
            (x.shape[0], x.shape[-1]))

    if channels is not None:
        channels = channels[0] if len(channels) == 1 else channels
        if len(channels) < 2:
            transforms_logger.critical("ERROR: two channels not specified")
            raise ValueError("ERROR: two channels not specified")
        x = reshape(x, channels=channels)

    else:
        # code above put channels last
        if nchan is not None and x.shape[-1] > nchan:
            transforms_logger.warning(
                "WARNING: more than %d channels given, use 'channels' input for specifying channels - just using first %d channels to run processing"
                % (nchan, nchan))
            x = x[..., :nchan]

        #if not do_3D and x.ndim > 3:
        #    transforms_logger.critical("ERROR: cannot process 4D images in 2D mode")
        #    raise ValueError("ERROR: cannot process 4D images in 2D mode")

        if nchan is not None and x.shape[-1] < nchan:
            x = np.concatenate((x, np.tile(np.zeros_like(x), (1, 1, nchan - 1))),
                               axis=-1)

    return x


def reshape(data, channels=[0, 0], chan_first=False):
    """Reshape data using channels.

    Args:
        data (numpy.ndarray): The input data. It should have shape (Z x ) Ly x Lx x nchan
            if data.ndim==8 and data.shape[0]<8, it is assumed to be nchan x Ly x Lx.
        channels (list of int, optional): The channels to use for reshaping. The first element
            of the list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). The
            second element of the list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
            For instance, to train on grayscale images, input [0,0]. To train on images with cells
            in green and nuclei in blue, input [2,3]. Defaults to [0, 0].
        chan_first (bool, optional): Whether to return the reshaped data with channel as the first
            dimension. Defaults to False.

    Returns:
        numpy.ndarray: The reshaped data with shape (Z x ) Ly x Lx x nchan (if chan_first==False).
    """
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:, :, np.newaxis]
    elif data.shape[0] < 8 and data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))

    # use grayscale image
    if data.shape[-1] == 1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0] == 0:
            data = data.mean(axis=-1, keepdims=True)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0] - 1]
            if channels[1] > 0:
                chanid.append(channels[1] - 1)
            data = data[..., chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[..., i]) == 0.0:
                    if i == 0:
                        warnings.warn("'chan to seg' to seg has value range of ZERO")
                    else:
                        warnings.warn(
                            "'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0"
                        )
            if data.shape[-1] == 1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data


def normalize_img(img, normalize=True, norm3D=False, invert=False, lowhigh=None,
                  percentile=None, sharpen_radius=0, smooth_radius=0,
                  tile_norm_blocksize=0, tile_norm_smooth3D=1, axis=-1):
    """Normalize each channel of the image.

    Args:
        img (ndarray): The input image. It should have at least 3 dimensions.
            If it is 4-dimensional, it assumes the first non-channel axis is the Z dimension.
        normalize (bool, optional): Whether to perform normalization. Defaults to True.
        norm3D (bool, optional): Whether to normalize in 3D. Defaults to False.
        invert (bool, optional): Whether to invert the image. Useful if cells are dark instead of bright.
            Defaults to False.
        lowhigh (tuple, optional): The lower and upper bounds for normalization. If provided, it should be a tuple
            of two values. Defaults to None.
        percentile (tuple, optional): The lower and upper percentiles for normalization. If provided, it should be
            a tuple of two values. Each value should be between 0 and 100. Defaults to None.
        sharpen_radius (int, optional): The radius for sharpening the image. Defaults to 0.
        smooth_radius (int, optional): The radius for smoothing the image. Defaults to 0.
        tile_norm_blocksize (int, optional): The block size for tile-based normalization. Defaults to 0.
        tile_norm_smooth3D (int, optional): The smoothness factor for tile-based normalization in 3D. Defaults to 1.
        axis (int, optional): The channel axis to loop over for normalization. Defaults to -1.

    Returns:
        ndarray: The normalized image of the same size.

    Raises:
        ValueError: If the image has less than 3 dimensions.
        ValueError: If the provided lowhigh or percentile values are invalid.
        ValueError: If the image is inverted without normalization.

    """
    if img.ndim < 3:
        error_message = "Image needs to have at least 3 dimensions"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if lowhigh is not None:
        assert len(lowhigh) == 2
        assert lowhigh[1] > lowhigh[0]
    elif percentile is not None:
        assert len(percentile) == 2
        assert percentile[0] >= 0 and percentile[1] > 0
        assert percentile[0] < 100 and percentile[1] <= 100
        assert percentile[1] > percentile[0]
    else:
        percentile = [1., 99.]

    img_norm = img.astype(np.float32)
    # move channel axis last
    img_norm = np.moveaxis(img_norm, axis, -1)
    nchan = img_norm.shape[-1]

    if lowhigh is not None:
        for c in range(nchan):
            img_norm[...,
                     c] = (img_norm[..., c] - lowhigh[0]) / (lowhigh[1] - lowhigh[0])
    else:
        if sharpen_radius > 0 or smooth_radius > 0:
            img_norm = smooth_sharpen_img(img_norm, sharpen_radius=sharpen_radius,
                                          smooth_radius=smooth_radius)

        if tile_norm_blocksize > 0:
            img_norm = normalize99_tile(img_norm, blocksize=tile_norm_blocksize,
                                        lower=percentile[0], upper=percentile[1],
                                        smooth3D=tile_norm_smooth3D, norm3D=norm3D)
        elif normalize:
            if img_norm.ndim == 3 or norm3D:
                for c in range(nchan):
                    img_norm[..., c] = normalize99(img_norm[...,
                                                            c], lower=percentile[0],
                                                   upper=percentile[1], copy=False)
            else:
                for z in range(img_norm.shape[0]):
                    for c in range(nchan):
                        img_norm[z, :, :,
                                 c] = normalize99(img_norm[z, :, :,
                                                           c], lower=percentile[0],
                                                  upper=percentile[1], copy=False)
        if (tile_norm_blocksize > 0 or normalize) and invert:
            img_norm[..., c] = -1 * img_norm[..., c] + 1
        elif invert:
            error_message = "cannot invert image without normalizing"
            transforms_logger.critical(error_message)
            raise ValueError(error_message)

    # move channel axis back to original position
    img_norm = np.moveaxis(img_norm, -1, axis)

    return img_norm


def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR,
                 no_channels=False):
    """Resize image for computing flows / unresize for computing dynamics.

    Args:
        img0 (ndarray): Image of size [Y x X x nchan] or [Lz x Y x X x nchan] or [Lz x Y x X].
        Ly (int, optional): Desired height of the resized image. Defaults to None.
        Lx (int, optional): Desired width of the resized image. Defaults to None.
        rsz (float, optional): Resize coefficient(s) for the image. If Ly is None, rsz is used. Defaults to None.
        interpolation (int, optional): OpenCV interpolation method. Defaults to cv2.INTER_LINEAR.
        no_channels (bool, optional): Flag indicating whether to treat the third dimension as a channel. 
            Defaults to False.

    Returns:
        ndarray: Resized image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan].

    Raises:
        ValueError: If Ly is None and rsz is None.

    """
    if Ly is None and rsz is None:
        error_message = "must give size to resize to or factor to use for resizing"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        if no_channels:
            Ly = int(img0.shape[-2] * rsz[-2])
            Lx = int(img0.shape[-1] * rsz[-1])
        else:
            Ly = int(img0.shape[-3] * rsz[-2])
            Lx = int(img0.shape[-2] * rsz[-1])

    # no_channels useful for z-stacks, so the third dimension is not treated as a channel
    # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but
    if (img0.ndim > 2 and no_channels) or (img0.ndim == 4 and not no_channels):
        if Ly == 0 or Lx == 0:
            raise ValueError(
                "anisotropy too high / low -- not enough pixels to resize to ratio")
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i, img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs


def pad_image_ND(img0, div=16, extra=1, min_size=None):
    """Pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D).

    Args:
        img0 (ndarray): Image of size [nchan (x Lz) x Ly x Lx].
        div (int, optional): Divisor for padding. Defaults to 16.
        extra (int, optional): Extra padding. Defaults to 1.
        min_size (tuple, optional): Minimum size of the image. Defaults to None.

    Returns:   
        tuple containing 
            - I (ndarray): Padded image.
            - ysub (ndarray): Y range of pixels in the padded image corresponding to img0.
            - xsub (ndarray): X range of pixels in the padded image corresponding to img0.
    """
    if min_size is None or img0.shape[-2] >= min_size[-2]:
        Lpad = int(div * np.ceil(img0.shape[-2] / div) - img0.shape[-2])
    else:
        Lpad = min_size[-2] - img0.shape[-2]
    xpad1 = extra * div // 2 + Lpad // 2
    xpad2 = extra * div // 2 + Lpad - Lpad // 2
    if min_size is None or img0.shape[-1] >= min_size[-1]:
        Lpad = int(div * np.ceil(img0.shape[-1] / div) - img0.shape[-1])
    else:
        Lpad = min_size[-1] - img0.shape[-1]
    ypad1 = extra * div // 2 + Lpad // 2
    ypad2 = extra * div // 2 + Lpad - Lpad // 2

    if img0.ndim > 3:
        pads = np.array([[0, 0], [0, 0], [xpad1, xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0, 0], [xpad1, xpad2], [ypad1, ypad2]])

    I = np.pad(img0, pads, mode="constant")

    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1 + Ly)
    xsub = np.arange(ypad1, ypad1 + Lx)

    return I, ysub, xsub


def random_rotate_and_resize(X, Y=None, scale_range=1., xy=(224, 224), do_3D=False,
                             do_flip=True, rotate=True, rescale=None, unet=False,
                             random_per_image=True):
    """Augmentation by random rotation and resizing.

    Args:
        X (list of ND-arrays, float): List of image arrays of size [nchan x Ly x Lx] or [Ly x Lx].
        Y (list of ND-arrays, float, optional): List of image labels of size [nlabels x Ly x Lx] or [Ly x Lx].
            The 1st channel of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow].
            If unet, second channel is dist_to_bound. Defaults to None.
        scale_range (float, optional): Range of resizing of images for augmentation.
            Images are resized by (1-scale_range/2) + scale_range * np.random.rand(). Defaults to 1.0.
        xy (tuple, int, optional): Size of transformed images to return. Defaults to (224,224).
        do_flip (bool, optional): Whether or not to flip images horizontally. Defaults to True.
        rotate (bool, optional): Whether or not to rotate images. Defaults to True.
        rescale (array, float, optional): How much to resize images by before performing augmentations. Defaults to None.
        unet (bool, optional): Whether or not to use unet. Defaults to False.
        random_per_image (bool, optional): Different random rotate and resize per image. Defaults to True.

    Returns:
        tuple containing
            - imgi (ND-array, float): Transformed images in array [nimg x nchan x xy[0] x xy[1]].
            - lbl (ND-array, float): Transformed labels in array [nimg x nchan x xy[0] x xy[1]].
            - scale (array, float): Amount each image was resized by.
    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    if do_3D and X[0].ndim > 3:
        shape = (X[0].shape[-3], xy[0], xy[1])
    else:
        shape = (xy[0], xy[1])
    imgi = np.zeros((nimg, nchan, *shape), np.float32)

    lbl = []
    if Y is not None:
        if Y[0].ndim > 2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, *shape), np.float32)

    scale = np.ones(nimg, np.float32)

    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        if random_per_image or n == 0:
            # generate random augmentation parameters
            flip = np.random.rand() > .5
            theta = np.random.rand() * np.pi * 2 if rotate else 0.
            scale[n] = (1 - scale_range / 2) + scale_range * np.random.rand()
            if rescale is not None:
                scale[n] *= 1. / rescale[n]
            dxy = np.maximum(0, np.array([Lx * scale[n] - xy[1],
                                          Ly * scale[n] - xy[0]]))
            dxy = (np.random.rand(2,) - .5) * dxy

            # create affine transform
            cc = np.array([Lx / 2, Ly / 2])
            cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy
            pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
            pts2 = np.float32([
                cc1,
                cc1 + scale[n] * np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale[n] *
                np.array([np.cos(np.pi / 2 + theta),
                          np.sin(np.pi / 2 + theta)])
            ])
            M = cv2.getAffineTransform(pts1, pts2)

        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim < 3:
                labels = labels[np.newaxis, :, :]

        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[-1] = -labels[-1]

        for k in range(nchan):
            if do_3D:
                for z in range(shape[0]):
                    I = cv2.warpAffine(img[k, z], M, (xy[1], xy[0]),
                                       flags=cv2.INTER_LINEAR)
                    imgi[n, k, z] = I
            else:
                I = cv2.warpAffine(img[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
                imgi[n, k] = I

        if Y is not None:
            for k in range(nt):
                flag = cv2.INTER_NEAREST if k == 0 else cv2.INTER_LINEAR
                if do_3D:
                    for z in range(shape[0]):
                        lbl[n, k, z] = cv2.warpAffine(labels[k, z], M, (xy[1], xy[0]),
                                                      flags=flag)
                else:
                    lbl[n, k] = cv2.warpAffine(labels[k], M, (xy[1], xy[0]), flags=flag)

            if nt > 1 and not unet:
                v1 = lbl[n, -1].copy()
                v2 = lbl[n, -2].copy()
                lbl[n, -2] = (-v1 * np.sin(-theta) + v2 * np.cos(-theta))
                lbl[n, -1] = (v1 * np.cos(-theta) + v2 * np.sin(-theta))

    return imgi, lbl, scale
