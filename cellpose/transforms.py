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
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize//2-ly//2 : bsize//2+ly//2+ly%2, 
                bsize//2-lx//2 : bsize//2+lx//2+lx%2]
    return mask

def unaugment_tiles(y, unet=False):
    """ reverse test-time augmentations for averaging

    Parameters
    ----------

    y: float32
        array that's ntiles_y x ntiles_x x chan x Ly x Lx where chan = (dY, dX, cell prob)

    unet: bool (optional, False)
        whether or not unet output or cellpose output
    
    Returns
    -------

    y: float32

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j%2==0 and i%2==1:
                y[j,i] = y[j,i, :,::-1, :]
                if not unet:
                    y[j,i,0] *= -1
            elif j%2==1 and i%2==0:
                y[j,i] = y[j,i, :,:, ::-1]
                if not unet:
                    y[j,i,1] *= -1
            elif j%2==1 and i%2==1:
                y[j,i] = y[j,i, :,::-1, ::-1]
                if not unet:
                    y[j,i,0] *= -1
                    y[j,i,1] *= -1
    return y

def average_tiles(y, ysub, xsub, Ly, Lx):
    """ average results of network over tiles

    Parameters
    -------------

    y: float, [ntiles x nclasses x bsize x bsize]
        output of cellpose network for each tile

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    Ly : int
        size of pre-tiled image in Y (may be larger than original image if
        image size is less than bsize)

    Lx : int
        size of pre-tiled image in X (may be larger than original image if
        image size is less than bsize)

    Returns
    -------------

    yf: float32, [nclasses x Ly x Lx]
        network output averaged over tiles

    """
    Navg = np.zeros((Ly,Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf

def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """ make tiles of image to run at test-time

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    
    """

    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize-Ly, Lx))), axis=1)
            Ly = bsize
        if Lx<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize-Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly-bsize, ny).astype(int)
        xstart = np.linspace(0, Lx-bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsize])
                xsub.append([xstart[i], xstart[i]+bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j%2==0 and i%2==1:
                    IMG[j,i] = IMG[j,i, :,::-1, :]
                elif j%2==1 and i%2==0:
                    IMG[j,i] = IMG[j,i, :,:, ::-1]
                elif j%2==1 and i%2==1:
                    IMG[j,i] = IMG[j,i,:, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly<=bsize else int(np.ceil((1.+2*tile_overlap) * Ly / bsize))
        nx = 1 if Lx<=bsize else int(np.ceil((1.+2*tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly-bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx-bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsizeY])
                xsub.append([xstart[i], xstart[i]+bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
        
    return IMG, ysub, xsub, Ly, Lx

def normalize99(Y, lower=1, upper=99, copy=True):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy() if copy else Y
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X = (X - x01) / (x99 - x01)
    else:
        X[:] = 0
    return X

def normalize99_tile(img, blocksize=100, lower=1., upper=99.,
                     tile_overlap=0.1, norm3D=False, smooth3D=1, 
                     is3D=False):
    """ compute normalization like normalize99 function but in tiles 
    
    Parameters
    ----------
    img : float32
        array that's (Lz x) x Ly x Lx (x nchan)

    blocksize : float (optional, default 100)
        size of tiles

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    norm3D: bool (optional, default False)
        use same tiled normalization for each z-plane

    is3D: bool (optional, default False)
        if image is 3D stack (only necessary to set if img.ndim==3)

    Returns
    -------
    img_norm : float32
        array that's (Lz x) x Ly x Lx (x nchan)
    
    """
    shape = img.shape
    is1c = True if img.ndim == 2 or (is3D and img.ndim==3) else False
    is3D = True if img.ndim > 3 or (is3D and img.ndim==3) else False
    img = img[...,np.newaxis] if is1c else img
    img = img[np.newaxis, ...] if img.ndim==3 else img
    Lz, Ly, Lx, nchan = img.shape
    
    tile_overlap = min(0.5, max(0.05, tile_overlap))
    blocksizeY, blocksizeX = min(blocksize, Ly), min(blocksize, Lx)
    blocksizeY = np.int32(blocksizeY)
    blocksizeX = np.int32(blocksizeX)
    # tiles overlap by 10% tile size
    ny = 1 if Ly<=blocksize else int(np.ceil((1.+2*tile_overlap) * Ly / blocksize))
    nx = 1 if Lx<=blocksize else int(np.ceil((1.+2*tile_overlap) * Lx / blocksize))
    ystart = np.linspace(0, Ly-blocksizeY, ny).astype(int)
    xstart = np.linspace(0, Lx-blocksizeX, nx).astype(int)
    ysub = []
    xsub = []
    for j in range(len(ystart)):
        for i in range(len(xstart)):
            ysub.append([ystart[j], ystart[j]+blocksizeY])
            xsub.append([xstart[i], xstart[i]+blocksizeX])

    x01_tiles_z = []
    x99_tiles_z = []
    for z in range(Lz):            
        IMG = np.zeros((len(ystart), len(xstart),  blocksizeY, blocksizeX, nchan), 'float32')
        k = 0
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                IMG[j, i] = img[z, ysub[k][0]:ysub[k][1],  xsub[k][0]:xsub[k][1], :]
                k += 1
        x01_tiles = np.percentile(IMG, lower, axis=(-3,-2))
        x99_tiles = np.percentile(IMG, upper, axis=(-3,-2))
        
        # fill areas with small differences with neighboring squares
        to_fill = np.zeros(x01_tiles.shape[:2], 'bool')
        for c in range(nchan):
            to_fill = x99_tiles[:,:,c] - x01_tiles[:,:,c] < +1e-3
            if to_fill.sum() > 0 and to_fill.sum() < x99_tiles[:,:,c].size:
                fill_vals = np.nonzero(to_fill)
                fill_neigh = np.nonzero(~to_fill)
                nearest_neigh = ((fill_vals[0] - fill_neigh[0][:,np.newaxis])**2 + 
                                    (fill_vals[1] - fill_neigh[1][:,np.newaxis])**2).argmin(axis=0)
                x01_tiles[fill_vals[0], fill_vals[1], c] = x01_tiles[fill_neigh[0][nearest_neigh], 
                                                                     fill_neigh[1][nearest_neigh], c]
                x99_tiles[fill_vals[0], fill_vals[1], c] = x99_tiles[fill_neigh[0][nearest_neigh], 
                                                                     fill_neigh[1][nearest_neigh], c]
            elif to_fill.sum() > 0 and to_fill.sum() == x99_tiles[:,:,c].size:
                x01_tiles[:,:,c] = 0
                x99_tiles[:,:,c] = 1
        x01_tiles_z.append(x01_tiles)
        x99_tiles_z.append(x99_tiles)
        
    x01_tiles_z = np.array(x01_tiles_z)
    x99_tiles_z = np.array(x99_tiles_z)
    # do not smooth over z-axis if not normalizing separately per plane
    for a in range(2):
        x01_tiles_z = gaussian_filter1d(x01_tiles_z, 1, axis=a)
        x99_tiles_z = gaussian_filter1d(x99_tiles_z, 1, axis=a)
    if norm3D:
        smooth3D = 1 if smooth3D==0 else smooth3D
        x01_tiles_z = gaussian_filter1d(x01_tiles_z, smooth3D, axis=a)
        x99_tiles_z = gaussian_filter1d(x99_tiles_z, smooth3D, axis=a)
    
    if not norm3D and Lz > 1:
        x01 = np.zeros((len(x01_tiles_z), Ly, Lx, nchan), 'float32')
        x99 = np.zeros((len(x01_tiles_z), Ly, Lx, nchan), 'float32')
        for z in range(Lz):
            x01_rsz = cv2.resize(x01_tiles_z[z], (Lx, Ly), interpolation=cv2.INTER_LINEAR)
            x01[z] = x01_rsz[..., np.newaxis] if nchan==1 else x01_rsz
            x99_rsz = cv2.resize(x99_tiles_z[z], (Lx, Ly), interpolation=cv2.INTER_LINEAR)
            x99[z] = x99_rsz[..., np.newaxis] if nchan==1 else x01_rsz
        if (x99 - x01).min() < 1e-3:
            raise ZeroDivisionError('cannot use norm3D=False with tile_norm, sample is too sparse; set norm3D=True or tile_norm=0')
    else:
        x01 = cv2.resize(x01_tiles_z.mean(axis=0), (Lx, Ly), interpolation=cv2.INTER_LINEAR)
        x99 = cv2.resize(x99_tiles_z.mean(axis=0), (Lx, Ly), interpolation=cv2.INTER_LINEAR)
        if x01.ndim < 3:
            x01 = x01[..., np.newaxis]
            x99 = x99[..., np.newaxis]
        
    if is1c:
        img, x01, x99 = img.squeeze(), x01.squeeze(), x99.squeeze()
    elif not is3D:
        img, x01, x99 = img[0], x01[0], x99[0]

    return (img - x01) / (x99 - x01)

def gaussian_kernel(sigma, Ly, Lx, device=torch.device('cpu')):
    y = torch.linspace(-Ly/2, Ly/2+1, Ly, device=device)
    x = torch.linspace(-Ly/2, Ly/2+1, Lx, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')
    kernel = torch.exp(-(y**2 + x**2) / (2*sigma**2)) 
    kernel /= kernel.sum()
    return kernel

def smooth_sharpen_img(img, smooth_radius=6, sharpen_radius=12, device=torch.device('cpu'), is3D=False):
    """ sharpen blurry images with surround subtraction and/or smooth noisy images, sigma recommended to be 1/4-1/8 of cell diameter in pixels

    sharpen subtracts image filtered with gaussian kernel with sharpen_radius; 
    set smooth_radius for gaussian smoothing

    Parameters
    ----------
    img : float32
        array that's (Lz x) Ly x Lx (x nchan)

    smooth_radius : float (optional, default 4)
        size of gaussian smoothing filter, recommended to be 1/10-1/4 of cell diameter
        (if also sharpening, should be 2-3x smaller than sharpen_radius)

    sharpen_radius : float (optional, default 10)
        size of gaussian surround filter, recommended to be 1/8-1/2 of cell diameter
        (if also smoothing, should be 2-3x larger than smooth_radius)

    device: torch.device (optional, default 'cpu')
        device on which to perform sharpening 
        (will be faster on GPU but need to ensure GPU has RAM for image)

    is3D: bool (optional, default False)
        if image is 3D stack (only necessary to set if img.ndim==3)

    Returns
    -------
    img_sharpen : float32
        array that's (Lz x) Ly x Lx (x nchan)

    """
    img_sharpen = torch.from_numpy(img.astype('float32')).to(device)
    shape = img_sharpen.shape 
    
    is1c = True if img_sharpen.ndim == 2 or (is3D and img_sharpen.ndim==3) else False
    is3D = True if img_sharpen.ndim > 3 or (is3D and img_sharpen.ndim==3) else False
    img_sharpen = img_sharpen.unsqueeze(-1) if is1c else img_sharpen
    img_sharpen = img_sharpen.unsqueeze(0) if img_sharpen.ndim==3 else img_sharpen
    Lz, Ly, Lx, nchan = img_sharpen.shape
    
    if smooth_radius > 0:
        kernel = gaussian_kernel(smooth_radius, Ly, Lx, device=device)
        if sharpen_radius > 0:
            kernel += -1 * gaussian_kernel(sharpen_radius, Ly, Lx, device=device)
    elif sharpen_radius > 0:
        kernel = -1 * gaussian_kernel(sharpen_radius, Ly, Lx, device=device)
        kernel[Ly//2, Lx//2] = 1
    
    fhp = fft2(kernel)
    for z in range(Lz):
        for c in range(nchan):
            img_filt = torch.real(ifft2(fft2(img_sharpen[z,:,:,c]) * torch.conj(fhp)))
            img_filt = fftshift(img_filt)
            img_sharpen[z,:,:,c] = img_filt

    img_sharpen = img_sharpen.reshape(shape)
    return img_sharpen.cpu().numpy()

def move_axis(img, m_axis=-1, first=True):
    """ move axis m_axis to first or last position """
    if m_axis==-1:
        m_axis = img.ndim-1
    m_axis = min(img.ndim-1, m_axis)
    axes = np.arange(0, img.ndim)
    if first:
        axes[1:m_axis+1] = axes[:m_axis]
        axes[0] = m_axis
    else:
        axes[m_axis:-1] = axes[m_axis+1:]
        axes[-1] = m_axis
    img = img.transpose(tuple(axes))
    return img

# This was edited to fix a bug where single-channel images of shape (y,x) would be 
# transposed to (x,y) if x<y, making the labels no longer correspond to the data. 
def move_min_dim(img, force=False):
    """ move minimum dimension last as channels if < 10, or force==True """
    if len(img.shape) > 2: #only makese sense to do this if channel axis is already present 
        min_dim = min(img.shape)
        if min_dim < 10 or force:
            if img.shape[-1]==min_dim:
                channel_axis = -1
            else:
                channel_axis = (img.shape).index(min_dim)
            img = move_axis(img, m_axis=channel_axis, first=False)
    return img

def update_axis(m_axis, to_squeeze, ndim):
    if m_axis==-1:
        m_axis = ndim-1
    if (to_squeeze==m_axis).sum() == 1:
        m_axis = None
    else:
        inds = np.ones(ndim, bool)
        inds[to_squeeze] = False
        m_axis = np.nonzero(np.arange(0, ndim)[inds]==m_axis)[0]
        if len(m_axis) > 0:
            m_axis = m_axis[0]
        else:
            m_axis = None
    return m_axis

def convert_image(x, channels, channel_axis=None, z_axis=None,
                  do_3D=False, nchan=2):
    """ return image with z first, channels last and normalized intensities """
    # check if image is a torch array instead of numpy array
    # converts torch to numpy
    if torch.is_tensor(x):
        transforms_logger.warning('torch array used as input, converting to numpy')
        x = x.cpu().numpy()
    
    # squeeze image, and if channel_axis or z_axis given, transpose image
    if x.ndim > 3:
        to_squeeze = np.array([int(isq) for isq,s in enumerate(x.shape) if s==1])
        # remove channel axis if number of channels is 1
        if len(to_squeeze) > 0: 
            channel_axis = update_axis(channel_axis, to_squeeze, x.ndim) if channel_axis is not None else channel_axis
            z_axis = update_axis(z_axis, to_squeeze, x.ndim) if z_axis is not None else z_axis
        x = x.squeeze()

    # put z axis first
    if z_axis is not None and x.ndim > 2 and z_axis != 0:
        x = move_axis(x, m_axis=z_axis, first=True)
        if channel_axis is not None:
            channel_axis += 1
        if x.ndim==3:
            x = x[...,np.newaxis]
        
    # put channel axis last
    if channel_axis is not None and x.ndim > 2:
        x = move_axis(x, m_axis=channel_axis, first=False)
    elif x.ndim == 2:
        x = x[:,:,np.newaxis]

    if do_3D :
        if x.ndim < 3:
            transforms_logger.critical('ERROR: cannot process 2D images in 3D mode')
            raise ValueError('ERROR: cannot process 2D images in 3D mode') 
        elif x.ndim<4:
            x = x[...,np.newaxis]

    if channel_axis is None:
        x = move_min_dim(x)
        
    if x.ndim > 3:
        transforms_logger.info('multi-stack tiff read in as having %d planes %d channels'%
                (x.shape[0], x.shape[-1]))

    if channels is not None:
        channels = channels[0] if len(channels)==1 else channels
        if len(channels) < 2:
            transforms_logger.critical('ERROR: two channels not specified')
            raise ValueError('ERROR: two channels not specified') 
        x = reshape(x, channels=channels)
        
    else:
        # code above put channels last
        if x.shape[-1] > nchan:
            transforms_logger.warning('WARNING: more than %d channels given, use "channels" input for specifying channels - just using first %d channels to run processing'%(nchan,nchan))
            x = x[...,:nchan]

        if not do_3D and x.ndim>3:
            transforms_logger.critical('ERROR: cannot process 4D images in 2D mode')
            raise ValueError('ERROR: cannot process 4D images in 2D mode')
            
        if x.shape[-1] < nchan:
            x = np.concatenate((x, 
                                np.tile(np.zeros_like(x), (1,1,nchan-1))), 
                                axis=-1)
      
    return x

def reshape(data, channels=[0,0], chan_first=False):
    """ reshape data using channels

    Parameters
    ----------

    data : numpy array that's (Z x ) Ly x Lx x nchan
        if data.ndim==8 and data.shape[0]<8, assumed to be nchan x Ly x Lx

    channels : list of int of length 2 (optional, default [0,0])
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    invert : bool
        invert intensities

    Returns
    -------
    data : numpy array that's (Z x ) Ly x Lx x nchan (if chan_first==False)

    """
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:,:,np.newaxis]
    elif data.shape[0]<8 and data.ndim==3:
        data = np.transpose(data, (1,2,0))

    # use grayscale image
    if data.shape[-1]==1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0]==0:
            data = data.mean(axis=-1, keepdims=True)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0]-1]
            if channels[1] > 0:
                chanid.append(channels[1]-1)
            data = data[...,chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[...,i]) == 0.0:
                    if i==0:
                        warnings.warn("chan to seg' has value range of ZERO")
                    else:
                        warnings.warn("'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            if data.shape[-1]==1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim==4:
            data = np.transpose(data, (3,0,1,2))
        else:
            data = np.transpose(data, (2,0,1))
    return data

def normalize_img(img, normalize=True, norm3D=False, invert=False, 
                  lowhigh=None, percentile=None,
                  sharpen_radius=0, smooth_radius=0,
                  tile_norm_blocksize=0, tile_norm_smooth3D=1, axis=-1):
    """ normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities


    Parameters
    ------------

    img: ND-array (at least 3 dimensions)
        if 4 dimensional, assumes 1st non-channel axis is Z dimension

    axis: channel axis to loop over for normalization

    invert: invert image (useful if cells are dark instead of bright)

    Returns
    ---------------

    img_norm: ND-array, float32
        normalized image of same size

    """
    
    if img.ndim<3:
        error_message = 'Image needs to have at least 3 dimensions'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if lowhigh is not None:
        assert len(lowhigh)==2
        assert lowhigh[1] > lowhigh[0]
    elif percentile is not None:
        assert len(percentile)==2
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
            img_norm[...,c] = (img_norm[...,c] - lowhigh[0]) / (lowhigh[1] - lowhigh[0])
    else:    
        if sharpen_radius > 0 or smooth_radius > 0:
            img_norm = smooth_sharpen_img(img_norm, sharpen_radius=sharpen_radius, 
                                          smooth_radius=smooth_radius)
            
        if tile_norm_blocksize > 0:
            img_norm = normalize99_tile(img_norm, blocksize=tile_norm_blocksize, 
                                        lower=percentile[0], upper=percentile[1],
                                        smooth3D=tile_norm_smooth3D, 
                                        norm3D=norm3D)
        elif normalize:
            if img_norm.ndim==3 or norm3D:
                for c in range(nchan):
                    img_norm[...,c] = normalize99(img_norm[...,c],
                                                lower=percentile[0], upper=percentile[1], 
                                                copy=False)
            else:
                for z in range(img_norm.shape[0]):
                    for c in range(nchan):
                        img_norm[z,:,:,c] = normalize99(img_norm[z,:,:,c],
                                                        lower=percentile[0], upper=percentile[1], 
                                                        copy=False)
        if (tile_norm_blocksize > 0 or normalize) and invert:
            img_norm[...,c] = -1*img_norm[...,c] + 1   
        elif invert:
            error_message = 'cannot invert image without normalizing'
            transforms_logger.critical(error_message)
            raise ValueError(error_message)
                
    # move channel axis back to original position
    img_norm = np.moveaxis(img_norm, -1, axis)
    
    return img_norm

def reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize=True):
    """ check sizes and reshape train and test data for training """
    nimg = len(train_data)
    # check that arrays are correct size
    if nimg != len(train_labels):
        error_message = 'train data and labels not same length'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        
    if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
        error_message = 'training data or labels are not at least two-dimensional'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        

    if train_data[0].ndim > 3:
        error_message = 'training data is more than three-dimensional (should be 2D or 3D array)'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        

    # check if test_data correct length
    if not (test_data is not None and test_labels is not None and
            len(test_data) > 0 and len(test_data)==len(test_labels)):
        test_data = None

    # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
    train_data, test_data, run_test = reshape_and_normalize_data(train_data, test_data=test_data, 
                                                                 channels=channels, normalize=normalize)

    if train_data is None:
        error_message = 'training data do not all have the same number of channels'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        

    if not run_test:
        test_data, test_labels = None, None

    return train_data, train_labels, test_data, test_labels, run_test

def reshape_and_normalize_data(train_data, test_data=None, channels=None, normalize=True):
    """ inputs converted to correct shapes for *training* and rescaled so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities in each channel

    Parameters
    --------------

    train_data: list of ND-arrays, float
        list of training images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    channels: list of int of length 2 (optional, default None)
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    normalize: bool or dict (optional, True)
        normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel, 
        or can pass in normalize_params as a dictionary for "normalize_img" function

    Returns
    -------------

    train_data: list of ND-arrays, float
        list of training images of size [2 x Ly x Lx]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [2 x Ly x Lx]

    run_test: bool
        whether or not test_data was correct size and is useable during training

    """

    # if training data is less than 2D
    run_test = False
    for test, data in enumerate([train_data, test_data]):
        if data is None:
            return train_data, test_data, run_test
        nimg = len(data)
        for i in range(nimg):
            if channels is not None:
                data[i] = move_min_dim(data[i], force=True)
                data[i] = reshape(data[i], channels=channels, chan_first=True)
            if data[i].ndim < 3:
                data[i] = data[i][np.newaxis,:,:]
            if normalize:
                if isinstance(normalize, dict):
                    data[i] = normalize_img(data[i], axis=0, **normalize)
                else:
                    data[i] = normalize_img(data[i], axis=0)

        nchan = [data[i].shape[0] for i in range(nimg)]
    run_test = True
    return train_data, test_data, run_test

def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False):
    """ resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [Y x X x nchan] or [Lz x Y x X x nchan] or [Lz x Y x X]

    Ly: int, optional

    Lx: int, optional

    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used

    interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)

    Returns
    --------------

    imgs: ND-array 
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    """
    if Ly is None and rsz is None:
        error_message = 'must give size to resize to or factor to use for resizing'
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
    
    # no_channels useful for z-stacks, sot he third dimension is not treated as a channel
    # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but 
    if (img0.ndim>2 and no_channels) or (img0.ndim==4 and not no_channels):
        if Ly==0 or Lx==0:
            raise ValueError('anisotropy too high / low -- not enough pixels to resize to ratio')
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i,img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs

def pad_image_ND(img0, div=16, extra=1, min_size=None):
    """ pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D)

    Parameters
    -------------

    img0: ND-array
        image of size [nchan (x Lz) x Ly x Lx]

    div: int (optional, default 16)

    Returns
    --------------

    I: ND-array
        padded image

    ysub: array, int
        yrange of pixels in I corresponding to img0

    xsub: array, int
        xrange of pixels in I corresponding to img0

    """
    if min_size is None or img0.shape[-2] >= min_size[-2]:
        Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    else:
        Lpad = min_size[-2] - img0.shape[-2]
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    if min_size is None or img0.shape[-1] >= min_size[-1]:
        Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    else:
        Lpad = min_size[-1] - img0.shape[-1]
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0,pads, mode='constant')

    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1+Ly)
    xsub = np.arange(ypad1, ypad1+Lx)
    
    return I, ysub, xsub

def normalize_field(mu):
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)
    return mu

def _X2zoom(img, X2=1):
    """ zoom in image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    Returns
    -------
    img : numpy array that's Ly x Lx

    """
    ny,nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2**X2)), int(ny * (2**X2))))
    return img

def _image_resizer(img, resize=512, to_uint8=False):
    """ resize image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    resize : int
        max size of image returned

    to_uint8 : bool
        convert image to uint8

    Returns
    -------
    img : numpy array that's Ly x Lx, Ly,Lx<resize

    """
    ny,nx = img.shape[:2]
    if to_uint8:
        if img.max()<=255 and img.min()>=0 and img.max()>1:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
    if np.array(img.shape).max() > resize:
        if ny>nx:
            nx = int(nx/ny * resize)
            ny = resize
        else:
            ny = int(ny/nx * resize)
            nx = resize
        shape = (nx,ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img


def random_rotate_and_resize(X, Y=None, scale_range=1., xy = (224,224), do_3D=False,
                             do_flip=True, rotate=True,
                             rescale=None, unet=False, random_per_image=True):
    """ augmentation by random rotation and resizing
        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)
        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow]. 
            If unet, second channel is dist_to_bound.
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        xy: tuple, int (optional, default (224,224))
            size of transformed images to return
        do_flip: bool (optional, default True)
            whether or not to flip images horizontally
        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations
        unet: bool (optional, default False)
        random_per_image: bool (optional, default True)
            different random rotate and resize per image
        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]
        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        scale: array, float
            amount each image was resized by
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
    imgi  = np.zeros((nimg, nchan, *shape), np.float32)

    lbl = []
    if Y is not None:
        if Y[0].ndim>2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, *shape), np.float32)

    scale = np.ones(nimg, np.float32)
    
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        if random_per_image or n==0:
            # generate random augmentation parameters
            flip = np.random.rand()>.5
            theta = np.random.rand() * np.pi * 2 if rotate else 0.
            scale[n] = (1-scale_range/2) + scale_range * np.random.rand()
            if rescale is not None:
                scale[n] *= 1. / rescale[n]
            dxy = np.maximum(0, np.array([Lx*scale[n]-xy[1],Ly*scale[n]-xy[0]]))
            dxy = (np.random.rand(2,) - .5) * dxy

            # create affine transform
            cc = np.array([Lx/2, Ly/2])
            cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
            pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
            pts2 = np.float32([cc1,
                    cc1 + scale[n]*np.array([np.cos(theta), np.sin(theta)]),
                    cc1 + scale[n]*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
            M = cv2.getAffineTransform(pts1,pts2)

        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim<3:
                labels = labels[np.newaxis,:,:]

        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[-1] = -labels[-1]

        for k in range(nchan):
            if do_3D:
                for z in range(shape[0]):
                    I = cv2.warpAffine(img[k,z], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
                    imgi[n,k,z] = I
            else:
                I = cv2.warpAffine(img[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
                imgi[n,k] = I

        if Y is not None:
            for k in range(nt):
                flag = cv2.INTER_NEAREST if k==0 else cv2.INTER_LINEAR
                if do_3D:
                    for z in range(shape[0]):
                        lbl[n,k,z] = cv2.warpAffine(labels[k,z], M, (xy[1],xy[0]), flags=flag)
                else:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=flag)

            if nt > 1 and not unet:
                v1 = lbl[n,-1].copy()
                v2 = lbl[n,-2].copy()
                lbl[n,-2] = (-v1 * np.sin(-theta) + v2*np.cos(-theta))
                lbl[n,-1] = (v1 * np.cos(-theta) + v2*np.sin(-theta))

    return imgi, lbl, scale
