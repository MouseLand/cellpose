"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import time, os
from scipy.ndimage import maximum_filter1d, find_objects, center_of_mass
import torch
import numpy as np
import tifffile
from tqdm import trange
from numba import njit, prange, float32, int32, vectorize
import cv2
import fastremap

import logging
dynamics_logger = logging.getLogger(__name__)

from . import utils, metrics, transforms

import torch
from torch import optim, nn
import torch.nn.functional as F
from . import resnet_torch
TORCH_ENABLED = True 
torch_GPU = torch.device('cuda')
torch_CPU = torch.device('cpu')

@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T

def _extend_centers_gpu(neighbors, meds, isneighbor, shape, n_iter=200, device=torch.device('cuda')):
    """ runs diffusion on GPU to generate flows for training images or quality control
    
    neighbors is 9 x pixels in masks, 
    centers are mask centers, 
    isneighbor is valid neighbor boolean 9 x pixels
    
    """
    if device is None:
        device = torch.device("cuda")
    
    T = torch.zeros(shape, dtype=torch.double, device=device)
    
    for i in range(n_iter):
        T[meds[:,0], meds[:,1]] +=1
        Tneigh = T[tuple(neighbors)] #neighbors[0], neighbors[1]]
        Tneigh *= isneighbor
        T[tuple(neighbors[:,0])] = Tneigh.mean(axis=0)
    del meds, isneighbor, Tneigh
    
    # gradient positions
    if T.ndim==2:
        grads = T[neighbors[0, [2,1,4,3]], neighbors[1, [2,1,4,3]]]
        del neighbors
        dy = grads[0] - grads[1]
        dx = grads[2] - grads[3]
        del grads
        mu_torch = np.stack((dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
    else:
        grads = T[:, pt[1:,:,0], pt[1:,:,1], pt[1:,:,2]]
        del pt
        dz = grads[:,0] - grads[:,1]
        dy = grads[:,2] - grads[:,3]
        dx = grads[:,4] - grads[:,5]
        del grads
        mu_torch = np.stack((dz.cpu().squeeze(0), dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
    return mu_torch

# def _extend_centers_gpu_3d(neighbors, centers, isneighbor, Lz, Ly, Lx, n_iter=200, device=torch.device('cuda')):
#     """ runs diffusion on GPU to generate flows for training images or quality control
    
#     neighbors is 9 x pixels in masks, 
#     centers are mask centers, 
#     isneighbor is valid neighbor boolean 9 x pixels
    
#     """
#     if device is not None:
#         device = device
#     nimg = neighbors.shape[0] // 7
#     pt = torch.from_numpy(neighbors).to(device)
    
#     T = torch.zeros((nimg,Lz,Ly,Lx), dtype=torch.double, device=device)
#     meds = torch.from_numpy(centers.astype(int)).to(device).long()
#     isneigh = torch.from_numpy(isneighbor).to(device)
#     for i in range(n_iter):
#         T[:, meds[:,0], meds[:,1], meds[:,2]] +=1
#         Tneigh = T[:, pt[...,0], pt[...,1], pt[...,2]]
#         Tneigh *= isneigh
#         T[:, pt[0,:,0], pt[0,:,1], pt[0,:,2]] = Tneigh.mean(axis=1)
#     del meds, isneigh, Tneigh
#     T = torch.log(1.+ T)
#     # gradient positions
#     grads = T[:, pt[1:,:,0], pt[1:,:,1], pt[1:,:,2]]
#     del pt
#     dz = grads[:,0] - grads[:,1]
#     dy = grads[:,2] - grads[:,3]
#     dx = grads[:,4] - grads[:,5]
#     del grads
#     mu_torch = np.stack((dz.cpu().squeeze(0), dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
#     return mu_torch


@njit(nogil=True)
def get_centers(masks, slices):
    centers = np.zeros((len(slices), 2), "int32")
    ext = np.zeros((len(slices),), "int32")
    for p in prange(len(slices)):
        si = slices[p]
        i = si[0]
        sr, sc = si[1:3], si[3:5]        
        # find center in slice around mask
        yi, xi = np.nonzero(masks[sr[0]:sr[-1], sc[0]:sc[-1]] == (i+1))
        ymed = yi.mean()
        xmed = xi.mean()
        # center is closest point to (ymed, xmed) within mask
        imin = ((xi-xmed)**2 + (yi-ymed)**2).argmin()
        ymed = yi[imin] + sr[0]
        xmed = xi[imin] + sc[0]
        centers[p] = np.array([ymed, xmed])    
        ext[p] = (sr[-1] - sr[0]) + (sc[-1] - sc[0]) + 2
    return centers, ext

def masks_to_flows_gpu(masks, device=None, niter=None):

    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined using COM
    Parameters
    -------------
    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 
    """
    if device is None:
        device = torch.device("cuda")
    
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2, Lx0+2

    masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
    masks_padded = F.pad(masks_padded, (1,1,1,1))

    ### get mask pixel neighbors
    y, x = torch.nonzero(masks_padded, as_tuple=True)
    neighborsY = torch.stack((y, y-1, y+1, 
                            y, y, y-1, 
                            y-1, y+1, y+1), dim=0)
    neighborsX = torch.stack((x, x, x, 
                            x-1, x+1, x-1, 
                            x+1, x-1, x+1), dim=0)
    neighbors = torch.stack((neighborsY, neighborsX), dim=0)
    neighbor_masks = masks_padded[neighbors[0], neighbors[1]]
    isneighbor = neighbor_masks == neighbor_masks[0]

    ### get center-of-mass within cell
    slices = find_objects(masks)
    # turn slices into array
    slices = np.array([np.array([i, si[0].start, si[0].stop, si[1].start, si[1].stop]) 
                        for i, si in enumerate(slices) if si is not None])
    centers, ext = get_centers(masks, slices)
    meds_p = torch.from_numpy(centers).to(device).long()
    meds_p += 1 # for padding

    ### run diffusion
    n_iter = 2 * ext.max() if niter is None else niter
    shape = masks_padded.shape
    mu = _extend_centers_gpu(neighbors, meds_p, isneighbor, shape,
                                    n_iter=n_iter, device=device)

    # new normalization
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)
    #mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() -1] = mu
    
    return mu0, meds_p.cpu().numpy() - 1


def masks_to_flows_gpu_3d(masks, device=None):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined using COM
    Parameters
    -------------
    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 
    """
    if device is None:
        device = torch.device('cuda')

    
    Lz0, Ly0, Lx0 = masks.shape
    Lz, Ly, Lx = Lz0+2, Ly0+2, Lx0+2

    masks_padded = np.zeros((Lz, Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1, 1:-1] = masks

    # get mask pixel neighbors
    z, y, x = np.nonzero(masks_padded)
    neighborsZ = np.stack((z, z+1, z-1,
                            z, z, 
                            z, z))
    neighborsY = np.stack((y, y, y,
                           y+1, y-1, 
                           y, y), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x, x,
                           x+1, x-1), axis=0)

    neighbors = np.stack((neighborsZ, neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = find_objects(masks)
    
    centers = np.zeros((masks.max(), 3), 'int')
    for i,si in enumerate(slices):
        if si is not None:
            sz, sy, sx = si
            #lz, ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            zi, yi, xi = np.nonzero(masks[sz, sy, sx] == (i+1))
            zi = zi.astype(np.int32) + 1 # add padding
            yi = yi.astype(np.int32) + 1 # add padding
            xi = xi.astype(np.int32) + 1 # add padding
            zmed = np.mean(zi)
            ymed = np.mean(yi)
            xmed = np.mean(xi)
            imin = np.argmin((zi-zmed)**2 + (xi-xmed)**2 + (yi-ymed)**2)
            zmed = zi[imin]
            ymed = yi[imin]
            xmed = xi[imin]
            centers[i,0] = zmed + sz.start 
            centers[i,1] = ymed + sy.start 
            centers[i,2] = xmed + sx.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1], neighbors[:,:,2]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array([[sz.stop - sz.start + 1, 
                     sy.stop - sy.start + 1, 
                     sx.stop - sx.start + 1] for sz, sy, sx in slices])
    n_iter = 6 * (ext.sum(axis=1)).max()

    # run diffusion
    shape = masks_padded.shape
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, shape,
                             n_iter=n_iter, device=device)
    # normalize
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((3, Lz0, Ly0, Lx0))
    mu0[:, z-1, y-1, x-1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c


def masks_to_flows_cpu(masks, device=None, niter=None):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 
    Parameters
    -------------
    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D array
        for each pixel, the distance to the center of the mask 
        in which it resides 
    """
    
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    
    slices = find_objects(masks)
    meds = []
    for i in prange(len(slices)):
        si = slices[i]
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 2, sc.stop - sc.start + 2
            ### get center-of-mass within cell
            y,x = np.nonzero(masks[sr, sc] == (i+1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = y.mean()
            xmed = x.mean()
            imin = ((x-xmed)**2 + (y-ymed)**2).argmin()
            xmed = x[imin]
            ymed = y[imin]
            
            n_iter = 2*np.int32(ly + lx) if niter is None else niter
            T = np.zeros((ly)*(lx), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(n_iter))
            dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
            dx = T[y*lx + x+1] - T[y*lx + x-1]
            mu[:, sr.start+y-1, sc.start+x-1] = np.stack((dy,dx))
            meds.append([ymed-1, xmed-1])
            
    # new normalization
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    return mu, meds


def masks_to_flows(masks, device=None, niter=None):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    if masks.max() == 0:
        dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), 'float32')

    if device is not None:
        if device.type=="cuda" or device.type=="mps":
            masks_to_flows_device = masks_to_flows_gpu
        else:
            masks_to_flows_device = masks_to_flows_cpu
    else:
        masks_to_flows_device = masks_to_flows_cpu
        
    if masks.ndim==3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device, niter=niter)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:,y], device=device, niter=niter)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:,:,x], device=device, niter=niter)[0]
            mu[[0,1], :, :, x] += mu0
        return mu
    elif masks.ndim==2:
        mu, mu_c = masks_to_flows_device(masks, device=device, niter=niter)
        return mu

    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')

def labels_to_flows(labels, files=None, device=None, redo_flows=False, niter=None):
    """ convert labels (list of masks or flows) to flows for training model 

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows and cell probabilities.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell distance transform, flows[k][2] is Y flow,
        flows[k][3] is X flow, and flows[k][4] is heat distribution

    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows: # flows need to be recomputed
        
        dynamics_logger.info('computing flows for labels')
        
        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        iterator = trange if nimg > 1 else range
        veci = [masks_to_flows(labels[n][0].astype(int), device=device, niter=niter) for n in iterator(nimg)]
        
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [np.concatenate((labels[n], labels[n]>0.5, veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imwrite(file_name+'_flows.tif', flow)
    else:
        dynamics_logger.info('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


@njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])', 
        '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C,Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly-1, max(0, yc_floor[i]))
        xf = min(Lx-1, max(0, xc_floor[i]))
        yf1= min(Ly-1, yf+1)
        xf1= min(Lx-1, xf+1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[c, yf, xf1]) * (1 - y) * x +
                      np.float32(I[c, yf1, xf]) * y * (1 - x) +
                      np.float32(I[c, yf1, xf1]) * y * x )


def steps2D_interp(p, dP, niter, device=None):
    shape = dP.shape[1:]
    if device is not None and device.type=="cuda":
        shape = np.array(shape)[[1,0]].astype('float')-1  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = torch.from_numpy(p[[1,0]].T).float().to(device).unsqueeze(0).unsqueeze(0) # p is n_points by 2, so pt is [1 1 2 n_points]
        im = torch.from_numpy(dP[[1,0]]).float().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2): 
            im[:,k,:,:] *= 2./shape[k]
            pt[:,:,:,k] /= shape[k]
            
        # normalize to between -1 and 1
        pt = pt*2-1 
        
        #here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            for k in range(2): #clamp the final pixel locations
                pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] + dPt[:,k,:,:], -1., 1.)
            
        #undo the normalization from before, reverse order of operations 
        pt = (pt+1)*0.5
        for k in range(2): 
            pt[:,:,:,k] *= shape[k]        
        
        p =  pt[:,:,:,[1,0]].cpu().numpy().squeeze().T
        return p

    else:
        dPt = np.zeros(p.shape, np.float32)
            
        for t in range(niter):
            map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
            for k in range(len(p)):
                p[k] = np.minimum(shape[k]-1, np.maximum(0, p[k] + dPt[k]))
        return p


@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 3D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 4D array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)

    dP: float32, 4D array
        flows [axis x Lz x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 3]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 4D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            z = inds[j,0]
            y = inds[j,1]
            x = inds[j,2]
            p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
            p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] + dP[0,p0,p1,p2]))
            p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] + dP[1,p0,p1,p2]))
            p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] + dP[2,p0,p1,p2]))
    return p

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 2D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)

    dP: float32, 3D array
        flows [axis x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            step = dP[:,p0,p1]
            for k in range(p.shape[0]):
                p[k,y,x] = min(shape[k]-1, max(0, p[k,y,x] + step[k]))
    return p

def follow_flows(dP, mask=None, niter=200, interp=True, device=None):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    
    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    inds: int32, 3D or 4D array
        indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    if len(shape)>2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                np.arange(shape[2]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP).max(axis=0)>1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)

        inds = np.array(np.nonzero(np.abs(dP).max(axis=0)>1e-3)).astype(np.int32).T
        
        if inds.ndim < 2 or inds.shape[0] < 5:
            dynamics_logger.warning('WARNING: no mask pixels found')
            return p, None
        
        if not interp:
            p = steps2D(p, dP.astype(np.float32), inds, niter)
        else:
            p_interp = steps2D_interp(p[:,inds[:,0], inds[:,1]], dP, niter, device=device)            
            p[:,inds[:,0],inds[:,1]] = p_interp
    return p, inds

def remove_bad_flow_masks(masks, flows, threshold=0.4, device=None):
    """ remove masks which have inconsistent flows 
    
    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    device0 = device
    if masks.size > 10000*10000 and (device is not None and device.type=="cuda"):
        
        major_version, minor_version, _ = torch.__version__.split(".")
        
        if major_version == "1" and int(minor_version) < 10:
            # for PyTorch version lower than 1.10
            def mem_info():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                used_mem = torch.cuda.memory_allocated()
                return total_mem, used_mem
        else:
            # for PyTorch version 1.10 and above
            def mem_info():
                total_mem, used_mem = torch.cuda.mem_get_info()
                return total_mem, used_mem
        
        if masks.size * 20 > mem_info()[0]:
            dynamics_logger.warning('WARNING: image is very large, not using gpu to compute flows from masks for QC step flow_threshold')
            dynamics_logger.info('turn off QC step with flow_threshold=0 if too slow')
            device0 = None

    merrors, _ = metrics.flow_error(masks, flows, device0)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, iscell=None, rpad=20):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 
    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded 
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s[:] = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    
    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0

def resize_and_compute_masks(dP, cellprob, p=None, niter=200,
                                cellprob_threshold=0.0,
                                flow_threshold=0.4, interp=True, do_3D=False,
                                min_size=15, resize=None,
                                device=None):
    """ compute masks using dynamics from dP, cellprob, and boundary """
    mask, p = compute_masks(dP, cellprob, p=p, niter=niter,
                            cellprob_threshold=cellprob_threshold,
                            flow_threshold=flow_threshold, interp=interp,
                            do_3D=do_3D, min_size=min_size,
                            device=device)

    if resize is not None:
        mask = transforms.resize_image(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST)
        p = np.array([transforms.resize_image(pi, resize[0], resize[1], interpolation=cv2.INTER_NEAREST) for pi in p])

    return mask, p


def compute_masks(dP, cellprob, p=None, niter=200, 
                   cellprob_threshold=0.0,
                   flow_threshold=0.4, interp=True, do_3D=False, 
                   min_size=15, device=None):
    """ compute masks using dynamics from dP, cellprob, and boundary """
    
    cp_mask = cellprob > cellprob_threshold 

    if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        if p is None:
            p, inds = follow_flows(dP * cp_mask / 5., niter=niter, interp=interp, 
                                            device=device)
            if inds is None:
                dynamics_logger.info('No cell pixels found.')
                shape = cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p
        
        #calculate masks
        mask = get_masks(p, iscell=cp_mask)
            
        # flow thresholding factored out of get_masks
        if not do_3D:
            if mask.max()>0 and flow_threshold is not None and flow_threshold > 0:
                # make sure labels are unique at output of get_masks
                mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold, device=device)
        
        if mask.max() > 2**16-1:
            recast = True
            mask = mask.astype(np.float32)
        else:
            recast = False
            mask = mask.astype(np.uint16)

        if recast:
            mask = mask.astype(np.uint32)

        if mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else: # nothing to compute, just make it compatible
        dynamics_logger.info('No cell pixels found.')
        shape = cellprob.shape
        mask = np.zeros(cellprob.shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p


    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    mask = utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    if mask.dtype==np.uint32:
        dynamics_logger.warning('more than 65535 masks in image, masks returned as np.uint32')

    return mask, p

