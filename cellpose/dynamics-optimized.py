"""
Copyright © 2023 Howard Hughes Medical Institute, 
Authored by Carsen Stringer and Marius Pachitariu.
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

@njit("(float64[:], int32[:], int32[:], int32, int32, int32, int32)", nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """Run diffusion from the center of the mask on the mask pixels.

    Args:
        T (numpy.ndarray): Array of shape (Ly * Lx) where diffusion is run.
        y (numpy.ndarray): Array of y-coordinates of pixels inside the mask.
        x (numpy.ndarray): Array of x-coordinates of pixels inside the mask.
        ymed (int): Center of the mask in the y-coordinate.
        xmed (int): Center of the mask in the x-coordinate.
        Lx (int): Size of the x-dimension of the masks.
        niter (int): Number of iterations to run diffusion.

    Returns:
        numpy.ndarray: Array of shape (Ly * Lx) representing the amount of diffused particles at each pixel.
    """
    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx +
          x] = 1 / 9. * (T[y * Lx + x] + T[(y - 1) * Lx + x] + T[(y + 1) * Lx + x] +
                         T[y * Lx + x - 1] + T[y * Lx + x + 1] +
                         T[(y - 1) * Lx + x - 1] + T[(y - 1) * Lx + x + 1] +
                         T[(y + 1) * Lx + x - 1] + T[(y + 1) * Lx + x + 1])
    return T


def _extend_centers_gpu(neighbors, meds, isneighbor, shape, n_iter=200, 
                        device=torch.device("cpu")):
    """Runs diffusion on GPU to generate flows for training images or quality control.

    Args:
        neighbors (torch.Tensor): 9 x pixels in masks.
        meds (torch.Tensor): Mask centers.
        isneighbor (torch.Tensor): Valid neighbor boolean 9 x pixels.
        shape (tuple): Shape of the tensor.
        n_iter (int, optional): Number of iterations. Defaults to 200.
        device (torch.device, optional): Device to run the computation on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Generated flows.
    """
    if torch.prod(torch.tensor(shape)) > 4e7:
        T = torch.zeros(shape, dtype=torch.float, device=device)
    else:
        T = torch.zeros(shape, dtype=torch.double, device=device)

    for i in range(n_iter):
        T[tuple(meds.T)] += 1
        Tneigh = T[tuple(neighbors)]
        Tneigh *= isneighbor
        T[tuple(neighbors[:, 0])] = Tneigh.mean(axis=0)
    del meds, isneighbor, Tneigh

    if T.ndim == 2:
        grads = T[neighbors[0, [2, 1, 4, 3]], neighbors[1, [2, 1, 4, 3]]]
        del neighbors
        dy = grads[0] - grads[1]
        dx = grads[2] - grads[3]
        del grads
        mu_torch = np.stack((dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
    else:
        grads = T[tuple(neighbors[:, 1:])]
        del neighbors
        dz = grads[0] - grads[1]
        dy = grads[2] - grads[3]
        dx = grads[4] - grads[5]
        del grads
        mu_torch = np.stack(
            (dz.cpu().squeeze(0), dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
    return mu_torch

@njit(nogil=True)
def get_centers(masks, slices):
    """
    Get the centers of the masks and their extents.

    Args:
        masks (ndarray): The labeled masks.
        slices (ndarray): The slices of the masks.

    Returns:
        tuple containing
            - centers (ndarray): The centers of the masks.
            - ext (ndarray): The extents of the masks.
    """
    centers = np.zeros((len(slices), 2), "int32")
    ext = np.zeros((len(slices),), "int32")
    for p in prange(len(slices)):
        si = slices[p]
        i = si[0]
        sr, sc = si[1:3], si[3:5]
        # find center in slice around mask
        yi, xi = np.nonzero(masks[sr[0]:sr[-1], sc[0]:sc[-1]] == (i + 1))
        ymed = yi.mean()
        xmed = xi.mean()
        # center is closest point to (ymed, xmed) within mask
        imin = ((xi - xmed)**2 + (yi - ymed)**2).argmin()
        ymed = yi[imin] + sr[0]
        xmed = xi[imin] + sc[0]
        centers[p] = np.array([ymed, xmed])
        ext[p] = (sr[-1] - sr[0]) + (sc[-1] - sc[0]) + 2
    return centers, ext


def masks_to_flows_gpu(masks, device=torch.device("cpu"), niter=None):
    """Convert masks to flows using diffusion from center pixel.

    Center of masks where diffusion starts is defined using COM.

    Args:
        masks (int, 2D or 3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.

    Returns:
        tuple containing
            - mu (float, 3D or 4D array): Flows in Y = mu[-2], flows in X = mu[-1].
                If masks are 3D, flows in Z = mu[0].
            - meds_p (float, 2D or 3D array): cell centers
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2
    
    masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
    masks_padded = F.pad(masks_padded, (1, 1, 1, 1))
    shape = masks_padded.shape
    
    ### get mask pixel neighbors
    y, x = torch.nonzero(masks_padded, as_tuple=True)
    y = y.int()
    x = x.int()
    neighbors = torch.zeros((2, 9, y.shape[0]), dtype=torch.int, device=device)
    yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
    for i in range(9):
        neighbors[0, i] = y + yxi[0][i]
        neighbors[1, i] = x + yxi[1][i]
    isneighbor = torch.ones((9, y.shape[0]), dtype=torch.bool, device=device)
    m0 = masks_padded[neighbors[0, 0], neighbors[1, 0]]
    for i in range(1, 9):
        isneighbor[i] = masks_padded[neighbors[0, i], neighbors[1, i]] == m0
    del m0, masks_padded
    
    ### get center-of-mass within cell
    slices = find_objects(masks)
    # turn slices into array
    slices = np.array([
        np.array([i, si[0].start, si[0].stop, si[1].start, si[1].stop])
        for i, si in enumerate(slices)
        if si is not None
    ])
    centers, ext = get_centers(masks, slices)
    meds_p = torch.from_numpy(centers).to(device).long()
    meds_p += 1  # for padding
    
    ### run diffusion
    n_iter = 2 * ext.max() if niter is None else niter
    mu = _extend_centers_gpu(neighbors, meds_p, isneighbor, shape, n_iter=n_iter,
                             device=device)
    mu = mu.astype("float64")

    # new normalization
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu

    return mu0, meds_p.cpu().numpy() - 1


def masks_to_flows_gpu_3d(masks, device=None, niter=None):
    """Convert masks to flows using diffusion from center pixel.

    Args:
        masks (int, 2D or 3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.

    Returns:
        tuple containing
            - mu (float, 3D or 4D array): Flows in Y = mu[-2], flows in X = mu[-1]. If masks are 3D, flows in Z = mu[0].
            - mu_c (float, 2D or 3D array): zeros
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None

    Lz0, Ly0, Lx0 = masks.shape
    Lz, Ly, Lx = Lz0 + 2, Ly0 + 2, Lx0 + 2

    masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
    masks_padded = F.pad(masks_padded, (1, 1, 1, 1, 1, 1))

    # get mask pixel neighbors
    z, y, x = torch.nonzero(masks_padded).T
    neighborsZ = torch.stack((z, z + 1, z - 1, z, z, z, z))
    neighborsY = torch.stack((y, y, y, y + 1, y - 1, y, y), axis=0)
    neighborsX = torch.stack((x, x, x, x, x, x + 1, x - 1), axis=0)

    neighbors = torch.stack((neighborsZ, neighborsY, neighborsX), axis=0)

    # get mask centers
    slices = find_objects(masks)

    centers = np.zeros((masks.max(), 3), "int")
    for i, si in enumerate(slices):
        if si is not None:
            sz, sy, sx = si
            #lz, ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            zi, yi, xi = np.nonzero(masks[sz, sy, sx] == (i + 1))
            zi = zi.astype(np.int32) + 1  # add padding
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            zmed = np.mean(zi)
            ymed = np.mean(yi)
            xmed = np.mean(xi)
            imin = np.argmin((zi - zmed)**2 + (xi - xmed)**2 + (yi - ymed)**2)
            zmed = zi[imin]
            ymed = yi[imin]
            xmed = xi[imin]
            centers[i, 0] = zmed + sz.start
            centers[i, 1] = ymed + sy.start
            centers[i, 2] = xmed + sx.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[tuple(neighbors)]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array(
        [[sz.stop - sz.start + 1, sy.stop - sy.start + 1, sx.stop - sx.start + 1]
         for sz, sy, sx in slices])
    n_iter = 6 * (ext.sum(axis=1)).max() if niter is None else niter

    # run diffusion
    shape = masks_padded.shape
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, shape, n_iter=n_iter,
                             device=device)
    # normalize
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((3, Lz0, Ly0, Lx0))
    mu0[:, z.cpu().numpy() - 1, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c


def masks_to_flows_cpu(masks, device=None, niter=None):
    """Convert masks to flows using diffusion from center pixel.

    Center of masks where diffusion starts is defined to be the closest pixel to the mean of all pixels that is inside the mask.
    Result of diffusion is converted into flows by computing the gradients of the diffusion density map.

    Args:
        masks (int, 2D or 3D array): Labelled masks 0=NO masks; 1,2,...=mask labels

    Returns:
        tuple containing
            - mu (float, 3D or 4D array): Flows in Y = mu[-2], flows in X = mu[-1].
                If masks are 3D, flows in Z = mu[0].
            - meds (float, 2D or 3D array): cell centers
    """
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)

    slices = find_objects(masks)
    meds = []
    for i in prange(len(slices)):
        si = slices[i]
        if si is not None:
            sr, sc = si
            ly, lx = sr.stop - sr.start + 2, sc.stop - sc.start + 2
            ### get center-of-mass within cell
            y, x = np.nonzero(masks[sr, sc] == (i + 1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = y.mean()
            xmed = x.mean()
            imin = ((x - xmed)**2 + (y - ymed)**2).argmin()
            xmed = x[imin]
            ymed = y[imin]

            n_iter = 2 * np.int32(ly + lx) if niter is None else niter
            T = np.zeros((ly) * (lx), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(n_iter))
            dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
            dx = T[y * lx + x + 1] - T[y * lx + x - 1]
            mu[:, sr.start + y - 1, sc.start + x - 1] = np.stack((dy, dx))
            meds.append([ymed - 1, xmed - 1])

    # new normalization
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    return mu, meds


def masks_to_flows(masks, device=torch.device("cpu"), niter=None):
    """Convert masks to flows using diffusion from center pixel.

    Center of masks where diffusion starts is defined to be the closest pixel to the mean of all pixels that is inside the mask.
    Result of diffusion is converted into flows by computing the gradients of the diffusion density map.

    Args:
        masks (int, 2D or 3D array): Labelled masks 0=NO masks; 1,2,...=mask labels

    Returns:
        mu (float, 3D or 4D array): Flows in Y = mu[-2], flows in X = mu[-1].
                If masks are 3D, flows in Z = mu[0].
    """
    if masks.max() == 0:
        dynamics_logger.warning("empty masks!")
        return np.zeros((2, *masks.shape), "float32")

    if device.type == "cuda" or device.type == "mps":
        masks_to_flows_device = masks_to_flows_gpu
    else:
        masks_to_flows_device = masks_to_flows_cpu

    if masks.ndim == 3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device, niter=niter)[0]
            mu[[1, 2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:, y], device=device, niter=niter)[0]
            mu[[0, 2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:, :, x], device=device, niter=niter)[0]
            mu[[0, 1], :, :, x] += mu0
        return mu
    elif masks.ndim == 2:
        mu, mu_c = masks_to_flows_device(masks, device=device, niter=niter)
        return mu

    else:
        raise ValueError("masks_to_flows only takes 2D or 3D arrays")


def labels_to_flows(labels, files=None, device=None, redo_flows=False, niter=None,
                    return_flows=True):
    """Converts labels (list of masks or flows) to flows for training model.

    Args:
        labels (list of ND-arrays): The labels to convert. labels[k] can be 2D or 3D. If [3 x Ly x Lx], 
            it is assumed that flows were precomputed. Otherwise, labels[k][0] or labels[k] (if 2D) 
            is used to create flows and cell probabilities.
        files (list of str, optional): The files to save the flows to. If provided, flows are saved to 
            files to be reused. Defaults to None.
        device (str, optional): The device to use for computation. Defaults to None.
        redo_flows (bool, optional): Whether to recompute the flows. Defaults to False.
        niter (int, optional): The number of iterations for computing flows. Defaults to None.

    Returns:
        list of [4 x Ly x Lx] arrays: The flows for training the model. flows[k][0] is labels[k], 
        flows[k][1] is cell distance transform, flows[k][2] is Y flow, flows[k][3] is X flow, 
        and flows[k][4] is heat distribution.
    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]

    flows = []
    # flows need to be recomputed
    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows:
        dynamics_logger.info("computing flows for labels")

        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        iterator = trange if nimg > 1 else range
        for n in iterator(nimg):
            labels[n][0] = fastremap.renumber(labels[n][0], in_place=True)[0]
            vecn = masks_to_flows(labels[n][0].astype(int), device=device, niter=niter)

            # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
            flow = np.concatenate((labels[n], labels[n] > 0.5, vecn),
                                  axis=0).astype(np.float32)
            if files is not None:
                file_name = os.path.splitext(files[n])[0]
                tifffile.imwrite(file_name + "_flows.tif", flow)
            if return_flows:
                flows.append(flow)
    else:
        dynamics_logger.info("flows precomputed")
        if return_flows:
            flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


@njit([
    "(int16[:,:,:], float32[:], float32[:], float32[:,:])",
    "(float32[:,:,:], float32[:], float32[:], float32[:,:])"
], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    Bilinear interpolation of image "I" in-place with y-coordinates yc and x-coordinates xc to Y.
    
    Args:
        I (numpy.ndarray): Input image of shape (C, Ly, Lx).
        yc (numpy.ndarray): New y-coordinates.
        xc (numpy.ndarray): New x-coordinates.
        Y (numpy.ndarray): Output array of shape (C, ni).

    Returns:
        None
    """
    C, Ly, Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly - 1, max(0, yc_floor[i]))
        xf = min(Lx - 1, max(0, xc_floor[i]))
        yf1 = min(Ly - 1, yf + 1)
        xf1 = min(Lx - 1, xf + 1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c, i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                       np.float32(I[c, yf, xf1]) * (1 - y) * x +
                       np.float32(I[c, yf1, xf]) * y * (1 - x) +
                       np.float32(I[c, yf1, xf1]) * y * x)


def steps_interp(dP, inds, niter, device=torch.device("cpu")):
    """ Run dynamics of pixels to recover masks in 2D/3D, with interpolation between pixel values.
    
    Args:
        dP (numpy.ndarray): Array of shape (2, Ly, Lx) or (3, Lz, Ly, Lx) representing the flow field.
        inds (tuple): Indices of pixels to run dynamics on
        niter (int): Number of iterations to perform.
        device (torch.device, optional): Device to use for computation. Defaults to torch.device("cpu").
    
    Returns:
        torch.Tensor: Final pixel locations as a tensor of shape (ndim, n_points).
    """
    # Ensure niter is an integer
    niter = int(niter)
    
    shape = dP.shape[1:]
    ndim = len(shape)
    if (device.type == "cuda" or device.type == "mps") or ndim==3:
        pt = torch.zeros((*[1]*ndim, len(inds[0]), ndim), dtype=torch.float32, device=device)
        im = torch.zeros((1, ndim, *shape), dtype=torch.float32, device=device)
        # Y and X dimensions, flipped X-1, Y-1
        # pt is [1 1 1 3 n_points]
        for n in range(ndim):
            if ndim==3:
                pt[0, 0, 0, :, ndim - n - 1] = torch.from_numpy(inds[n]).to(device, dtype=torch.float32)
            else:
                pt[0, 0, :, ndim - n - 1] = torch.from_numpy(inds[n]).to(device, dtype=torch.float32)
            im[0, ndim - n - 1] = torch.from_numpy(dP[n]).to(device, dtype=torch.float32)
        shape_np = np.array(shape)[::-1].astype("float") - 1  
        
        # normalize pt between  0 and  1, normalize the flow
        for k in range(ndim):
            im[:, k] *= 2. / shape_np[k]
            pt[..., k] /= shape_np[k]

        # normalize to between -1 and 1
        pt *= 2 
        pt -= 1
        
        # dynamics
        for t in range(niter):
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            for k in range(ndim):  # clamp the final pixel locations
                pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1., 1.)

        # undo the normalization from before, reverse order of operations
        pt += 1 
        pt *= 0.5
        for k in range(ndim):
            pt[..., k] *= shape_np[k]

        if ndim == 3:
            # Replace .T with .permute to handle higher dimensions
            pt = pt[..., [2, 1, 0]].squeeze()
            if pt.dim() == 1:
                pt = pt.unsqueeze(1)
            else:
                pt = pt.permute(1, 0)
            return pt
        else:
            # Replace .T with .mT for 2D tensors
            pt = pt[..., [1, 0]].squeeze()
            if pt.dim() == 1:
                pt = pt.unsqueeze(1)
            else:
                pt = pt.mT
            return pt

    else:
        p = np.zeros((ndim, len(inds[0])), "float32")
        for n in range(ndim):
            p[n] = inds[n]        
        dPt = np.zeros(p.shape, "float32")
        for t in range(niter):
            map_coordinates(dP, p[0], p[1], dPt)
            for k in range(len(p)):
                p[k] = np.minimum(shape[k] - 1, np.maximum(0, p[k] + dPt[k]))
        return p


@njit("(float32[:,:],float32[:,:,:,:], int32)", nogil=True)
def steps3D(p, dP, niter):
    """ Run dynamics of pixels to recover masks in 3D.

    Euler integration of dynamics dP for niter steps.

    Args:
        p (np.ndarray): Pixels with cellprob > cellprob_threshold [3 x npts].
        dP (np.ndarray): Flows [3 x Lz x Ly x Lx].
        niter (int): Number of iterations of dynamics to run.

    Returns:
        np.ndarray: Final locations of each pixel after dynamics.
    """
    shape = dP.shape[1:]
    for t in range(niter):
        for j in range(p.shape[1]):
            p0, p1, p2 = int(p[0, j]), int(p[1, j]), int(p[2, j])
            step = dP[:, p0, p1, p2]
            for k in range(3):
                p[k, j] = min(shape[k] - 1, max(0, p[k, j] + step[k]))
    return p


@njit("(float32[:,:], float32[:,:,:], int32)", nogil=True)
def steps2D(p, dP, niter):
    """Run dynamics of pixels to recover masks in 2D.

    Euler integration of dynamics dP for niter steps.

    Args:
        p (np.ndarray): Pixels with cellprob > cellprob_threshold [2 x npts].
        dP (np.ndarray): Flows [2 x Ly x Lx].
        niter (int): Number of iterations of dynamics to run.

    Returns:
        np.ndarray: Final locations of each pixel after dynamics.
    """
    shape = dP.shape[1:]
    for t in range(niter):
        for j in range(p.shape[1]):
            # starting coordinates
            p0, p1 = int(p[0, j]), int(p[1, j])
            step = dP[:, p0, p1]
            for k in range(p.shape[0]):
                p[k, j] = min(shape[k] - 1, max(0, p[k, j] + step[k]))
    return p


@njit("(float32[:,:], float32[:,:,:], int32)", nogil=True, parallel=True)
def steps2D_vectorized(p, dP, niter):
    """Vectorized version of steps2D with early stopping"""
    shape = dP.shape[1:]
    for t in range(niter):
        # Get integer indices for all points at once
        p0 = p[0].astype(np.int32)
        p1 = p[1].astype(np.int32)
        
        # Clip indices to valid range
        p0 = np.clip(p0, 0, shape[0]-1)
        p1 = np.clip(p1, 0, shape[1]-1)
        
        # Track total movement
        total_movement = 0.0
        
        # Get steps for all points in parallel
        for j in prange(p.shape[1]):
            step0 = dP[0, p0[j], p1[j]]
            step1 = dP[1, p0[j], p1[j]]
            
            # Update positions
            p[0, j] = min(shape[0]-1, max(0, p[0, j] + step0))
            p[1, j] = min(shape[1]-1, max(0, p[1, j] + step1))
            
            total_movement += abs(step0) + abs(step1)
            
        # Early stopping if movement is small
        if total_movement / p.shape[1] < 0.1:
            break
    
    return p


def follow_flows(dP, inds, niter=200, interp=True, device=torch.device("cpu")):
    """Run dynamics of pixels to recover masks.
    
    Args:
        dP: flow field array
        inds: initial pixel coordinates
        niter: number of iterations (will be capped at 100)
        interp: whether to interpolate
        device: computation device
    Returns:
        p: final pixel locations
    """
    # Optimization: Cap iterations at 100 since convergence usually happens before then
    niter = min(niter, 100)
    dynamics_logger.debug(f"Running dynamics for {niter} iterations")
    
    # Original processing
    if not interp:
        shape = np.array(dP.shape[1:]).astype(np.int32)
        if len(shape) == 2:
            p = steps2D_vectorized(inds, dP, niter)
        else:
            p = steps3D(inds, dP, niter)
        return p
    else:
        return steps_interp(dP, inds, niter, device=device)


def remove_bad_flow_masks(masks, flows, threshold=0.4, device=torch.device("cpu")):
    """Remove masks which have inconsistent flows with caching."""
    # Cache for GPU memory check
    device0 = device
    if masks.size > 10000 * 10000 and device.type == "cuda":
        # Check available memory only once
        if not hasattr(remove_bad_flow_masks, '_mem_checked'):
            total_mem, free_mem = torch.cuda.mem_get_info(device0.index)
            remove_bad_flow_masks._use_cpu = masks.size * 32 > free_mem
            remove_bad_flow_masks._mem_checked = True
            
        if getattr(remove_bad_flow_masks, '_use_cpu', False):
            device0 = torch.device("cpu")
    
    merrors, _ = metrics.flow_error(masks, flows, device0)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def max_pool3d(h, kernel_size=5):
    """ memory efficient max_pool thanks to Mark Kittisopikul 
    
    for stride=1, padding=kernel_size//2, requires odd kernel_size >= 3
    
    """
    _, nd, ny, nx = h.shape
    m = h.clone().detach()
    kruns, k0 = kernel_size // 2, 1
    for k in range(kruns):
        for d in range(-k0, k0+1):
            for y in range(-k0, k0+1):
                for x in range(-k0, k0+1):
                    mv = m[:, max(-d,0):min(nd-d,nd), max(-y,0):min(ny-y,ny), max(-x,0):min(nx-x,nx)]
                    hv = h[:,  max(d,0):min(nd+d,nd),  max(y,0):min(ny+y,ny),  max(x,0):min(nx+x,nx)]
                    torch.maximum(mv, hv, out=mv)
    return m

def max_pool2d(h, kernel_size=5):
    """ memory efficient max_pool thanks to Mark Kittisopikul """
    _, ny, nx = h.shape
    m = h.clone().detach()
    k0 = kernel_size // 2
    for y in range(-k0, k0+1):
        for x in range(-k0, k0+1):
            mv = m[:, max(-y,0):min(ny-y,ny), max(-x,0):min(nx-x,nx)]
            hv = h[:, max(y,0):min(ny+y,ny),  max(x,0):min(nx+x,nx)]
            torch.maximum(mv, hv, out=mv)
    return m

def max_pool1d(h, kernel_size=5, axis=1, out=None):
    """ Memory efficient max_pool thanks to Mark Kittisopikul 
    
    For stride=1, padding=kernel_size//2, requires odd kernel_size >= 3
    Uses in-place operations to minimize memory allocations.

    Args:
        h: input tensor
        kernel_size: size of pooling window
        axis: axis to pool over
        out: optional output tensor for in-place operation
    """
    if out is None:
        out = h.clone()
    else:
        out.copy_(h)

    nd = h.shape[axis]    
    k0 = kernel_size // 2
    for d in range(-k0, k0+1):
        if axis==1:
            mv = out[:, max(-d,0):min(nd-d,nd)]
            hv = h[:, max(d,0):min(nd+d,nd)]
        elif axis==2:
            mv = out[:, :, max(-d,0):min(nd-d,nd)]
            hv = h[:,  :, max(d,0):min(nd+d,nd)]
        elif axis==3:
            mv = out[:, :, :, max(-d,0):min(nd-d,nd)]
            hv = h[:, :,  :, max(d,0):min(nd+d,nd)]
        torch.maximum(mv, hv, out=mv)
    return out

def max_pool_nd(h, kernel_size=5):
    """Memory efficient max pooling using PyTorch's built-in functions"""
    if len(h.shape) == 3:  # 2D case [batch, height, width]
        return F.max_pool2d(h, kernel_size, stride=1, padding=kernel_size//2)
    else:  # 3D case [batch, depth, height, width]
        return F.max_pool3d(h, kernel_size, stride=1, padding=kernel_size//2)

# from torch.nn.functional import max_pool2d
def get_masks_torch(pt, inds, shape0, rpad=20, max_size_fraction=0.4):
    """Create masks using pixel convergence after running dynamics.
    
    Args:
        pt: final pixel locations
        inds: initial pixel coordinates
        shape0: original shape of image
        rpad: padding for histogram
        max_size_fraction: max size of mask as fraction of image
    Returns:
        masks: integer masks
    """
    ndim = len(shape0)
    device = pt.device

    try:
        # Add padding
        pt = pt.clone()  # Create a copy to avoid modifying the input
        pt += rpad
        pt = torch.clamp(pt, min=0)
        for i in range(len(pt)):
            pt[i] = torch.clamp(pt[i], max=shape0[i]+rpad-1)

        shape = tuple(np.array(shape0) + 2*rpad)

        # Create sparse tensor efficiently
        coo = torch.sparse_coo_tensor(pt, torch.ones(pt.shape[1], device=pt.device), shape)
        h1 = coo.to_dense()
        del coo  # Free memory early

        # Find local maxima
        h1_maxpool = max_pool_nd(h1.unsqueeze(0), kernel_size=5)
        h1_maxpool = h1_maxpool.squeeze()
        seeds = torch.nonzero((h1 - h1_maxpool > -1e-6) & (h1 > 10))
        del h1_maxpool

        if seeds.shape[0] == 0:
            return np.zeros(shape0, dtype="uint16")

        # Sort seeds by density
        npts = h1[tuple(seeds.T)]
        isort = torch.argsort(npts, descending=True)
        seeds = seeds[isort]
        del npts, isort

        # Initialize masks
        M = torch.zeros(shape, dtype=torch.int32, device=device)
        for i, seed in enumerate(seeds):
            M[tuple(seed)] = i + 1

        # Extend masks
        for _ in range(5):
            M_maxpool = max_pool_nd(M.unsqueeze(0).float(), kernel_size=3)
            M = (M_maxpool.squeeze(0) * (h1 > 2)).long()
            del M_maxpool

        # Get mask for original points
        M0 = M[tuple(pt.long())]
        M0 = M0.cpu().numpy()

        # Create output mask
        final_mask = np.zeros(shape0, dtype="uint16")
        final_mask[inds] = M0

        # Remove big masks
        uniq, counts = fastremap.unique(final_mask, return_counts=True)
        big = np.prod(shape0) * max_size_fraction
        bigc = uniq[counts > big]
        if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
            final_mask = fastremap.mask(final_mask, bigc)
        
        # Renumber masks
        fastremap.renumber(final_mask, in_place=True)

        return final_mask

    except Exception as e:
        dynamics_logger.error(f"Error in get_masks_torch: {str(e)}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return np.zeros(shape0, dtype="uint16")


def resize_and_compute_masks(dP, cellprob, niter=200, cellprob_threshold=0.0,
                             flow_threshold=0.4, interp=True, do_3D=False, min_size=15,
                             max_size_fraction=0.4, resize=None, device=torch.device("cpu")):
    """Compute masks using dynamics from dP and cellprob, and resizes masks if resize is not None.

    Args:
        dP (numpy.ndarray): The dynamics flow field array.
        cellprob (numpy.ndarray): The cell probability array.
        niter (int, optional): The number of iterations for mask computation. Defaults to 200.
        cellprob_threshold (float, optional): The threshold for cell probability. Defaults to 0.0.
        flow_threshold (float, optional): The threshold for quality control metrics. Defaults to 0.4.
        interp (bool, optional): Whether to interpolate during dynamics computation. Defaults to True.
        do_3D (bool, optional): Whether to perform mask computation in 3D. Defaults to False.
        min_size (int, optional): The minimum size of the masks. Defaults to 15.
        max_size_fraction (float, optional): Masks larger than max_size_fraction of
            total image size are removed. Default is 0.4.
        resize (tuple, optional): The desired size for resizing the masks. Defaults to None.
        device (torch.device, optional): The device to use for computation. Defaults to torch.device("cpu").

    Returns:
        tuple: A tuple containing the computed masks and the final pixel locations.
    """
    mask = compute_masks(dP, cellprob, niter=niter,
                            cellprob_threshold=cellprob_threshold,
                            flow_threshold=flow_threshold, interp=interp, do_3D=do_3D,
                            max_size_fraction=max_size_fraction, 
                            device=device)

    if resize is not None:
        if len(resize) == 2:
            mask = transforms.resize_image(mask, resize[0], resize[1], no_channels=True,
                                           interpolation=cv2.INTER_NEAREST)
        else:
            Lz, Ly, Lx = resize
            if mask.shape[0] != Lz or mask.shape[1] != Ly:
                dynamics_logger.info("resizing 3D masks to original image size")
                if mask.shape[1] != Ly:
                    mask = transforms.resize_image(mask, Ly=Ly, Lx=Lx,
                                                no_channels=True, 
                                                interpolation=cv2.INTER_NEAREST)
                if mask.shape[0] != Lz:
                    mask = transforms.resize_image(mask.transpose(1,0,2),
                                                    Ly=Lz, Lx=Lx,
                                                    no_channels=True, 
                                                    interpolation=cv2.INTER_NEAREST).transpose(1,0,2)

    mask = utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    return mask


def compute_masks(dP, cellprob, p=None, niter=200, cellprob_threshold=0.0,
                  flow_threshold=0.4, interp=True, do_3D=False, min_size=-1,
                  max_size_fraction=0.4, device=torch.device("cpu")):
    """Compute masks using dynamics from dP and cellprob."""
    
    try:
        # Quick check for empty masks
        cellprob_mask = cellprob > cellprob_threshold
        if not cellprob_mask.any():
            dynamics_logger.info("No cells found above threshold")
            return np.zeros(cellprob.shape, dtype="uint16")

        # Get indices efficiently
        inds = np.nonzero(cellprob_mask)
        if len(inds[0]) == 0:
            dynamics_logger.info("No valid indices found")
            return np.zeros(cellprob.shape, dtype="uint16")
        
        # Pre-compute scaled dP to avoid repeated computation
        dP_scaled = dP * cellprob_mask / 5.
        p_final = follow_flows(dP_scaled, inds=inds, niter=niter, 
                             interp=interp, device=device)
        del dP_scaled  # Free memory early

        if p_final is None:
            dynamics_logger.warning("Flow following failed")
            return np.zeros(cellprob.shape, dtype="uint16")

        # Convert to torch tensor if needed
        if not torch.is_tensor(p_final):
            p_final = torch.from_numpy(p_final).to(device)

        # Calculate masks
        mask = get_masks_torch(p_final, inds, cellprob.shape, 
                             max_size_fraction=max_size_fraction)
        
        if mask is None:
            dynamics_logger.warning("Mask computation failed")
            return np.zeros(cellprob.shape, dtype="uint16")

        # Flow thresholding
        if not do_3D:
            if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
                mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold,
                                          device=device)

        return mask

    except Exception as e:
        dynamics_logger.error(f"Error in compute_masks: {str(e)}")
        return np.zeros(cellprob.shape, dtype="uint16")


def get_masks_orig(p, iscell=None, rpad=20, max_size_fraction=0.4):
    """Create masks using pixel convergence after running dynamics.

    Original implementation on CPU with histogramdd
    (histogramdd uses excessive memory with large images)

    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 

    Parameters:
        p (float32, 3D or 4D array): Final locations of each pixel after dynamics,
            size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
        iscell (bool, 2D or 3D array): If iscell is not None, set pixels that are 
            iscell False to stay in their original location.
        rpad (int, optional): Histogram edge padding. Default is 20.
        max_size_fraction (float, optional): Masks larger than max_size_fraction of
            total image size are removed. Default is 0.4.

    Returns:
        M0 (int, 2D or 3D array): Masks with inconsistent flow masks removed, 
            0=NO masks; 1,2,...=mask labels, size [Ly x Lx] or [Lz x Ly x Lx].
    """
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims == 3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               np.arange(shape0[2]), indexing="ij")
        elif dims == 2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               indexing="ij")
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s[:] = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims == 3:
        expand = np.nonzero(np.ones((3, 3, 3)))
    else:
        expand = np.nonzero(np.ones((3, 3)))

    for iter in range(5):
        for k in range(len(pix)):
            if iter == 0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * max_size_fraction
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  # convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0
