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
torch_GPU = torch.device("cuda")
torch_CPU = torch.device("cpu")


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
                        device=torch.device("cuda")):
    """Runs diffusion on GPU to generate flows for training images or quality control.

    Args:
        neighbors (torch.Tensor): 9 x pixels in masks.
        meds (torch.Tensor): Mask centers.
        isneighbor (torch.Tensor): Valid neighbor boolean 9 x pixels.
        shape (tuple): Shape of the tensor.
        n_iter (int, optional): Number of iterations. Defaults to 200.
        device (torch.device, optional): Device to run the computation on. Defaults to torch.device("cuda").

    Returns:
        torch.Tensor: Generated flows.

    """
    if device is None:
        device = torch.device("cuda")

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


def masks_to_flows_gpu(masks, device=None, niter=None):
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
        device = torch.device("cuda")

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2

    masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
    masks_padded = F.pad(masks_padded, (1, 1, 1, 1))

    ### get mask pixel neighbors
    y, x = torch.nonzero(masks_padded, as_tuple=True)
    neighborsY = torch.stack((y, y - 1, y + 1, y, y, y - 1, y - 1, y + 1, y + 1), dim=0)
    neighborsX = torch.stack((x, x, x, x - 1, x + 1, x - 1, x + 1, x - 1, x + 1), dim=0)
    neighbors = torch.stack((neighborsY, neighborsX), dim=0)
    neighbor_masks = masks_padded[tuple(neighbors)]
    isneighbor = neighbor_masks == neighbor_masks[0]

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
    shape = masks_padded.shape
    mu = _extend_centers_gpu(neighbors, meds_p, isneighbor, shape, n_iter=n_iter,
                             device=device)

    # new normalization
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)
    #mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu

    return mu0, meds_p.cpu().numpy() - 1


def masks_to_flows_gpu_3d(masks, device=None):
    """Convert masks to flows using diffusion from center pixel.

    Args:
        masks (int, 2D or 3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.

    Returns:
        tuple containing
            - mu (float, 3D or 4D array): Flows in Y = mu[-2], flows in X = mu[-1]. If masks are 3D, flows in Z = mu[0].
            - mu_c (float, 2D or 3D array): zeros
    """
    if device is None:
        device = torch.device("cuda")

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
    n_iter = 6 * (ext.sum(axis=1)).max()

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


def masks_to_flows(masks, device=None, niter=None):
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

    if device is not None:
        if device.type == "cuda" or device.type == "mps":
            masks_to_flows_device = masks_to_flows_gpu
        else:
            masks_to_flows_device = masks_to_flows_cpu
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


def steps2D_interp(p, dP, niter, device=None):
    """ Run dynamics of pixels to recover masks in 2D, with interpolation between pixel values.

    Euler integration of dynamics dP for niter steps.

    Args:
        p (numpy.ndarray): Array of shape (n_points, 2) representing the initial pixel locations.
        dP (numpy.ndarray): Array of shape (2, Ly, Lx) representing the flow field.
        niter (int): Number of iterations to perform.
        device (torch.device, optional): Device to use for computation. Defaults to None.

    Returns:
        numpy.ndarray: Array of shape (n_points, 2) representing the final pixel locations.

    Raises:
        None

    """

    shape = dP.shape[1:]
    if device is not None and device.type == "cuda":
        shape = np.array(shape)[[
            1, 0
        ]].astype("float") - 1  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = torch.from_numpy(p[[1, 0]].T).float().to(device).unsqueeze(0).unsqueeze(
            0)  # p is n_points by 2, so pt is [1 1 2 n_points]
        im = torch.from_numpy(dP[[1, 0]]).float().to(device).unsqueeze(
            0)  #covert flow numpy array to tensor on GPU, add dimension
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2):
            im[:, k, :, :] *= 2. / shape[k]
            pt[:, :, :, k] /= shape[k]

        # normalize to between -1 and 1
        pt = pt * 2 - 1

        #here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            for k in range(2):  #clamp the final pixel locations
                pt[:, :, :, k] = torch.clamp(pt[:, :, :, k] + dPt[:, k, :, :], -1., 1.)

        #undo the normalization from before, reverse order of operations
        pt = (pt + 1) * 0.5
        for k in range(2):
            pt[:, :, :, k] *= shape[k]

        p = pt[:, :, :, [1, 0]].cpu().numpy().squeeze().T
        return p

    else:
        dPt = np.zeros(p.shape, np.float32)

        for t in range(niter):
            map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
            for k in range(len(p)):
                p[k] = np.minimum(shape[k] - 1, np.maximum(0, p[k] + dPt[k]))
        return p


@njit("(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)", nogil=True)
def steps3D(p, dP, inds, niter):
    """ Run dynamics of pixels to recover masks in 3D.

    Euler integration of dynamics dP for niter steps.

    Args:
        p (np.ndarray): Pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid).
        dP (np.ndarray): Flows [axis x Lz x Ly x Lx].
        inds (np.ndarray): Non-zero pixels to run dynamics on [npixels x 3].
        niter (int): Number of iterations of dynamics to run.

    Returns:
        np.ndarray: Final locations of each pixel after dynamics.
    """
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            z = inds[j, 0]
            y = inds[j, 1]
            x = inds[j, 2]
            p0, p1, p2 = int(p[0, z, y, x]), int(p[1, z, y, x]), int(p[2, z, y, x])
            p[0, z, y, x] = min(shape[0] - 1, max(0, p[0, z, y, x] + dP[0, p0, p1, p2]))
            p[1, z, y, x] = min(shape[1] - 1, max(0, p[1, z, y, x] + dP[1, p0, p1, p2]))
            p[2, z, y, x] = min(shape[2] - 1, max(0, p[2, z, y, x] + dP[2, p0, p1, p2]))
    return p


@njit("(float32[:,:,:], float32[:,:,:], int32[:,:], int32)", nogil=True)
def steps2D(p, dP, inds, niter):
    """Run dynamics of pixels to recover masks in 2D.

    Euler integration of dynamics dP for niter steps.

    Args:
        p (np.ndarray): Pixel locations [axis x Ly x Lx] (start at initial meshgrid).
        dP (np.ndarray): Flows [axis x Ly x Lx].
        inds (np.ndarray): Non-zero pixels to run dynamics on [npixels x 2].
        niter (int): Number of iterations of dynamics to run.

    Returns:
        np.ndarray: Final locations of each pixel after dynamics.
    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j, 0]
            x = inds[j, 1]
            p0, p1 = int(p[0, y, x]), int(p[1, y, x])
            step = dP[:, p0, p1]
            for k in range(p.shape[0]):
                p[k, y, x] = min(shape[k] - 1, max(0, p[k, y, x] + step[k]))
    return p


def follow_flows(dP, mask=None, niter=200, interp=True, device=None):
    """ Run dynamics to recover masks in 2D or 3D.

    Pixels are represented as a meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds).

    Args:
        dP (np.ndarray): Flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
        mask (np.ndarray, optional): Pixel mask to seed masks. Useful when flows have low magnitudes.
        niter (int, optional): Number of iterations of dynamics to run. Default is 200.
        interp (bool, optional): Interpolate during 2D dynamics (not available in 3D). Default is True.
        use_gpu (bool, optional): Use GPU to run interpolated dynamics (faster than CPU). Default is False.

    Returns:
        tuple containing:
            - p (np.ndarray): Final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
            - inds (np.ndarray): Indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    if len(shape) > 2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                        indexing="ij")
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP).max(axis=0) > 1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        p = np.array(p).astype(np.float32)

        inds = np.array(np.nonzero(np.abs(dP).max(axis=0) > 1e-3)).astype(np.int32).T

        if inds.ndim < 2 or inds.shape[0] < 5:
            dynamics_logger.warning("WARNING: no mask pixels found")
            return p, None

        if not interp:
            p = steps2D(p, dP.astype(np.float32), inds, niter)
        else:
            p_interp = steps2D_interp(p[:, inds[:, 0], inds[:, 1]], dP, niter,
                                      device=device)
            p[:, inds[:, 0], inds[:, 1]] = p_interp
    return p, inds


def remove_bad_flow_masks(masks, flows, threshold=0.4, device=None):
    """Remove masks which have inconsistent flows.

    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from the network. Discards 
    masks with flow errors greater than the threshold.

    Args:
        masks (int, 2D or 3D array): Labelled masks, 0=NO masks; 1,2,...=mask labels,
            size [Ly x Lx] or [Lz x Ly x Lx].
        flows (float, 3D or 4D array): Flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
        threshold (float, optional): Masks with flow error greater than threshold are discarded.
            Default is 0.4.

    Returns:
        masks (int, 2D or 3D array): Masks with inconsistent flow masks removed,
            0=NO masks; 1,2,...=mask labels, size [Ly x Lx] or [Lz x Ly x Lx].
    """
    device0 = device
    if masks.size > 10000 * 10000 and (device is not None and device.type == "cuda"):

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
            dynamics_logger.warning(
                "WARNING: image is very large, not using gpu to compute flows from masks for QC step flow_threshold"
            )
            dynamics_logger.info("turn off QC step with flow_threshold=0 if too slow")
            device0 = None

    merrors, _ = metrics.flow_error(masks, flows, device0)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def get_masks(p, iscell=None, rpad=20):
    """Create masks using pixel convergence after running dynamics.

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
    big = np.prod(shape0) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  #convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0


def resize_and_compute_masks(dP, cellprob, p=None, niter=200, cellprob_threshold=0.0,
                             flow_threshold=0.4, interp=True, do_3D=False, min_size=15,
                             resize=None, device=None):
    """Compute masks using dynamics from dP and cellprob, and resizes masks if resize is not None.

    Args:
        dP (numpy.ndarray): The dynamics flow field array.
        cellprob (numpy.ndarray): The cell probability array.
        p (numpy.ndarray, optional): The pixels on which to run dynamics. Defaults to None
        niter (int, optional): The number of iterations for mask computation. Defaults to 200.
        cellprob_threshold (float, optional): The threshold for cell probability. Defaults to 0.0.
        flow_threshold (float, optional): The threshold for quality control metrics. Defaults to 0.4.
        interp (bool, optional): Whether to interpolate during dynamics computation. Defaults to True.
        do_3D (bool, optional): Whether to perform mask computation in 3D. Defaults to False.
        min_size (int, optional): The minimum size of the masks. Defaults to 15.
        resize (tuple, optional): The desired size for resizing the masks. Defaults to None.
        device (str, optional): The torch device to use for computation. Defaults to None.

    Returns:
        tuple: A tuple containing the computed masks and the final pixel locations.
    """
    mask, p = compute_masks(dP, cellprob, p=p, niter=niter,
                            cellprob_threshold=cellprob_threshold,
                            flow_threshold=flow_threshold, interp=interp, do_3D=do_3D,
                            min_size=min_size, device=device)

    if resize is not None:
        mask = transforms.resize_image(mask, resize[0], resize[1],
                                       interpolation=cv2.INTER_NEAREST)
        p = np.array([
            transforms.resize_image(pi, resize[0], resize[1],
                                    interpolation=cv2.INTER_NEAREST) for pi in p
        ])

    return mask, p


def compute_masks(dP, cellprob, p=None, niter=200, cellprob_threshold=0.0,
                  flow_threshold=0.4, interp=True, do_3D=False, min_size=15,
                  device=None):
    """Compute masks using dynamics from dP and cellprob.

    Args:
        dP (numpy.ndarray): The dynamics flow field array.
        cellprob (numpy.ndarray): The cell probability array.
        p (numpy.ndarray, optional): The pixels on which to run dynamics. Defaults to None
        niter (int, optional): The number of iterations for mask computation. Defaults to 200.
        cellprob_threshold (float, optional): The threshold for cell probability. Defaults to 0.0.
        flow_threshold (float, optional): The threshold for quality control metrics. Defaults to 0.4.
        interp (bool, optional): Whether to interpolate during dynamics computation. Defaults to True.
        do_3D (bool, optional): Whether to perform mask computation in 3D. Defaults to False.
        min_size (int, optional): The minimum size of the masks. Defaults to 15.
        device (str, optional): The torch device to use for computation. Defaults to None.

    Returns:
        tuple: A tuple containing the computed masks and the final pixel locations.
    """
    cp_mask = cellprob > cellprob_threshold

    if np.any(cp_mask):  #mask at this point is a cell cluster binary map, not labels
        # follow flows
        if p is None:
            p, inds = follow_flows(dP * cp_mask / 5., niter=niter, interp=interp,
                                   device=device)
            if inds is None:
                dynamics_logger.info("No cell pixels found.")
                shape = cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p

        #calculate masks
        mask = get_masks(p, iscell=cp_mask)

        # flow thresholding factored out of get_masks
        if not do_3D:
            if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
                # make sure labels are unique at output of get_masks
                mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold,
                                             device=device)

        if mask.max() > 2**16 - 1:
            recast = True
            mask = mask.astype(np.float32)
        else:
            recast = False
            mask = mask.astype(np.uint16)

        if recast:
            mask = mask.astype(np.uint32)

        if mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else:  # nothing to compute, just make it compatible
        dynamics_logger.info("No cell pixels found.")
        shape = cellprob.shape
        mask = np.zeros(cellprob.shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p

    mask = utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    if mask.dtype == np.uint32:
        dynamics_logger.warning(
            "more than 65535 masks in image, masks returned as np.uint32")

    return mask, p
