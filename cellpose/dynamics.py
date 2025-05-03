"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os
from scipy.ndimage import find_objects, center_of_mass, mean
import torch
import numpy as np
import tifffile
from tqdm import trange
import fastremap

import logging

dynamics_logger = logging.getLogger(__name__)

from . import utils

import torch
import torch.nn.functional as F

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
    if torch.prod(torch.tensor(shape)) > 4e7 or device.type == "mps":
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

def center_of_mass(mask):
    yi, xi = np.nonzero(mask)
    ymean = int(np.round(yi.sum() / len(yi)))
    xmean = int(np.round(xi.sum() / len(xi)))
    if not ((yi==ymean) * (xi==xmean)).sum():
        # center is closest point to (ymean, xmean) within mask
        imin = ((xi - xmean)**2 + (yi - ymean)**2).argmin()
        ymean = yi[imin]
        xmean = xi[imin]
    
    return ymean, xmean

def get_centers(masks, slices):
    centers = [center_of_mass(masks[slices[i]]==(i+1)) for i in range(len(slices))]
    centers = np.array([np.array([centers[i][0] + slices[i][0].start, centers[i][1] + slices[i][1].start]) 
                    for i in range(len(slices))])
    exts = np.array([(slc[0].stop - slc[0].start) + (slc[1].stop - slc[1].start) + 2 for slc in slices])
    return centers, exts


def masks_to_flows_gpu(masks, device=torch.device("cpu"), niter=None):
    """Convert masks to flows using diffusion from center pixel.

    Center of masks where diffusion starts is defined by pixel closest to median within the mask.

    Args:
        masks (int, 2D or 3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.
        device (torch.device, optional): The device to run the computation on. Defaults to torch.device("cpu").
        niter (int, optional): Number of iterations for the diffusion process. Defaults to None.

    Returns:
        np.ndarray: A 4D array representing the flows for each pixel in Z, X, and Y.
       

    Returns:
        A tuple containing (mu, meds_p). mu is float 3D or 4D array of flows in (Z)XY. 
        meds_p are cell centers.
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None

    if masks.max() > 0:
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
    else:
        # no masks, return empty flows
        mu0 = np.zeros((2, masks.shape[0], masks.shape[1]))
    return mu0

def masks_to_flows_gpu_3d(masks, device=None, niter=None):
    """Convert masks to flows using diffusion from center pixel.

    Args:
        masks (int, 2D or 3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.
        device (torch.device, optional): The device to run the computation on. Defaults to None.
        niter (int, optional): Number of iterations for the diffusion process. Defaults to None.

    Returns:
        np.ndarray: A 4D array representing the flows for each pixel in Z, X, and Y.
        
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
    return mu0

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
            vecn = masks_to_flows_gpu(labels[n][0].astype(int), device=device, niter=niter)

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


def flow_error(maski, dP_net, device=None):
    """Error in flows from predicted masks vs flows predicted by network run on image.

    This function serves to benchmark the quality of masks. It works as follows:
    1. The predicted masks are used to create a flow diagram.
    2. The mask-flows are compared to the flows that the network predicted.

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. This setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Args:
        maski (np.ndarray, int): Masks produced from running dynamics on dP_net, where 0=NO masks; 1,2... are mask labels.
        dP_net (np.ndarray, float): ND flows where dP_net.shape[1:] = maski.shape.

    Returns:
        A tuple containing (flow_errors, dP_masks): flow_errors (np.ndarray, float): Mean squared error between predicted flows and flows from masks; 
        dP_masks (np.ndarray, float): ND flows produced from the predicted masks.
    """
    if dP_net.shape[1:] != maski.shape:
        print("ERROR: net flow is not same size as predicted masks")
        return

    # flows predicted from estimated masks
    dP_masks = masks_to_flows_gpu(maski, device=device)
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i] / 5.)**2, maski,
                            index=np.arange(1,
                                            maski.max() + 1))

    return flow_errors, dP_masks


def steps_interp(dP, inds, niter, device=torch.device("cpu")):
    """ Run dynamics of pixels to recover masks in 2D/3D, with interpolation between pixel values.

    Euler integration of dynamics dP for niter steps.

    Args:
        p (numpy.ndarray): Array of shape (n_points, 2 or 3) representing the initial pixel locations.
        dP (numpy.ndarray): Array of shape (2, Ly, Lx) or (3, Lz, Ly, Lx) representing the flow field.
        niter (int): Number of iterations to perform.
        device (torch.device, optional): Device to use for computation. Defaults to None.

    Returns:
        numpy.ndarray: Array of shape (n_points, 2) or (n_points, 3) representing the final pixel locations.

    Raises:
        None

    """
    
    shape = dP.shape[1:]
    ndim = len(shape)
    
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
    shape = np.array(shape)[::-1].astype("float") - 1  
    
    # normalize pt between  0 and  1, normalize the flow
    for k in range(ndim):
        im[:, k] *= 2. / shape[k]
        pt[..., k] /= shape[k]

    # normalize to between -1 and 1
    pt *= 2 
    pt -= 1
    
    # dynamics
    for t in range(niter):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
        for k in range(ndim):  #clamp the final pixel locations
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1., 1.)

    #undo the normalization from before, reverse order of operations
    pt += 1 
    pt *= 0.5
    for k in range(ndim):
        pt[..., k] *= shape[k]

    if ndim==3:
        pt = pt[..., [2, 1, 0]].squeeze()
        pt = pt.unsqueeze(0) if pt.ndim==1 else pt 
        return pt.T
    else:
        pt = pt[..., [1, 0]].squeeze()
        pt = pt.unsqueeze(0) if pt.ndim==1 else pt
        return pt.T

def follow_flows(dP, inds, niter=200, device=torch.device("cpu")):
    """ Run dynamics to recover masks in 2D or 3D.

    Pixels are represented as a meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds).

    Args:
        dP (np.ndarray): Flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
        mask (np.ndarray, optional): Pixel mask to seed masks. Useful when flows have low magnitudes.
        niter (int, optional): Number of iterations of dynamics to run. Default is 200.
        interp (bool, optional): Interpolate during 2D dynamics (not available in 3D). Default is True.
        device (torch.device, optional): Device to use for computation. Default is None.

    Returns:
        A tuple containing (p, inds): p (np.ndarray): Final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]; 
        inds (np.ndarray): Indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    ndim = len(inds)
    
    p = steps_interp(dP, inds, niter, device=device)
        
    return p


def remove_bad_flow_masks(masks, flows, threshold=0.4, device=torch.device("cpu")):
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

        major_version, minor_version = torch.__version__.split(".")[:2]
        torch.cuda.empty_cache()
        if major_version == "1" and int(minor_version) < 10:
            # for PyTorch version lower than 1.10
            def mem_info():
                total_mem = torch.cuda.get_device_properties(device0.index).total_memory
                used_mem = torch.cuda.memory_allocated(device0.index)
                free_mem = total_mem - used_mem
                return total_mem, free_mem
        else:
            # for PyTorch version 1.10 and above
            def mem_info():
                free_mem, total_mem = torch.cuda.mem_get_info(device0.index)
                return total_mem, free_mem
        total_mem, free_mem = mem_info()
        if masks.size * 32 > free_mem:
            dynamics_logger.warning(
                "WARNING: image is very large, not using gpu to compute flows from masks for QC step flow_threshold"
            )
            dynamics_logger.info("turn off QC step with flow_threshold=0 if too slow")
            device0 = torch.device("cpu")

    merrors, _ = flow_error(masks, flows, device0)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def max_pool1d(h, kernel_size=5, axis=1, out=None):
    """ memory efficient max_pool thanks to Mark Kittisopikul 
    
    for stride=1, padding=kernel_size//2, requires odd kernel_size >= 3

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
    """ memory efficient max_pool in 2d or 3d """
    ndim = h.ndim - 1
    hmax = max_pool1d(h, kernel_size=kernel_size, axis=1)
    hmax2 = max_pool1d(hmax, kernel_size=kernel_size, axis=2)
    if ndim==2:
        del hmax
        return hmax2
    else:
        hmax = max_pool1d(hmax2, kernel_size=kernel_size, axis=3, out=hmax)
        del hmax2 
        return hmax

def get_masks_torch(pt, inds, shape0, rpad=20, max_size_fraction=0.4):
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
        max_size_fraction (float, optional): Masks larger than max_size_fraction of
            total image size are removed. Default is 0.4.

    Returns:
        M0 (int, 2D or 3D array): Masks with inconsistent flow masks removed, 
            0=NO masks; 1,2,...=mask labels, size [Ly x Lx] or [Lz x Ly x Lx].
    """
    
    ndim = len(shape0)
    device = pt.device
    
    rpad = 20
    pt += rpad
    pt = torch.clamp(pt, min=0)
    for i in range(len(pt)):
        pt[i] = torch.clamp(pt[i], max=shape0[i]+rpad-1)

    # # add extra padding to make divisible by 5
    # shape = tuple((np.ceil((shape0 + 2*rpad)/5) * 5).astype(int))
    shape = tuple(np.array(shape0) + 2*rpad)

    # sparse coo torch
    coo = torch.sparse_coo_tensor(pt, torch.ones(pt.shape[1], device=pt.device, dtype=torch.int), 
                                shape)
    h1 = coo.to_dense()
    del coo

    hmax1 = max_pool_nd(h1.unsqueeze(0), kernel_size=5)
    hmax1 = hmax1.squeeze()
    seeds1 = torch.nonzero((h1 - hmax1 > -1e-6) * (h1 > 10))
    del hmax1
    if len(seeds1) == 0:
        dynamics_logger.warning("no seeds found in get_masks_torch - no masks found.")
        return np.zeros(shape0, dtype="uint16")
    
    npts = h1[tuple(seeds1.T)]
    isort1 = npts.argsort()
    seeds1 = seeds1[isort1]

    n_seeds = len(seeds1)
    h_slc = torch.zeros((n_seeds, *[11]*ndim), device=seeds1.device)
    for k in range(n_seeds):
        slc = tuple([slice(seeds1[k][j]-5, seeds1[k][j]+6) for j in range(ndim)])
        h_slc[k] = h1[slc]
    del h1
    seed_masks = torch.zeros((n_seeds, *[11]*ndim), device=seeds1.device)
    if ndim==2:
        seed_masks[:,5,5] = 1
    else:
        seed_masks[:,5,5,5] = 1
    
    for iter in range(5):
        # extend
        seed_masks = max_pool_nd(seed_masks, kernel_size=3)
        seed_masks *= h_slc > 2
    del h_slc 
    seeds_new = [tuple((torch.nonzero(seed_masks[k]) + seeds1[k] - 5).T) 
            for k in range(n_seeds)]
    del seed_masks 
    
    dtype = torch.int32 if n_seeds < 2**16 else torch.int64
    M1 = torch.zeros(shape, dtype=dtype, device=device)
    for k in range(n_seeds):
        M1[seeds_new[k]] = 1 + k

    M1 = M1[tuple(pt)]
    M1 = M1.cpu().numpy()

    dtype = "uint16" if n_seeds < 2**16 else "uint32"
    M0 = np.zeros(shape0, dtype=dtype)
    M0[inds] = M1
        
    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * max_size_fraction
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  #convenient to guarantee non-skipped labels
    M0 = M0.reshape(tuple(shape0))
    
    #print(f"mem used: {torch.cuda.memory_allocated()/1e9:.3f} gb, max mem used: {torch.cuda.max_memory_allocated()/1e9:.3f} gb")
    return M0


def resize_and_compute_masks(dP, cellprob, niter=200, cellprob_threshold=0.0,
                             flow_threshold=0.4, do_3D=False, min_size=15,
                             max_size_fraction=0.4, resize=None, device=torch.device("cpu")):
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
        max_size_fraction (float, optional): Masks larger than max_size_fraction of
            total image size are removed. Default is 0.4.
        resize (tuple, optional): The desired size for resizing the masks. Defaults to None.
        device (torch.device, optional): The device to use for computation. Defaults to torch.device("cpu").

    Returns:
        tuple: A tuple containing the computed masks and the final pixel locations.
    """
    mask = compute_masks(dP, cellprob, niter=niter,
                            cellprob_threshold=cellprob_threshold,
                            flow_threshold=flow_threshold, do_3D=do_3D,
                            max_size_fraction=max_size_fraction, 
                            device=device)

    if resize is not None:
        dynamics_logger.warning("Resizing is depricated in v4.0.1+")

    mask = utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    return mask


def compute_masks(dP, cellprob, p=None, niter=200, cellprob_threshold=0.0,
                  flow_threshold=0.4, do_3D=False, min_size=-1,
                  max_size_fraction=0.4, device=torch.device("cpu")):
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
        max_size_fraction (float, optional): Masks larger than max_size_fraction of
            total image size are removed. Default is 0.4.
        device (torch.device, optional): The device to use for computation. Defaults to torch.device("cpu").

    Returns:
        tuple: A tuple containing the computed masks and the final pixel locations.
    """
    
    if (cellprob > cellprob_threshold).sum():  #mask at this point is a cell cluster binary map, not labels
        inds = np.nonzero(cellprob > cellprob_threshold)
        if len(inds[0]) == 0:
            dynamics_logger.info("No cell pixels found.")
            shape = cellprob.shape
            mask = np.zeros(shape, "uint16")
            return mask

        p_final = follow_flows(dP * (cellprob > cellprob_threshold) / 5., 
                               inds=inds, niter=niter, 
                                device=device)
        if not torch.is_tensor(p_final):
            p_final = torch.from_numpy(p_final).to(device, dtype=torch.int)
        else:
            p_final = p_final.int()
        # calculate masks
        if device.type == "mps":
            p_final = p_final.to(torch.device("cpu"))
        mask = get_masks_torch(p_final, inds, dP.shape[1:], 
                               max_size_fraction=max_size_fraction)
        del p_final
        # flow thresholding factored out of get_masks
        if not do_3D:
            if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
                # make sure labels are unique at output of get_masks
                mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold,
                                             device=device)

        if mask.max() < 2**16 and mask.dtype != "uint16":
            mask = mask.astype("uint16")

    else:  # nothing to compute, just make it compatible
        dynamics_logger.info("No cell pixels found.")
        shape = cellprob.shape
        mask = np.zeros(cellprob.shape, "uint16")
        return mask
    
    if min_size > 0:
        mask = utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    if mask.dtype == np.uint32:
        dynamics_logger.warning(
            "more than 65535 masks in image, masks returned as np.uint32")

    return mask
