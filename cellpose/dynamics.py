import time, os
from scipy.ndimage.filters import maximum_filter1d
import scipy.ndimage
import numpy as np
import tifffile
from tqdm import trange
from numba import njit, float32, int32, vectorize
import edt
import fastremap
from skimage.morphology import binary_erosion

import logging
dynamics_logger = logging.getLogger(__name__)
dynamics_logger.setLevel(logging.DEBUG)

from . import utils, metrics, transforms

try:
    import torch
    from torch import optim, nn
    from . import resnet_torch
    TORCH_ENABLED = True 
    torch_GPU = torch.device('cuda')
    torch_CPU = torch.device('cpu')
except:
    TORCH_ENABLED = False

@njit('(float64[:], int32[:], int32[:], int32[:], int32[:], int32, int32, boolean)', nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter, skel=True):
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
        if skel:
            T[ymed*Lx + xmed] += 1/9.**2
            T[ymed*Lx + xmed] /= (1+T[ymed*Lx + xmed])
        else:
            T[ymed*Lx + xmed] += 1

        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T


tic=time.time()

# edited slightly to fix a 'bleeding' issue with the gradient; now identical to CPU version
def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda'),skel=True):
    """ runs diffusion on GPU to generate flows for training images or quality control
    
    neighbors is 9 x pixels in masks, 
    centers are mask centers (or any distribution of heat, as in the generalized algorithm)
    isneighbor is valid neighbor boolean 9 x pixels
    
    """
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)
    
    T = torch.zeros((nimg,Ly,Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device)
    isneigh = torch.from_numpy(isneighbor).to(device)
    for t in range(n_iter):
        if skel: 
            T[:, meds[:,0], meds[:,1]] += 1/9.**2
            T[:, meds[:,0], meds[:,1]] /= (1+T[:, meds[:,0], meds[:,1]])
        else:
            T[:, meds[:,0], meds[:,1]] += 1
        Tneigh = T[:, pt[:,:,0], pt[:,:,1]] 
        Tneigh *= isneigh
        T[:, pt[0,:,0], pt[0,:,1]] = Tneigh.mean(axis=1)

    Tcpy = T.clone()
    idx = [2,1,4,3]
    mask = isneigh[idx]
    grads = T[:, pt[idx,:,0], pt[idx,:,1]]*mask # prevent bleedover
    dy = (grads[:,0] - grads[:,1])
    dx = (grads[:,2] - grads[:,3])
    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)

    return mu_torch, Tcpy.cpu().squeeze()


def masks_to_flows_gpu(masks, device=None, skel=True):
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
        
    pad = 1
    
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2*pad, Lx0+2*pad

    masks_padded = np.pad(masks,pad)
    
    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    
    neighborsY = np.stack((y, y-1, y+1, 
                           y, y, y-1, 
                           y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x-1, x+1, x-1, 
                           x+1, x-1, x+1), axis=0)
    
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)
    
    if not skel: # do original centroid projection algrorithm
        # get mask centers
        centers = np.array(scipy.ndimage.center_of_mass(masks_padded, labels=masks_padded, 
                                                        index=np.arange(1, masks_padded.max()+1))).astype(int)
        # (check mask center inside mask)
        valid = masks_padded[centers[:,0], centers[:,1]] == np.arange(1, masks_padded.max()+1)
        for i in np.nonzero(~valid)[0]:
            yi,xi = np.nonzero(masks_padded==(i+1))
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
            centers[i,0] = yi[imin]
            centers[i,1] = xi[imin] 
    else: # do 'skeletonized' algorithm (see cpu flow code for more details)
        centers = np.stack((y,x),axis=1)
    
    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]] #extract list of label values, 
    isneighbor = neighbor_masks == neighbor_masks[0] # 0 corresponds to x,y
    
    slices = scipy.ndimage.find_objects(masks)
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
#    n_iter = 10*np.int32(np.ptp(x)+np.ptp(y))
    # run diffusion
    mu, T = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, 
                             n_iter=n_iter, device=device)

    # normalize
    mag = (mu**2).sum(axis=0)**0.5
    mu = np.divide(mu, mag, out=np.zeros_like(mu), where=np.logical_and(mag!=0,~np.isnan(mag)))

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    
    mu_c = T[pad:-pad,pad:-pad] # mu_c now heat
    return mu0, mu_c

def masks_to_flows_cpu(masks, device=None, skel=True):
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
    
    # Get the dimensions of the mask, preallocate arrays to store flow values
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)
    
    
    nmask = masks.max() # Assumes a 'proper' label mask, values 1,2,... 
    slices = scipy.ndimage.find_objects(masks) # partitions into 'slices' like regionprops
    pad = 1
    #slice tuples contain the same info as boundingbox
    for i,si in enumerate(slices):
        if si is not None:
            
            sr,sc = si
            mask = np.pad((masks[sr, sc] == (i+1)),pad)
            # lx,ly the dimensions of the boundingbox
            ly, lx = sr.stop - sr.start + 2*pad, sc.stop - sc.start + 2*pad
            # x, y ordered list of componenets for the mask pixels
            y, x = np.nonzero(mask) 
            
            ly = np.int32(ly)
            lx = np.int32(lx)
            y = y.astype(np.int32)  #no need to shift, as array already padded
            x = x.astype(np.int32)    
            
            
            # number of iterations defined by range of points in the mask
            # _extend_centers handles actual diffusion
            niter = 10*np.int32(np.ptp(x)+np.ptp(y))

            # T is a vector of length (ly+2*pad)*(lx+2*pad), not a grid
            # should double-check to make sure that the padding isn't having unforeseen consequences 
            # same number of points as a grid with  1px around the whole thing
            T = np.zeros(ly*lx, np.float64)

            if (skel):
                # skeletonization now is far less explicit now (no skeletonization computation per se)
                # the skel flag effectively sets boundary conditions that produce a field extemely close
                # to that of an explicitly defined skeleton, but even better than those ad-hoc methods.
                # All pixels are used as heat sources.
                T = _extend_centers(T, y, x, y, x, lx, niter, skel)
            else:
                # original boundary projection
                ymed = np.mean(y)
                xmed = np.mean(x)
                imin = np.argmin((x-xmed)**2 + (y-ymed)**2) 
                xmed = np.array([x[imin]],np.int32)
                ymed = np.array([y[imin]],np.int32)
                T = _extend_centers(T, y, x, ymed, xmed, lx, niter, skel) # centroid model
            
            heat = T.copy()
            T  = np.interp(T, (T[y*lx + x].min(), T[y*lx + x].max()), (0, 1))
            # finite difference approximation
            dy = np.empty(ly*lx, np.float64) 
            dx = np.empty(ly*lx, np.float64)
  
            dy[y*lx + x] = T[(y+1)*lx + x] - T[(y-1)*lx + x]
            dx[y*lx + x] = T[y*lx + x+1] - T[y*lx + x-1]
 
            dy,dx = transforms.normalize_field(dy,dx)

            Dy = np.copy(dy[y*lx + x])
            Dx = np.copy(dx[y*lx + x])


            mu[:, sr.start+y-1, sc.start+x-1] = np.stack((Dy,Dx))
            mu_c[sr.start+y-1, sc.start+x-1] = heat[y*lx + x]
            
    # pass heat back instead of zeros - not sure what mu_c was originally
    # intended for, but it is apparently not used for anything else
    return mu, mu_c

def masks_to_flows(masks, use_gpu=False, device=None, skel=True):
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
    masks = masks.astype('int32')
    pad = 15
    masks = np.pad(masks,pad,mode='reflect')
    fastremap.renumber(masks,in_place=True); 
    if TORCH_ENABLED and use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
        masks_to_flows_device = masks_to_flows_gpu 
    else:
        masks_to_flows_device = masks_to_flows_cpu
        
    if masks.ndim==3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:,y], device=device)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:,:,x], device=device)[0]
            mu[[0,1], :, :, x] += mu0
        return mu, None
    elif masks.ndim==2:
        mu, T = masks_to_flows_device(masks, device=device, skel=skel)
#         print(mu.shape)
        return mu[:,pad:-pad,pad:-pad], T[pad:-pad,pad:-pad]
#         return mu, T
    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')

# Could prune this down to only save labels, distance, and heat,as all the rest are either replaced in
# the augmentation step or can quickly be computed beforehnad from these, no need to store them
def labels_to_flows(labels, files=None, use_gpu=False, device=None, skel=True):
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

    if labels[0].shape[0] == 1 or labels[0].ndim < 3:
        dynamics_logger.info('NOTE: computing flows for labels (could be done before to save time)')
        # compute flows        
#         veci = [masks_to_flows(labels[n][0],use_gpu=use_gpu, device=device, skel=skel)[0] for n in trange(nimg)]
        
        veci,heat = map(list,zip(*[masks_to_flows(labels[n][0],use_gpu=use_gpu, device=device, skel=skel) for n in trange(nimg)]))
        # compute distance transform
        dist = [edt.edt(labels[n][0],parallel=8)[np.newaxis,:,:] for n in trange(nimg)]
#         for n in trange(nimg):
#             dist[labels[n][0]>0] = -5 # background distance set to -5 in augmentations
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [np.concatenate((labels[n][[0]], dist[n][[0]], veci[n], heat[n][np.newaxis,:,:]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_flows.tif', flow)
    else:
        dynamics_logger.info('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


@njit(['(int16[:,:,:],float32[:], float32[:], float32[:,:])', 
        '(float32[:,:,:],float32[:], float32[:], float32[:,:])'], cache=True)
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


def steps2D_interp(p, dP, niter, use_gpu=False, device=None, skel=True, calc_trace=False):
    shape = dP.shape[1:]

    if use_gpu and TORCH_ENABLED:
        if device is None:
            device = torch_GPU
        shape = np.array(shape)[[1,0]].astype('double')-1  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = torch.from_numpy(p[[1,0]].T).double().to(device).unsqueeze(0).unsqueeze(0) # p is n_points by 2, so pt is [1 1 2 n_points]
        im = torch.from_numpy(dP[[1,0]]).double().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2): 
            im[:,k,:,:] *= 2./shape[k]
            pt[:,:,:,k] /= shape[k]
            
        # normalize to between -1 and 1
        pt = pt*2-1 
        
        # make an array to track the trajectories 
        if calc_trace:
            trace = torch.clone(pt).detach()
        
        #here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            if skel:
                dPt = dPt/(1+t) #this supression is key to the 'skeleton' method
            for k in range(2): #clamp the final pixel locations
                pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] + dPt[:,k,:,:], -1., 1.)
            
            if calc_trace:
                trace = torch.cat((trace,pt))
            
        #undo the normalization from before, reverse order of operations 
        pt = (pt+1)*0.5
        for k in range(2): 
            pt[:,:,:,k] *= shape[k]
        
        #pass back to cpu
        if calc_trace:
            tr =  trace[:,:,:,[1,0]].cpu().numpy().squeeze().T
        else:
            tr = None
        
        skelmask =  pt[:,:,:,[1,0]].cpu().numpy().squeeze().T
        return skelmask, tr
    else:
        dPt = np.zeros(p.shape, np.float32)
        if calc_trace:
            Ly = shape[0]
            Lx = shape[1]
            tr = np.zeros((niter,2,Ly,Lx))
        for t in range(niter):
            map_coordinates(dP, p[0], p[1], dPt)
            if skel:
                dPt = dPt/(1+t) #this supression is key to the 'skeleton' method
            for k in range(len(p)):
                p[k] = np.minimum(shape[k]-1, np.maximum(0, p[k] + dPt[k]))
                if calc_trace:
                    tr[t] = p
        return p, None 


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

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32, boolean, boolean)', nogil=True)
def steps2D(p, dP, inds, niter, skel=True, calc_trace=False):
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
    if calc_trace:
        Ly = shape[0]
        Lx = shape[1]
        tr = np.zeros((niter,2,Ly,Lx))
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            dy = dP[0,p0,p1]
            dx = dP[1,p0,p1]
            if skel: # suppress each step 
                dy, dx = dy/(t+1), dx/(t+1)
            for k in range(p.shape[0]):
                p[k,y,x] = min(shape[k]-1, max(0, p[k,y,x] + dy))
            if calc_trace:
                tr[t] = p
                
    return p, tr

def follow_flows(dP, mask=None, niter=200, interp=True, use_gpu=True, device=None, skel=True, calc_trace=False):
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

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.int32(niter)
    if len(shape)>2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                np.arange(shape[2]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        #inds = np.array(np.nonzero(dP[0]!=0)).astype(np.int32).T
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        if mask is None:
            mask = np.abs(dP[0])>1e-3
        inds = np.array(np.nonzero(np.logical_or(mask,np.abs(dP[0])>1e-3))).astype(np.int32).T
        
        if inds.ndim < 2 or inds.shape[0] < 5:
            dynamics_logger.warning('WARNING: no mask pixels found')
            return p, inds, None
        if not interp:
            print('warning: not interp')
            p, tr = steps2D(p, dP, inds, niter,skel=skel,calc_trace=calc_trace)
        else:
            p[:,inds[:,0],inds[:,1]], tr = steps2D_interp(p[:,inds[:,0], inds[:,1]], 
                                                      dP, niter, use_gpu=use_gpu,
                                                      device=device, skel=skel,
                                                      calc_trace=calc_trace)
    return p, inds, tr

def remove_bad_flow_masks(masks, flows, threshold=0.4, use_gpu=False, device=None):
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
    merrors, _ = metrics.flow_error(masks, flows, use_gpu, device)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, iscell=None, rpad=20, flows=None, threshold=0.4, use_gpu=False, device=None):
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
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

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
    
    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    _,counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    for i in np.nonzero(counts > big)[0]:
        M0[M0==i] = 0
    _,M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    if M0.max()>0 and threshold is not None and threshold > 0 and flows is not None:
        M0 = remove_bad_flow_masks(M0, flows, threshold=threshold, use_gpu=use_gpu, device=device)
        _,M0 = np.unique(M0, return_inverse=True)
        M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0
