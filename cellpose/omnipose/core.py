import numpy as np
from numba import njit
import cv2
import edt
from scipy.ndimage import binary_dilation, binary_opening, label

from . import utils

try:
    import torch
    from torch import optim, nn
    from . import resnet_torch
    TORCH_ENABLED = True 
    torch_GPU = torch.device('cuda')
    torch_CPU = torch.device('cpu')
except:
    TORCH_ENABLED = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_ENABLED = True 
except:
    SKLEARN_ENABLED = False

try:
    from skimage.util import random_noise
    from skimage.filters import gaussian
    from skimage import measure
    SKIMAGE_ENABLED = True 
except:
    from scipy.ndimage import gaussian_filter as gaussian
    SKIMAGE_ENABLED = False
    
import logging
omnipose_logger = logging.getLogger(__name__)
omnipose_logger.setLevel(logging.DEBUG)

#utilities

# By testing for convergence across a range of superellipses, I found that the following
# ratio guarantees convergence. The edt() package gives a quick (but rough) distance field,
# and it allows us to find a least upper bound for the number of iterations needed for our
# smooth distance field computation. 
def get_niter(dists):
    return np.ceil(np.max(dists)*1.16).astype(int)+1

def dist_to_diam(dt_pos):
    return 6*np.mean(dt_pos)
#     return np.exp(3/2)*gmean(dt_pos[dt_pos>=gmean(dt_pos)])

def diameters(masks,dist_threshold=0):
    dt = edt.edt(np.int32(masks))
    dt_pos = np.abs(dt[dt>dist_threshold])
    return dist_to_diam(np.abs(dt_pos))
    
# Omnipose distance field is built on the following modified FIM update. 

@njit('(float64[:], int32[:], int32[:], int32)', nogil=True)
def eikonal_update_cpu(T, y, x, Lx):
    """Update for iterative solution of the eikonal equation on CPU."""
    minx = np.minimum(T[y*Lx + x-1],T[y*Lx + x+1])
    miny = np.minimum(T[(y-1)*Lx + x],T[(y+1)*Lx + x],)
    mina = np.minimum(T[(y-1)*Lx + x-1],T[(y+1)*Lx + x+1])
    minb = np.minimum(T[(y-1)*Lx + x+1],T[(y+1)*Lx + x-1])
    
    A = np.where(np.abs(mina-minb) >= 2, np.minimum(mina,minb)+np.sqrt(2), (1./2)*(mina+minb+np.sqrt(4-(mina-minb)**2)))
    B = np.where(np.abs(miny-minx) >= np.sqrt(2), np.minimum(miny,minx)+1, (1./2)*(miny+minx+np.sqrt(2-(miny-minx)**2)))
    
    return np.sqrt(A*B)

def eikonal_update_gpu(T,pt,isneigh):
    """Update for iterative solution of the eikonal equation on GPU."""
    
    # zero out the non-neighbor elements so that they do not participate in min
    Tneigh = T[:, pt[:,:,0], pt[:,:,1]] 
    Tneigh *= isneigh

    # using flattened index for the lattice points, just like gradient below
    minx = torch.minimum(Tneigh[:,3,:],Tneigh[:,5,:]) 
    mina = torch.minimum(Tneigh[:,2,:],Tneigh[:,6,:])
    miny = torch.minimum(Tneigh[:,1,:],Tneigh[:,7,:])
    minb = torch.minimum(Tneigh[:,0,:],Tneigh[:,8,:])

    A = torch.where(torch.abs(mina-minb) >= 2, torch.minimum(mina,minb) + np.sqrt(2), (1./2)*(mina+minb+torch.sqrt(4-(mina-minb)**2)))
    B = torch.where(torch.abs(miny-minx) >= np.sqrt(2), torch.minimum(miny,minx) + 1, (1./2)*(miny+minx+torch.sqrt(2-(miny-minx)**2)))

    return torch.sqrt(A*B)

def smooth_distance(masks, dists=None, device=None):
    if device is None:
        device = torch.device('cuda')
    if dists is None:
        dists = edt.edt(masks)
        
    pad = 1
    
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2*pad, Lx0+2*pad

    masks_padded = np.pad(masks,pad)
    
    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)

    neighborsY = np.stack((y-1, y-1, y-1, 
                           y  , y  , y  ,
                           y+1, y+1, y+1), axis=0)
    neighborsX = np.stack((x-1, x  , x+1, 
                           x-1, x  , x+1, 
                           x-1, x  , x+1), axis=0)
    
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)
    
    centers = np.stack((y,x),axis=1)
    
    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]] #extract list of label values, 
    isneighbor = neighbor_masks == neighbor_masks[4] # 4 corresponds to x,y now
        
    # set number of iterations
    n_iter = get_niter(dists)
        
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)
    T = torch.zeros((nimg,Ly,Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device)
    isneigh = torch.from_numpy(isneighbor).to(device)

    for t in range(n_iter):
        T[:, pt[4,:,0], pt[4,:,1]] = eikonal_update_gpu(T,pt,isneigh)

    return T.cpu().squeeze().numpy()[pad:-pad,pad:-pad]

# Omnipose requires (a) a special suppressed Euler step and (b) a special mask reconstruction algorithm. 

# no reason to use njit here except for compatibility with jitted fuctions that call it 
#this way, the same factor is used everywhere (CPU+-interp, GPU)
@njit()
def step_factor(t):
    """ Euler integration suppression factor."""
    return (1+t)

def div_rescale(dP,mask):
    dP = dP.copy()
    dP *= mask 
    dP = utils.normalize_field(dP)

    # compute the divergence
    Y, X = np.nonzero(mask)
    Ly,Lx = mask.shape
    pad = 1
    Tx = np.zeros((Ly+2*pad)*(Lx+2*pad), np.float64)
    Tx[Y*Lx+X] = np.reshape(dP[1].copy(),Ly*Lx)[Y*Lx+X]
    Ty = np.zeros((Ly+2*pad)*(Lx+2*pad), np.float64)
    Ty[Y*Lx+X] = np.reshape(dP[0].copy(),Ly*Lx)[Y*Lx+X]

    # Rescaling by the divergence
    div = np.zeros(Ly*Lx, np.float64)
    div[Y*Lx+X]=(Ty[(Y+2)*Lx+X]+8*Ty[(Y+1)*Lx+X]-8*Ty[(Y-1)*Lx+X]-Ty[(Y-2)*Lx+X]+
                 Tx[Y*Lx+X+2]+8*Tx[Y*Lx+X+1]-8*Tx[Y*Lx+X-1]-Tx[Y*Lx+X-2])
    div = utils.normalize99(div)
    div.shape = (Ly,Lx)
    #add sigmoid on boundary output to help push pixels away - the final bit needed in some cases!
    # specifically, places where adjacent cell flows are too colinear and therefore had low divergence
#                 mag = div+1/(1+np.exp(-bd))
    dP *= div
    return dP

def get_masks(p,bd,dist,mask,inds,nclasses=4,cluster=False,diam_threshold=12.,verbose=False):
    """Omnipose mask recontruction algorithm."""
    if nclasses == 4:
        dt = np.abs(dist[mask]) #abs needed if the threshold is negative
        d = dist_to_diam(dt)
        eps = 1+1/3

    else: #backwards compatibility, doesn't help for *clusters* of thin/small cells
        d = diameters(mask)
        eps = np.sqrt(2)

    # The mean diameter can inform whether or not the cells are too small to form contiguous blobs.
    # My first solution was to upscale everything before Euler integration to give pixels 'room' to
    # stay together. My new solution is much better: use a clustering algorithm on the sub-pixel coordinates
    # to assign labels. It works just as well and is faster because it doesn't require increasing the 
    # number of points or taking time to upscale/downscale the data. Users can toggle cluster on manually or
    # by setting the diameter threshold higher than the average diameter of the cells. 
    if verbose:
        omnipose_logger.info('Mean diameter is %f'%d)

    if d <= diam_threshold:
        cluster = True
        if verbose:
            omnipose_logger.info('Turning on subpixel clustering for label continuity.')
    y,x = np.nonzero(mask)
    newinds = p[:,inds[:,0],inds[:,1]].swapaxes(0,1)
    mask = np.zeros((p.shape[1],p.shape[2]))
    
    # the eps parameter needs to be adjustable... maybe a function of the distance
    if cluster and SKLEARN_ENABLED:
        if verbose:
            omnipose_logger.info('Doing DBSCAN clustering with eps=%f'%eps)
        db = DBSCAN(eps=eps, min_samples=3,n_jobs=8).fit(newinds)
        labels = db.labels_
        mask[inds[:,0],inds[:,1]] = labels+1
    else:
        newinds = np.rint(newinds).astype(int)
        skelmask = np.zeros_like(dist, dtype=bool)
        skelmask[newinds[:,0],newinds[:,1]] = 1

        #disconnect skeletons at the edge, 5 pixels in 
        border_mask = np.zeros(skelmask.shape, dtype=bool)
        border_px =  border_mask.copy()
        border_mask = binary_dilation(border_mask, border_value=1, iterations=5)

        border_px[border_mask] = skelmask[border_mask]
        if nclasses == 4: #can use boundary to erase joined edge skelmasks 
            border_px[bd>-1] = 0
            if verbose:
                omnipose_logger.info('Using boundary output to split edge defects')
        else: #otherwise do morphological opening to attempt splitting 
            border_px = binary_opening(border_px,border_value=0,iterations=3)

        skelmask[border_mask] = border_px[border_mask]

        if SKIMAGE_ENABLED:
            LL = measure.label(skelmask,connectivity=1) 
        else:
            LL = label(skelmask)[0]
        mask[inds[:,0],inds[:,1]] = LL[newinds[:,0],newinds[:,1]]
    
    return mask



# Omnipose has special training settings. Loss function and augmentation. 

def random_rotate_and_resize(X, Y=None, scale_range=1., gamma_range=0.5, xy = (224,224), 
                             do_flip=True, rescale=None, inds=None):
    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)

        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]

        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3, then the labels are assumed to be [cell probability, Y flow, X flow]. 

        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
            
        gamma_range: float (optional, default 0.5)
           Images are gamma-adjusted im**gamma for gamma in (1-gamma_range,1+gamma_range) 

        xy: tuple, int (optional, default (224,224))
            size of transformed images to return

        do_flip: bool (optional, default True)
            whether or not to flip images horizontally

        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations

        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]

        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]

        scale: array, float
            amount each image was resized by

    """
    dist_bg = 5 # background distance field is set to -dist_bg 

    # While in other parts of Cellpose channels are put last by default, here we have chan x Ly x Lx 
    if X[0].ndim>2:
        nchan = X[0].shape[0] 
    else:
        nchan = 1 
    
    nimg = len(X)
    imgi  = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)
        
    if Y is not None:
        for n in range(nimg):
            labels = Y[n].copy()
            if labels.ndim<3:
                labels = labels[np.newaxis,:,:]
            dist = labels[1]
            dist[dist==0] = - dist_bg
            if labels.shape[0]<6:
                bd = 5.*(labels[1]==1)
                bd[bd==0] = -5.
                labels = np.concatenate((labels, bd[np.newaxis,:]))# add a boundary layer
            if labels.shape[0]<7:
                mask = labels[0]>0
                labels = np.concatenate((labels, mask[np.newaxis,:])) # add a mask layer
            Y[n] = labels

        if Y[0].ndim>2:
            nt = Y[0].shape[0] +1 #(added one for weight array)
        else:
            nt = 1
    else:
        nt = 1
    lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)
    

    scale = np.zeros((nimg,2), np.float32)
    for n in range(nimg):
        img = X[n].copy()
        y = None if Y is None else Y[n]
        # use recursive function here to pass back single image that was cropped appropriately 
        imgi[n], lbl[n], scale[n] = random_crop_warp(img, y, nt, xy, nchan, scale[n], 
                                                     rescale is None if rescale is None else rescale[n], 
                                                     scale_range, gamma_range, do_flip, 
                                                     inds is None if inds is None else inds[n], dist_bg)
        
    return imgi, lbl, np.mean(scale) #for size training, must output scalar size (need to check this again)

# This function allows a more efficient implementation for recursively checking that the random crop includes cell pixels.
# Now it is rerun on a per-image basis if a crop fails to capture .1 percent cell pixels (minimum). 
def random_crop_warp(img, Y, nt, xy, nchan, scale, rescale, scale_range, gamma_range, do_flip, ind, dist_bg, depth=0):
    if depth>20:
        error_message = 'Sparse or over-dense image detected. Problematic index is: '+str(ind)
        omnipose_logger.critical(error_message)
        raise ValueError(error_message)
    
    if depth>100:
        error_message = 'Recusion depth exceeded. Check that your images contain cells and background within a typical crop. Failed index is: '+str(ind)
        omnipose_logger.critical(error_message)
        raise ValueError(error_message)
        return
    
    do_old = True # Recomputing flow will never work because labels are jagged...
    lbl = np.zeros((nt, xy[0], xy[1]), np.float32)
    numpx = xy[0]*xy[1]
    if Y is not None:
        labels = Y.copy()
        # We want the scale distibution to have a mean of 1
        # There may be a better way to skew the distribution to
        # interpolate the parameter space without skewing the mean 
        ds = scale_range/2
        if do_old:
            scale = np.random.uniform(low=1-ds,high=1+ds,size=2) #anisotropic
        else:
            scale = [np.random.uniform(low=1-ds,high=1+ds,size=1)]*2 # isotropic
        if rescale is not None:
            scale *= 1. / rescale

    # image dimensions are always the last two in the stack (again, convention here is different)
    Ly, Lx = img.shape[-2:]

    # generate random augmentation parameters
    dg = gamma_range/2 
    flip = np.random.choice([0,1])

    if do_old:
        theta = np.random.rand() * np.pi * 2
    else:
        theta = np.random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4]) 

    # random translation, take the difference between the scaled dimensions and the crop dimensions
    dxy = np.maximum(0, np.array([Lx*scale[1]-xy[1],Ly*scale[0]-xy[0]]))
    # multiplies by a pair of random numbers from -.5 to .5 (different for each dimension) 
    dxy = (np.random.rand(2,) - .5) * dxy 

    # create affine transform
    cc = np.array([Lx/2, Ly/2])
    # xy are the sizes of the cropped image, so this is the center coordinates minus half the difference
    cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
    # unit vectors from the center
    pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
    # transformed unit vectors
    pts2 = np.float32([cc1,
            cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
            cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
    M = cv2.getAffineTransform(pts1,pts2)


    method = cv2.INTER_LINEAR
    # the mode determines what happens with out of bounds regions. If we recompute the flow, we can
    # reflect all the scalar quantities then take the derivative. If we just rotate the field, then
    # the reflection messes up the directions. For now, we are returning to the default of padding
    # with zeros. In the future, we may only predict a scalar field and can use reflection to fill
    # the entire FoV with data - or we can work out how to properly extend the flow field. 
    if do_old:
        mode = 0
    else:
        mode = cv2.BORDER_DEFAULT # Does reflection 
        
    label_method = cv2.INTER_NEAREST
    
    imgi  = np.zeros((nchan, xy[0], xy[1]), np.float32)
    for k in range(nchan):
        I = cv2.warpAffine(img[k], M, (xy[1],xy[0]),borderMode=mode, flags=method)
        
        # gamma agumentation 
        gamma = np.random.uniform(low=1-dg,high=1+dg) 
        imgi[k] = I ** gamma
        
        # percentile clipping augmentation 
        dp = 10
        dpct = np.random.triangular(left=0, mode=0, right=dp, size=2) # weighted toward 0
        imgi[k] = utils.normalize99(imgi[k],upper=100-dpct[0],lower=dpct[1])
        
        # noise augmentation 
        if SKIMAGE_ENABLED:
            imgi[k] = random_noise(imgi[k], mode="poisson")
        else:
            imgi[k] = np.random.poisson(imgi[k])

    if Y is not None:
        for k in [0,1,2,3,4,5,6]: # was skipping 2 and 3, now not 
            
            if k==0:
                l = labels[k]
                lbl[k] = cv2.warpAffine(l, M, (xy[1],xy[0]), borderMode=mode, flags=label_method)

                # check to make sure the region contains at enough cell pixels; if not, retry
                cellpx = np.sum(lbl[0]>0)
                cutoff = (numpx/1000) # .1 percent of pixels must be cells
                if cellpx<cutoff or cellpx==numpx:
                    return random_crop_warp(img, Y, nt, xy, nchan, scale, rescale, scale_range, gamma_range, do_flip, ind, dist_bg, depth=depth+1)

            else:
                lbl[k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), borderMode=mode, flags=method)
        
        if nt > 1:
            
            mask = lbl[6]
            l = lbl[0].astype(int)
#                 smooth_dist = lbl[n,4].copy()
            dist = edt.edt(l,parallel=8) # raplace with smooth dist function 
            lbl[5] = dist==1 # boundary 

            if do_old:
                v1 = lbl[3].copy() # x component
                v2 = lbl[2].copy() # y component 
                dy = (-v1 * np.sin(-theta) + v2*np.cos(-theta))
                dx = (v1 * np.cos(-theta) + v2*np.sin(-theta))

                lbl[3] = 5.*dx*mask # factor of 5 is applied here to rescale flow components to [-5,5] range 
                lbl[2] = 5.*dy*mask
                
                smooth_dist = smooth_distance(l,dist)
                smooth_dist[dist<=0] = -dist_bg
                lbl[1] = smooth_dist
#                 dist[dist<=0] = -dist_bg
#                 lbl[1] = dist
            else:
#                 _, _, smooth_dist, mu = dynamics.masks_to_flows_gpu(l,dists=dist,omni=omni) #would want to replace this with a dedicated dist-only function
                lbl[3] = 5.*mu[1]
                lbl[2] = 5.*mu[0]

                smooth_dist[smooth_dist<=0] = -dist_bg
                lbl[1] = smooth_dist

            bg_edt = edt.edt(mask<0.5,black_border=True) #last arg gives weight to the border, which seems to always lose
            cutoff = 9
            lbl[7] = (gaussian(1-np.clip(bg_edt,0,cutoff)/cutoff, 1)+0.5)
    else:
        lbl = np.zeros((nt,imgi.shape[-2], imgi.shape[-1]))
    
    # Moved to the end because it conflicted with the recursion. Also, flipping the crop is ultimately equivalent and slightly faster. 
    if flip and do_flip:
        imgi = imgi[..., ::-1]
        if Y is not None:
            lbl = lbl[..., ::-1]
            if nt > 1:
                lbl[3] = -lbl[3]
    return imgi, lbl, scale

def loss(self, lbl, y):
    """ Loss function for Omnipose.
    
    Parameters
    --------------
    lbl: ND-array, float
        transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        lbl[:,0] cell masks
        lbl[:,1] distance fields
        lbl[:,2:4] flow fields
        lbl[:,4] distance fields
        lbl[:,5] boundary fields    
        lbl[:,6] thresholded mask layer 
        lbl[:,7] boundary-emphasized weights        
    
    y:  ND-tensor, float
        network predictions
        y[:,:2] flow fields
        y[:,2] distance fields
        y[:,3] boundary fields
    
    """
    
    veci = self._to_device(lbl[:,2:4]) #scaled to 5 in augmentation 
    dist = lbl[:,1] # now distance transform replaces probability
    boundary =  lbl[:,5]
    cellmask = dist>0
    w =  self._to_device(lbl[:,7])  
    dist = self._to_device(dist)
    boundary = self._to_device(boundary)
    cellmask = self._to_device(cellmask).bool()
    flow = y[:,:2] # 0,1
    dt = y[:,2]
    bd = y[:,3]
    a = 10.

    wt = torch.stack((w,w),dim=1)
    ct = torch.stack((cellmask,cellmask),dim=1) 

    loss1 = 10.*self.criterion12(flow,veci,wt)  #weighted MSE 
    loss2 = self.criterion14(flow,veci,w,cellmask) #ArcCosDotLoss
    loss3 = self.criterion11(flow,veci,wt,ct)/a # DerivativeLoss
    loss4 = 2.*self.criterion2(bd,boundary)
    loss5 = 2.*self.criterion15(flow,veci,w,cellmask) # loss on norm 
    loss6 = 2.*self.criterion12(dt,dist,w) #weighted MSE 
    loss7 = self.criterion11(dt.unsqueeze(1),dist.unsqueeze(1),w.unsqueeze(1),cellmask.unsqueeze(1))/a  

    return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7


