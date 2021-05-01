import os, warnings, time, tempfile, datetime, pathlib, shutil, subprocess
from tqdm import tqdm
from urllib.request import urlopen
from urllib.parse import urlparse
import cv2
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
from scipy.spatial import ConvexHull
import numpy as np
import colorsys
import io

from . import metrics

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h,s,v), axis=-1)
    return hsv

def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.
            Thanks to torch, slightly modified
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def distance_to_boundary(masks):
    """ get distance to boundary of mask pixels
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    dist_to_bound: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx]

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('distance_to_boundary takes 2D or 3D array, not %dD array'%masks.ndim)
    dist_to_bound = np.zeros(masks.shape, np.float64)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            dist_to_bound[i] = distance_to_boundary(masks[i])
        return dist_to_bound
    else:
        slices = find_objects(masks)
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T  
                ypix, xpix = np.nonzero(mask)
                min_dist = ((ypix[:,np.newaxis] - pvr)**2 + 
                            (xpix[:,np.newaxis] - pvc)**2).min(axis=1)
                dist_to_bound[ypix + sr.start, xpix + sc.start] = min_dist
        return dist_to_bound

def masks_to_edges(masks, threshold=1.0):
    """ get edges of masks as a 0-1 array 
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    edges: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are edge pixels

    """
    dist_to_bound = distance_to_boundary(masks)
    edges = (dist_to_bound < threshold) * (masks > 0)
    return edges

def remove_edge_masks(masks, change_index=True):
    """ remove masks with pixels on edge of image
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    change_index: bool (optional, default True)
        if True, after removing masks change indexing so no missing label numbers

    Returns
    ----------------

    outlines: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    """
    slices = find_objects(masks.astype(int))
    for i,si in enumerate(slices):
        remove = False
        if si is not None:
            for d,sid in enumerate(si):
                if sid.start==0 or sid.stop==masks.shape[d]:
                    remove=True
                    break  
            if remove:
                masks[si][masks[si]==i+1] = 0
    shape = masks.shape
    if change_index:
        _,masks = np.unique(masks, return_inverse=True)
        masks = np.reshape(masks, shape).astype(np.int32)

    return masks

def masks_to_outlines(masks):
    """ get outlines of masks as a 0-1 array 
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    outlines: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    outlines = np.zeros(masks.shape, np.bool)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
                vr, vc = pvr + sr.start, pvc + sc.start 
                outlines[vr, vc] = 1
        return outlines

def outlines_list(masks):
    """ get outlines of masks as a list to loop over for plotting """
    outpix=[]
    for n in np.unique(masks)[1:]:
        mn = masks==n
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix)>4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0,2)))
    return outpix

def get_perimeter(points):
    """ perimeter of points - npoints x ndim """
    if points.shape[0]>4:
        points = np.append(points, points[:1], axis=0)
        return ((np.diff(points, axis=0)**2).sum(axis=1)**0.5).sum()
    else:
        return 0

def get_mask_compactness(masks):
    perimeters = get_mask_perimeters(masks)
    #outlines = masks_to_outlines(masks)
    #perimeters = np.unique(outlines*masks, return_counts=True)[1][1:]
    npoints = np.unique(masks, return_counts=True)[1][1:]
    areas = npoints
    compactness =  4 * np.pi * areas / perimeters**2
    compactness[perimeters==0] = 0
    compactness[compactness>1.0] = 1.0
    return compactness

def get_mask_perimeters(masks):
    """ get perimeters of masks """
    perimeters = np.zeros(masks.max())
    for n in range(masks.max()):
        mn = masks==(n+1)
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)[-2]
            #cmax = np.argmax([c.shape[0] for c in contours])
            #perimeters[n] = get_perimeter(contours[cmax].astype(int).squeeze())
            perimeters[n] = np.array([get_perimeter(c.astype(int).squeeze()) for c in contours]).sum()

    return perimeters

def circleMask(d0):
    """ creates array with indices which are the radius of that x,y point
        inputs:
            d0 (patch of (-d0,d0+1) over which radius computed
        outputs:
            rs: array (2*d0+1,2*d0+1) of radii
            dx,dy: indices of patch
    """
    dx  = np.tile(np.arange(-d0[1],d0[1]+1), (2*d0[0]+1,1))
    dy  = np.tile(np.arange(-d0[0],d0[0]+1), (2*d0[1]+1,1))
    dy  = dy.transpose()

    rs  = (dy**2 + dx**2) ** 0.5
    return rs, dx, dy

def get_mask_stats(masks_true):
    mask_perimeters = get_mask_perimeters(masks_true)

    # disk for compactness
    rs,dy,dx = circleMask(np.array([100, 100]))
    rsort = np.sort(rs.flatten())

    # area for solidity
    npoints = np.unique(masks_true, return_counts=True)[1][1:]
    areas = npoints - mask_perimeters / 2 - 1
    
    compactness = np.zeros(masks_true.max())
    convexity = np.zeros(masks_true.max())
    solidity = np.zeros(masks_true.max())
    convex_perimeters = np.zeros(masks_true.max())
    convex_areas = np.zeros(masks_true.max())
    for ic in range(masks_true.max()):
        points = np.array(np.nonzero(masks_true==(ic+1))).T
        if len(points)>15 and mask_perimeters[ic] > 0:
            med = np.median(points, axis=0)
            # compute compactness of ROI
            r2 = ((points - med)**2).sum(axis=1)**0.5
            compactness[ic] = (rsort[:r2.size].mean() + 1e-10) / r2.mean()
            try:
                hull = ConvexHull(points)
                convex_perimeters[ic] = hull.area
                convex_areas[ic] = hull.volume
            except:
                convex_perimeters[ic] = 0
                
    convexity[mask_perimeters>0.0] = (convex_perimeters[mask_perimeters>0.0] / 
                                      mask_perimeters[mask_perimeters>0.0])
    solidity[convex_areas>0.0] = (areas[convex_areas>0.0] / 
                                     convex_areas[convex_areas>0.0])
    convexity = np.clip(convexity, 0.0, 1.0)
    solidity = np.clip(solidity, 0.0, 1.0)
    compactness = np.clip(compactness, 0.0, 1.0)
    return convexity, solidity, compactness

def get_masks_unet(output, cell_threshold=0, boundary_threshold=0):
    """ create masks using cell probability and cell boundary """
    cells = (output[...,1] - output[...,0])>cell_threshold
    selem = generate_binary_structure(cells.ndim, connectivity=1)
    labels, nlabels = label(cells, selem)

    if output.shape[-1]>2:
        slices = find_objects(labels)
        dists = 10000*np.ones(labels.shape, np.float32)
        mins = np.zeros(labels.shape, np.int32)
        borders = np.logical_and(~(labels>0), output[...,2]>boundary_threshold)
        pad = 10
        for i,slc in enumerate(slices):
            if slc is not None:
                slc_pad = tuple([slice(max(0,sli.start-pad), min(labels.shape[j], sli.stop+pad))
                                    for j,sli in enumerate(slc)])
                msk = (labels[slc_pad] == (i+1)).astype(np.float32)
                msk = 1 - gaussian_filter(msk, 5)
                dists[slc_pad] = np.minimum(dists[slc_pad], msk)
                mins[slc_pad][dists[slc_pad]==msk] = (i+1)
        labels[labels==0] = borders[labels==0] * mins[labels==0]
        
    masks = labels
    shape0 = masks.shape
    _,masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape0)
    return masks

def stitch3D(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        iou = metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
        iou[iou < stitch_threshold] = 0.0
        iou[iou < iou.max(axis=0)] = 0.0
        istitch = iou.argmax(axis=1) + 1
        ino = np.nonzero(iou.max(axis=1)==0.0)[0]
        istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)
        masks[i+1] = istitch[masks[i+1]]
    return masks

def diameters(masks):
    """ get median 'diameter' of masks """
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5)/2
    return md, counts**0.5

def radius_distribution(masks, bins):
    unique, counts = np.unique(masks, return_counts=True)
    counts = counts[unique!=0]
    nb, _ = np.histogram((counts**0.5)*0.5, bins)
    nb = nb.astype(np.float32)
    if nb.sum() > 0:
        nb = nb / nb.sum()
    md = np.median(counts**0.5)*0.5
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5)/2
    return nb, md, (counts**0.5)/2

def size_distribution(masks):
    counts = np.unique(masks, return_counts=True)[1][1:]
    return np.percentile(counts, 25) / np.percentile(counts, 75)

def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def process_cells(M0, npix=20):
    unq, ic = np.unique(M0, return_counts=True)
    for j in range(len(unq)):
        if ic[j]<npix:
            M0[M0==unq[j]] = 0
    return M0


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    
    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('fill_holes_and_remove_small_masks takes 2D or 3D array, not %dD array'%masks.ndim)
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:    
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks