import os, warnings, time, tempfile, datetime, pathlib, shutil
from tqdm import tqdm
from urllib.request import urlopen
from urllib.parse import urlparse
import cv2
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
import numpy as np
import mxnet as mx



def use_gpu(gpu_number=0):
    """ check if mxnet gpu works """
    try:
        _ = mx.ndarray.array([1, 2, 3], ctx=mx.gpu(gpu_number))
        print('CUDA version installed and working.')
        return True
    except mx.MXNetError:
        print('CUDA version not installed/working, will use CPU version.')
        return False

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
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
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
                pvc, pvr = np.concatenate(contours[0], axis=0).squeeze().T  
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
        slices = find_objects(masks)
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[0], axis=0).squeeze().T            
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
            contours = contours[0]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix)>4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0,2)))
    return outpix

def get_masks_unet(output, cell_threshold=0, boundary_threshold=0):
    """ create masks using cell probability and cell boundary """
    cells = (output[:,:,1] - output[:,:,0])>cell_threshold
    selem = generate_binary_structure(cells.ndim, connectivity=1)
    labels, nlabels = label(cells, selem)

    if output.shape[-1]>2:
        slices = find_objects(labels)
        dists = 10000*np.ones(labels.shape, np.float32)
        mins = np.zeros(labels.shape, np.int32)
        borders = np.logical_and(~(labels>0), output[:,:,2]>boundary_threshold)
        pad = 10
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                sr = slice(sr.start-pad, sr.stop+pad)
                sc = slice(sc.start-pad, sc.stop+pad)
                ly, lx = sr.stop - sr.start + pad, sc.stop - sc.start + pad
                msk = (labels[sr,sc] == (i+1)).astype(np.float32)
                msk = 1 - gaussian_filter(msk, 5)
                dists[sr,sc] = np.minimum(dists[sr,sc], msk)
                mins[sr,sc][dists[sr,sc]==msk] = (i+1)
        labels[labels==0] = borders[labels==0] * mins[labels==0]
        
    masks = labels
    masks = fill_holes(masks)
    shape0 = masks.shape
    _,masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape0)
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


def fill_holes(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    
    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    
    slices = find_objects(masks)
    i = 0
    for slc in slices:
        if slc is not None:
            msk = masks[slc] == (i+1)
            if msk.ndim==3:
                small_objects = np.zeros(msk.shape, np.bool)
                for k in range(msk.shape[0]):
                    msk[k] = binary_fill_holes(msk[k])
                    #small_objects[k] = ~remove_small_objects(msk[k], min_size=min_size)
            else:
                msk = binary_fill_holes(msk)
                small_objects = ~remove_small_objects(msk, min_size=min_size)
            sm = np.logical_and(msk, small_objects)
            #~skimage.morphology.remove_small_objects(msk, min_size=min_size, connectivity=1))
            masks[slc][msk] = (i+1)
            masks[slc][sm] = 0
        i+=1
    return masks

def remove_small_objects(ar, min_size=64, connectivity=1):
    """ copied from skimage.morphology.remove_small_objects (required to be separate for pyinstaller) """
    out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
