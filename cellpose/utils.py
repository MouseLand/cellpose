"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import logging
import os, tempfile, shutil, io
from tqdm import tqdm, trange
from urllib.request import urlopen
import cv2
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label
from scipy.spatial import ConvexHull
import numpy as np
import colorsys
import fastremap
import fill_voids
from multiprocessing import Pool, cpu_count
from cellpose import metrics

try:
    from skimage.morphology import remove_small_holes
    SKIMAGE_ENABLED = True
except:
    SKIMAGE_ENABLED = False


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h, s, v), axis=-1)
    return hsv


def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r, g, b), axis=-1)
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
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, "getheaders"):
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
        with tqdm(total=file_size, disable=not progress, unit="B", unit_scale=True,
                  unit_divisor=1024) as pbar:
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
    """Get the distance to the boundary of mask pixels.

    Args:
        masks (int, 2D or 3D array): The masks array. Size [Ly x Lx] or [Lz x Ly x Lx], where 0 represents no mask and 1, 2, ... represent mask labels.

    Returns:
        dist_to_bound (2D or 3D array): The distance to the boundary. Size [Ly x Lx] or [Lz x Ly x Lx].

    Raises:
        ValueError: If the masks array is not 2D or 3D.

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("distance_to_boundary takes 2D or 3D array, not %dD array" %
                         masks.ndim)
    dist_to_bound = np.zeros(masks.shape, np.float64)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            dist_to_bound[i] = distance_to_boundary(masks[i])
        return dist_to_bound
    else:
        slices = find_objects(masks)
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                ypix, xpix = np.nonzero(mask)
                min_dist = ((ypix[:, np.newaxis] - pvr)**2 +
                            (xpix[:, np.newaxis] - pvc)**2).min(axis=1)
                dist_to_bound[ypix + sr.start, xpix + sc.start] = min_dist
        return dist_to_bound


def masks_to_edges(masks, threshold=1.0):
    """Get edges of masks as a 0-1 array.

    Args:
        masks (int, 2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where 0=NO masks and 1,2,...=mask labels.
        threshold (float, optional): Threshold value for distance to boundary. Defaults to 1.0.

    Returns:
        edges (2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where True pixels are edge pixels.
    """
    dist_to_bound = distance_to_boundary(masks)
    edges = (dist_to_bound < threshold) * (masks > 0)
    return edges


def remove_edge_masks(masks, change_index=True):
    """Removes masks with pixels on the edge of the image.

    Args:
        masks (int, 2D or 3D array): The masks to be processed. Size [Ly x Lx] or [Lz x Ly x Lx], where 0 represents no mask and 1, 2, ... represent mask labels.
        change_index (bool, optional): If True, after removing masks, changes the indexing so that there are no missing label numbers. Defaults to True.

    Returns:
        outlines (2D or 3D array): The processed masks. Size [Ly x Lx] or [Lz x Ly x Lx], where 0 represents no mask and 1, 2, ... represent mask labels.
    """
    slices = find_objects(masks.astype(int))
    for i, si in enumerate(slices):
        remove = False
        if si is not None:
            for d, sid in enumerate(si):
                if sid.start == 0 or sid.stop == masks.shape[d]:
                    remove = True
                    break
            if remove:
                masks[si][masks[si] == i + 1] = 0
    shape = masks.shape
    if change_index:
        _, masks = np.unique(masks, return_inverse=True)
        masks = np.reshape(masks, shape).astype(np.int32)

    return masks


def masks_to_outlines(masks):
    """Get outlines of masks as a 0-1 array.

    Args:
        masks (int, 2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where 0=NO masks and 1,2,...=mask labels.

    Returns:
        outlines (2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where True pixels are outlines.
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" %
                         masks.ndim)
    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                vr, vc = pvr + sr.start, pvc + sc.start
                outlines[vr, vc] = 1
        return outlines


def outlines_list(masks, multiprocessing_threshold=1000, multiprocessing=None):
    """Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): Array of masks.
        multiprocessing_threshold (int, optional): Threshold for enabling multiprocessing. Defaults to 1000.
        multiprocessing (bool, optional): Flag to enable multiprocessing. Defaults to None.

    Returns:
        list: List of outlines.

    Raises:
        None

    Notes:
        - This function is a wrapper for outlines_list_single and outlines_list_multi.
        - Multiprocessing is disabled for Windows.
    """
    # default to use multiprocessing if not few_masks, but allow user to override
    if multiprocessing is None:
        few_masks = np.max(masks) < multiprocessing_threshold
        multiprocessing = not few_masks

    # disable multiprocessing for Windows
    if os.name == "nt":
        if multiprocessing:
            logging.getLogger(__name__).warning(
                "Multiprocessing is disabled for Windows")
        multiprocessing = False

    if multiprocessing:
        return outlines_list_multi(masks)
    else:
        return outlines_list_single(masks)


def outlines_list_single(masks):
    """Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.

    """
    outpix = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0, 2)))
    return outpix


def outlines_list_multi(masks, num_processes=None):
    """
    Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.
    """
    if num_processes is None:
        num_processes = cpu_count()

    unique_masks = np.unique(masks)[1:]
    with Pool(processes=num_processes) as pool:
        outpix = pool.map(get_outline_multi, [(masks, n) for n in unique_masks])
    return outpix


def get_outline_multi(args):
    """Get the outline of a specific mask in a multi-mask image.

    Args:
        args (tuple): A tuple containing the masks and the mask number.

    Returns:
        numpy.ndarray: The outline of the specified mask as an array of coordinates.

    """
    masks, n = args
    mn = masks == n
    if mn.sum() > 0:
        contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_NONE)
        contours = contours[-2]
        cmax = np.argmax([c.shape[0] for c in contours])
        pix = contours[cmax].astype(int).squeeze()
        return pix if len(pix) > 4 else np.zeros((0, 2))
    return np.zeros((0, 2))


def dilate_masks(masks, n_iter=5):
    """Dilate masks by n_iter pixels.

    Args:
        masks (ndarray): Array of masks.
        n_iter (int, optional): Number of pixels to dilate the masks. Defaults to 5.

    Returns:
        ndarray: Dilated masks.
    """
    dilated_masks = masks.copy()
    for n in range(n_iter):
        # define the structuring element to use for dilation
        kernel = np.ones((3, 3), "uint8")
        # find the distance to each mask (distances are zero within masks)
        dist_transform = cv2.distanceTransform((dilated_masks == 0).astype("uint8"),
                                               cv2.DIST_L2, 5)
        # dilate each mask and assign to it the pixels along the border of the mask
        # (does not allow dilation into other masks since dist_transform is zero there)
        for i in range(1, np.max(masks) + 1):
            mask = (dilated_masks == i).astype("uint8")
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            dilated_mask = np.logical_and(dist_transform < 2, dilated_mask)
            dilated_masks[dilated_mask > 0] = i
    return dilated_masks


def get_perimeter(points):
    """
    Calculate the perimeter of a set of points.

    Parameters:
        points (ndarray): An array of points with shape (npoints, ndim).

    Returns:
        float: The perimeter of the points.

    """
    if points.shape[0] > 4:
        points = np.append(points, points[:1], axis=0)
        return ((np.diff(points, axis=0)**2).sum(axis=1)**0.5).sum()
    else:
        return 0


def get_mask_compactness(masks):
    """
    Calculate the compactness of masks.
    
    Parameters:
        masks (ndarray): Binary masks representing objects.
        
    Returns:
        ndarray: Array of compactness values for each mask.
    """
    perimeters = get_mask_perimeters(masks)
    npoints = np.unique(masks, return_counts=True)[1][1:]
    areas = npoints
    compactness = 4 * np.pi * areas / perimeters**2
    compactness[perimeters == 0] = 0
    compactness[compactness > 1.0] = 1.0
    return compactness


def get_mask_perimeters(masks):
    """
    Calculate the perimeters of the given masks.

    Parameters:
        masks (numpy.ndarray): Binary masks representing objects.

    Returns:
        numpy.ndarray: Array containing the perimeters of each mask.
    """
    perimeters = np.zeros(masks.max())
    for n in range(masks.max()):
        mn = masks == (n + 1)
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)[-2]
            perimeters[n] = np.array(
                [get_perimeter(c.astype(int).squeeze()) for c in contours]).sum()

    return perimeters


def circleMask(d0):
    """
    Creates an array with indices which are the radius of that x,y point.

    Args:
        d0 (tuple): Patch of (-d0, d0+1) over which radius is computed.

    Returns:
        tuple: A tuple containing:
            - rs (ndarray): Array of radii with shape (2*d0[0]+1, 2*d0[1]+1).
            - dx (ndarray): Indices of the patch along the x-axis.
            - dy (ndarray): Indices of the patch along the y-axis.
    """
    dx = np.tile(np.arange(-d0[1], d0[1] + 1), (2 * d0[0] + 1, 1))
    dy = np.tile(np.arange(-d0[0], d0[0] + 1), (2 * d0[1] + 1, 1))
    dy = dy.transpose()

    rs = (dy**2 + dx**2)**0.5
    return rs, dx, dy


def get_mask_stats(masks_true):
    """
    Calculate various statistics for the given binary masks.

    Parameters:
        masks_true (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        convexity (ndarray): Convexity values for each mask.
        solidity (ndarray): Solidity values for each mask.
        compactness (ndarray): Compactness values for each mask.
    """
    mask_perimeters = get_mask_perimeters(masks_true)

    # disk for compactness
    rs, dy, dx = circleMask(np.array([100, 100]))
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
        points = np.array(np.nonzero(masks_true == (ic + 1))).T
        if len(points) > 15 and mask_perimeters[ic] > 0:
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

    convexity[mask_perimeters > 0.0] = (convex_perimeters[mask_perimeters > 0.0] /
                                        mask_perimeters[mask_perimeters > 0.0])
    solidity[convex_areas > 0.0] = (areas[convex_areas > 0.0] /
                                    convex_areas[convex_areas > 0.0])
    convexity = np.clip(convexity, 0.0, 1.0)
    solidity = np.clip(solidity, 0.0, 1.0)
    compactness = np.clip(compactness, 0.0, 1.0)
    return convexity, solidity, compactness


def get_masks_unet(output, cell_threshold=0, boundary_threshold=0):
    """Create masks using cell probability and cell boundary.

    Args:
        output (ndarray): The output array containing cell probability and cell boundary.
        cell_threshold (float, optional): The threshold value for cell probability. Defaults to 0.
        boundary_threshold (float, optional): The threshold value for cell boundary. Defaults to 0.

    Returns:
        ndarray: The masks representing the segmented cells.

    """
    cells = (output[..., 1] - output[..., 0]) > cell_threshold
    selem = generate_binary_structure(cells.ndim, connectivity=1)
    labels, nlabels = label(cells, selem)

    if output.shape[-1] > 2:
        slices = find_objects(labels)
        dists = 10000 * np.ones(labels.shape, np.float32)
        mins = np.zeros(labels.shape, np.int32)
        borders = np.logical_and(~(labels > 0), output[..., 2] > boundary_threshold)
        pad = 10
        for i, slc in enumerate(slices):
            if slc is not None:
                slc_pad = tuple([
                    slice(max(0, sli.start - pad), min(labels.shape[j], sli.stop + pad))
                    for j, sli in enumerate(slc)
                ])
                msk = (labels[slc_pad] == (i + 1)).astype(np.float32)
                msk = 1 - gaussian_filter(msk, 5)
                dists[slc_pad] = np.minimum(dists[slc_pad], msk)
                mins[slc_pad][dists[slc_pad] == msk] = (i + 1)
        labels[labels == 0] = borders[labels == 0] * mins[labels == 0]

    masks = labels
    shape0 = masks.shape
    _, masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape0)
    return masks


def stitch3D(masks, stitch_threshold=0.25):
    """
    Stitch 2D masks into a 3D volume using a stitch_threshold on IOU.

    Args:
        masks (list or ndarray): List of 2D masks.
        stitch_threshold (float, optional): Threshold value for stitching. Defaults to 0.25.

    Returns:
        list: List of stitched 3D masks.
    """
    mmax = masks[0].max()
    empty = 0
    for i in trange(len(masks) - 1):
        iou = metrics._intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
        if not iou.size and empty == 0:
            masks[i + 1] = masks[i + 1]
            mmax = masks[i + 1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i + 1].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, masks.dtype)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, masks.dtype)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1

    return masks


def diameters(masks):
    """
    Calculate the diameters of the objects in the given masks.

    Parameters:
    masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        tuple: A tuple containing the median diameter and an array of diameters for each object.

    Examples:
    >>> masks = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    >>> diameters(masks)
    (1.0, array([1.41421356, 1.0, 1.0]))
    """
    uniq, counts = fastremap.unique(masks.astype("int32"), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5) / 2
    return md, counts**0.5


def radius_distribution(masks, bins):
    """
    Calculate the radius distribution of masks.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)
        bins (int): Number of bins for the histogram.

    Returns:
        A tuple containing a normalized histogram of radii, median radius, array of radii.

    """
    unique, counts = np.unique(masks, return_counts=True)
    counts = counts[unique != 0]
    nb, _ = np.histogram((counts**0.5) * 0.5, bins)
    nb = nb.astype(np.float32)
    if nb.sum() > 0:
        nb = nb / nb.sum()
    md = np.median(counts**0.5) * 0.5
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5) / 2
    return nb, md, (counts**0.5) / 2


def size_distribution(masks):
    """
    Calculates the size distribution of masks.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        float: The ratio of the 25th percentile of mask sizes to the 75th percentile of mask sizes.
    """
    counts = np.unique(masks, return_counts=True)[1][1:]
    return np.percentile(counts, 25) / np.percentile(counts, 75)


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ Fills holes in masks (2D/3D) and discards masks smaller than min_size.

    This function fills holes in each mask using fill_voids.fill.
    It also removes masks that are smaller than the specified min_size.

    Parameters:
    masks (ndarray): Int, 2D or 3D array of labelled masks.
        0 represents no mask, while positive integers represent mask labels.
        The size can be [Ly x Lx] or [Lz x Ly x Lx].
    min_size (int, optional): Minimum number of pixels per mask.
        Masks smaller than min_size will be removed.
        Set to -1 to turn off this functionality. Default is 15.

    Returns:
        ndarray: Int, 2D or 3D array of masks with holes filled and small masks removed.
            0 represents no mask, while positive integers represent mask labels.
            The size is [Ly x Lx] or [Lz x Ly x Lx].
    """

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" %
                         masks.ndim)

    # Filter small masks
    if min_size > 0:
        counts = fastremap.unique(masks, return_counts=True)[1][1:]
        masks = fastremap.mask(masks, np.nonzero(counts < min_size)[0] + 1)
        fastremap.renumber(masks, in_place=True)
        
    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            msk = fill_voids.fill(msk)
            masks[slc][msk] = (j + 1)
            j += 1

    if min_size > 0:
        counts = fastremap.unique(masks, return_counts=True)[1][1:]
        masks = fastremap.mask(masks, np.nonzero(counts < min_size)[0] + 1)
        fastremap.renumber(masks, in_place=True)
    
    return masks
