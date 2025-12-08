import logging
import numpy as np

from typing import Tuple, List


logger = logging.getLogger(__name__)


def get_block_crops(shape, blocksize, overlaps, mask):
    """
    Given a voxel grid shape, blocksize, and overlap size, construct
       tuples of slices for every block; optionally only include blocks
       that contain foreground in the mask. Returns parallel lists,
       the block indices and the slice tuples.
    """
    blocksize = np.array(blocksize, dtype=int)
    blockoverlaps = np.array(overlaps, dtype=int)

    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)
    else:
        mask_blocksize = None

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - blockoverlaps
        stop = start + blocksize + 2 * blockoverlaps
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]):
                foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)

    return indices, crops


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def prepare_blocksize(shape: Tuple[int, ...]|List[int],
                      blocksize: Tuple[int, ...]|List[int]) -> List[int]:
    ndim = len(shape)
    blocksize_ndim = len(blocksize)
    final_blocksize = []

    # the blocksize may have fewer elements than the image shape
    # in that case we right align it to shape
    # if somehow blocksize has more elements than shape
    # we drop the first elements until the sizes match
    offset = ndim - blocksize_ndim
    
    for si in range(ndim):
        final_blocksize.append(shape[si] if si < offset else blocksize[si - offset])

    return final_blocksize


def prepare_overlaps(shape: Tuple[int, ...], 
                     blocksize: Tuple[int, ...]|List[int],
                     blockoverlaps: Tuple[int, ...]|List[int]|None,
                     default_overlap: float|None=None) -> List[int]:
    ndim = len(shape)
    blocksize_ndim = len(blocksize)

    def default_overlap_forsize(s):
        return int(s * 0.1) if default_overlap is None else int(default_overlap)

    # If overlaps not provided (None / empty), compute defaults for every blocksize dim
    if not blockoverlaps:
        offset = ndim - blocksize_ndim  # blocksize is right-aligned to shape
        return [
            0 if blocksize[i] == shape[i + offset] 
              else default_overlap_forsize(blocksize[i])
            for i in range(blocksize_ndim)
        ]

    # If overlaps provided as list/tuple, right-align overlaps to blocksize
    if isinstance(blockoverlaps, (list, tuple)):
        bo_ndim = len(blockoverlaps)
        offset_shape = ndim - blocksize_ndim
        offset_ov = max(blocksize_ndim - bo_ndim, 0)  # how much overlaps lags behind blocksize
        return [
            0 if blocksize[i] == shape[i + offset_shape]
            else (int(blockoverlaps[i - offset_ov])
                  if i >= offset_ov 
                  else default_overlap_forsize(blocksize[i]))
            for i in range(blocksize_ndim)
        ]

    raise ValueError(f"Invalid block overlaps argument: {blockoverlaps}")


def remove_overlaps(array, crop, overlaps, blocksize):
    """
    Overlaps are only there to provide context for boundary voxels
    and can be removed after segmentation is complete
    reslice array to remove the overlaps
    """
    logger.debug((
        f'Remove overlaps: {overlaps} '
        f'crop: {crop} '
        f'blocksize is {blocksize} '
        f'block shape: {array.shape} '
    ))
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        # left side
        if crop[axis].start != 0:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(overlaps[axis], None)
            loverlap_index = tuple(slc)
            logger.debug((
                f'Remove left overlap on axis {axis}: {loverlap_index} ({type(loverlap_index)}) '
                f'from labeled block of shape: {array.shape} '
            ))
            array = array[loverlap_index]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlaps[axis], b)
        # right side
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(None, blocksize[axis])
            roverlap_index = tuple(slc)
            logger.debug((
                f'Remove right overlap on axis {axis}: {roverlap_index} ({type(roverlap_index)}) '
                f'from labeled block of shape: {array.shape} '
            ))
            array = array[roverlap_index]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed
