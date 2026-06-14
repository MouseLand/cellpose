"""
Utilities for deleting and relabeling GUI masks.
"""

import numpy as np


def normalize_remove_ids(remove_ids, ncells):
    """Return unique valid label IDs in descending order."""
    remove_ids = np.asarray(remove_ids, dtype=np.int64).reshape(-1)
    if remove_ids.size == 0 or ncells <= 0:
        return np.zeros(0, dtype=np.int64)
    valid = (remove_ids > 0) & (remove_ids <= int(ncells))
    if not np.any(valid):
        return np.zeros(0, dtype=np.int64)
    remove_ids = np.unique(remove_ids[valid])
    return np.sort(remove_ids)[::-1]


def batch_delete_reindex(cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids):
    """Delete labels and reindex all state in one pass.

    Returns updated `(cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids, remove_mask)`.
    """
    if cellpix.shape != outpix.shape:
        raise ValueError("cellpix and outpix must have the same shape")

    ncells = int(len(cellcolors) - 1)
    remove_ids = normalize_remove_ids(remove_ids, ncells)
    if remove_ids.size == 0:
        remove_mask = np.zeros(ncells + 1, dtype=bool)
        return (
            cellpix,
            outpix,
            ismanual,
            cellcolors,
            list(zdraw),
            remove_ids,
            remove_mask,
        )

    remove_mask = np.zeros(ncells + 1, dtype=bool)
    remove_mask[remove_ids] = True
    keep_mask = ~remove_mask

    lut_dtype = cellpix.dtype if np.issubdtype(cellpix.dtype, np.integer) else np.int64
    relabel_map = np.cumsum(keep_mask, dtype=lut_dtype) - 1
    relabel_map[remove_mask] = 0

    cellpix = relabel_map[cellpix]
    outpix = relabel_map[outpix]
    ismanual = ismanual[keep_mask[1:]]
    cellcolors = cellcolors[keep_mask]
    zdraw = [z for z, keep in zip(zdraw, keep_mask[1:]) if keep]

    return cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids, remove_mask
