import numpy as np
from cellpose.utils import fill_holes_and_remove_small_masks
import fastremap


def test_fill_holes_and_remove_small_masks():
    # make a 2-channel mask with holes and small objects. The first channel is the
    # "ground truth" without holes or small objects, the second channel needs to be cleaned.
    masks = np.zeros((2, 100, 100), dtype=np.uint16)
    masks[:, 10:30, 10:30] = 1  # object 1
    masks[1, 15:25, 15:25] = 0  # hole in object 1
    masks[1, 40:45, 40:45] = 2  # small object 2
    masks[:, 60:90, 60:90] = 4  # object 4 (skip 3)
    masks[1, 70:80, 70:80] = 0  # hole in object 4
    masks[1, 10:15, 80:82] = 5  # small object 4

    # apply function
    min_size = 30
    masks_cleaned = fill_holes_and_remove_small_masks(masks[1], min_size=min_size)

    gt_masks = fastremap.renumber(masks[0], in_place=False)[0]

    assert (gt_masks == masks_cleaned).all()
