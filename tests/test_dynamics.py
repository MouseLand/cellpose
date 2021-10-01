import pytest

from cellpose.core import use_gpu
from cellpose.dynamics import *


NO_GPU_AVAILABLE = not use_gpu()

## mask_to_flows_gpu
@pytest.mark.skipif(NO_GPU_AVAILABLE, reason='Requires GPU support')
def test_masks_to_flows_gpu__blobs():
    from skimage import data
    masks = data.binary_blobs(length=5, blob_size_fraction=0.2) 
    dists = edt.edt(masks)

    masks_to_flows_gpu(masks, dists)

    return


@pytest.mark.skipif(NO_GPU_AVAILABLE, reason='Requires GPU support')
def test_masks_to_flows_gpu__empty():
    masks = np.zeros((16, 16), dtype=int)
    dists = edt.edt(masks) 

    masks_to_flows_gpu(masks, dists)

    return