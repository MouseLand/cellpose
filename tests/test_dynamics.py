import pytest

from cellpose.core import use_gpu
from cellpose.dynamics import *


NO_GPU_AVAILABLE = not use_gpu()

BLOBS = np.array([[ 1, 0, 1, 1, 1],
                  [ 1, 1, 1, 0, 1],
                  [ 0, 1, 0, 1, 1],
                  [ 1, 0, 0, 1, 1],
                  [ 1, 0, 0, 0, 1]])
EMPTY = np.zeros_like(BLOBS, dtype=int)

@pytest.mark.skipif(NO_GPU_AVAILABLE, reason='Requires GPU support')
@pytest.mark.parametrize('masks', [BLOBS, EMPTY])
def test_masks_to_flows__gpu(masks):
    dists = edt.edt(masks)

    masks_to_flows(masks, dists, use_gpu=use_gpu)

    return


@pytest.mark.parametrize("masks", [BLOBS, EMPTY])
def test_masks_to_flows__non_gpu(masks):
    dists = edt.edt(masks) 

    masks_to_flows(masks, dists, use_gpu=False)

    return


def test_labels_to_flows__empty():
    label_shape = (16, 16)
    flow_shape = (4,) + label_shape

    labels = [np.zeros(label_shape, dtype=int)]

    flows = labels_to_flows(labels)
    
    assert all([flow.shape == flow_shape for flow in flows])
    assert all([np.all(flow == 0) for flow in flows])
    return