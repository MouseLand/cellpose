import pytest

from cellpose.core import use_gpu
from cellpose.dynamics import *


NO_GPU_AVAILABLE = not use_gpu()


@pytest.mark.skipif(NO_GPU_AVAILABLE, reason='Requires GPU support')
def test_masks_to_flows_gpu__blobs():
    masks = np.array([[ True, False,  True,  True,  True],
                      [ True,  True,  True, False,  True],
                      [False,  True, False,  True,  True],
                      [ True, False, False,  True,  True],
                      [ True, False, False, False,  True]])
    dists = edt.edt(masks)

    masks_to_flows_gpu(masks, dists)
    return


@pytest.mark.skipif(NO_GPU_AVAILABLE, reason='Requires GPU support')
def test_masks_to_flows_gpu__empty():
    masks = np.zeros((16, 16), dtype=int)
    dists = edt.edt(masks) 

    masks_to_flows_gpu(masks, dists)
    return


def test_labels_to_flows__empty():
    label_shape = (16, 16)
    flow_shape = (4,) + label_shape

    labels = [np.zeros(label_shape, dtype=int)]

    flows = labels_to_flows(labels)
    
    assert all([flow.shape == flow_shape for flow in flows])
    assert all([np.all(flow == 0) for flow in flows])
    return