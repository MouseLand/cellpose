import pytest
import numpy as np
import torch
from cellpose import io, models
from cellpose.contrib import openvino_utils


def create_model():
    return models.CellposeModel(gpu=False,
                                pretrained_model="cyto",
                                net_avg=True,
                                device=torch.device("cpu"))

def test_unet(data_dir):
    image_name = 'rgb_2D.png'
    img = io.imread(str(data_dir.joinpath('2D').joinpath(image_name)))

    # Get a reference results
    ref_model = create_model()
    ref_masks, ref_flows, ref_styles = ref_model.eval(img, net_avg=True)

    # Convert model to OpenVINO format
    ov_model = create_model()
    ov_model = openvino_utils.to_openvino(ov_model)

    out_masks, out_flows, out_styles = ov_model.eval(img, net_avg=True)

    assert ref_masks.shape == out_masks.shape
    assert ref_styles.shape == out_styles.shape

    assert np.all(ref_masks == out_masks)
    assert np.max(np.abs(ref_styles - out_styles)) < 1e-5

    for ref_flow, out_flow in zip(ref_flows, out_flows):
        if ref_flow is None or np.prod(ref_flow.shape) == 0:
            continue

        assert ref_flow.shape == out_flow.shape
        assert np.max(np.abs(ref_flow - out_flow)) < 1e-4
