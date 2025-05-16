import numpy as np
import pytest


#################### 2D Tests ####################
def test_shape_2D_grayscale(cellposemodel_fixture_24layer):
    img = np.zeros((224, 224))
    masks, _, _ = cellposemodel_fixture_24layer.eval(img)
    assert masks.shape == (224, 224)


def test_shape_2D_chan_first_diam_resize(cellposemodel_fixture_24layer):
    img = np.zeros((1, 224, 224))
    masks, flows, _ = cellposemodel_fixture_24layer.eval(img, diameter=50)
    assert masks.shape == (224, 224), 'mask shape mismatch'
    assert flows[1].shape == (2, 224, 224), 'dP shape mismatch'
    assert flows[2].shape == (224, 224), 'cellprob shape mismatch'


@pytest.mark.slow
def test_shape_2D_chan_diam_resize(cellposemodel_fixture_24layer):
    img = np.zeros((1, 224, 224))
    masks, _, _ = cellposemodel_fixture_24layer.eval(img, diameter=50)
    assert masks.shape == (224, 224)


def test_shape_2D_chan_last(cellposemodel_fixture_24layer):
    img = np.zeros((224, 224, 2))
    masks, flows, _ = cellposemodel_fixture_24layer.eval(img)
    assert masks.shape == (224, 224), 'mask shape mismatch'
    assert flows[1].shape == (2, 224, 224), 'dP shape mismatch'
    assert flows[2].shape == (224, 224), 'cellprob shape mismatch'



@pytest.mark.slow
def test_shape_2D_chan_specify(cellposemodel_fixture_24layer):
    img = np.zeros((224, 224, 2))
    masks, _, _ = cellposemodel_fixture_24layer.eval(img, channel_axis=-1)
    assert masks.shape == (224, 224)


def test_shape_2D_2chan_specify(cellposemodel_fixture_24layer):
    img = np.zeros((224, 5, 224))
    masks, flows, _ = cellposemodel_fixture_24layer.eval(img, channels=[2, 1], channel_axis=1)
    assert masks.shape == (224, 224), 'mask shape mismatch'
    assert flows[1].shape == (2, 224, 224), 'dP shape mismatch'
    assert flows[2].shape == (224, 224), 'cellprob shape mismatch'
    

#################### 3D Tests ####################
def test_shape_stitch(cellposemodel_fixture_2layer):
    img = np.zeros((5, 80, 80, 2)) # 5 layer 3d input, 2 channels
    masks, flows, _ = cellposemodel_fixture_2layer.eval(img, channels=[0, 0],
                                    stitch_threshold=0.9, 
                                    channel_axis=3, z_axis=0, 
                                    do_3D=False)
    # Need do_3D=False to stitch, returns dim0 shape 2 because of stitching 
    assert masks.shape == (5, 80, 80), 'mask shape mismatch'
    assert flows[1].shape == (2, 5, 80, 80), 'dP shape mismatch'
    assert flows[2].shape == (5, 80, 80), 'cellprob shape mismatch'


@pytest.mark.slow
def test_shape_3D(cellposemodel_fixture_2layer):
    img = np.zeros((80, 80, 5, 1))
    masks, _, _ = cellposemodel_fixture_2layer.eval(img, channel_axis=3, z_axis=2, do_3D=True)
    assert masks.shape == (5, 80, 80)


@pytest.mark.slow
def test_shape_3D_1ch(cellposemodel_fixture_2layer):
    img = np.zeros((5, 80, 80, 1))
    masks, _, _ = cellposemodel_fixture_2layer.eval(img, channel_axis=3, z_axis=0, do_3D=True)
    assert masks.shape == (5, 80, 80)


@pytest.mark.slow
def test_shape_3D_1ch_3ndim(cellposemodel_fixture_2layer):
    img = np.zeros((5, 80, 80))
    masks, _, _ = cellposemodel_fixture_2layer.eval(img, channel_axis=None, z_axis=0, do_3D=True)
    assert masks.shape == (5, 80, 80)


@pytest.mark.slow
def test_shape_3D_1ch_3ndim_diam(cellposemodel_fixture_2layer):
    img = np.zeros((5, 80, 80))
    masks, _, _ = cellposemodel_fixture_2layer.eval(img, channel_axis=None, diameter=50, z_axis=0, do_3D=True)
    assert masks.shape == (5, 80, 80)


def test_shape_3D_2ch(cellposemodel_fixture_2layer):
    img = np.zeros((80, 2, 80, 4))

    masks, flows, _ = cellposemodel_fixture_2layer.eval(img, z_axis=-1, channel_axis=1, do_3D=True)
    assert masks.shape == (4, 80, 80), 'mask shape mismatch'
    assert flows[1].shape == (3, 4, 80, 80), 'dP shape mismatch'
    assert flows[2].shape == (4, 80, 80), 'cellprob shape mismatch'


@pytest.mark.slow
def test_shape_3D_rgb_diam(cellposemodel_fixture_2layer):
    img = np.zeros((5, 80, 80, 3))
    masks, _, _ = cellposemodel_fixture_2layer.eval(img, diameter=50, channels=[0, 0],
                                    channel_axis=3, z_axis=0, do_3D=True)
    assert masks.shape == (5, 80, 80)
    
@pytest.mark.slow
def test_shape_3D_rgb(cellposemodel_fixture_2layer):
    img = np.zeros((5, 80, 80, 3))
    masks, _, _ = cellposemodel_fixture_2layer.eval(img, channels=[0, 0],
                                    channel_axis=3, z_axis=0, do_3D=True)
    assert masks.shape == (5, 80, 80)