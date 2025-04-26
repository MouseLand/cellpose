from cellpose import io, models, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
import torch
import numpy as np


#################### 2D Tests ####################
# def test_shape_2D_grayscale():
#     img = np.zeros((224, 224))
#     model = models.CellposeModel()
#     masks, _, _ = model.eval(img)
#     assert masks.shape == (224, 224)


def test_shape_2D_chan_first_diam_resize():
    img = np.zeros((1, 224, 224))
    model = models.CellposeModel()
    masks, _, _ = model.eval(img, diameter=50)
    assert masks.shape == (224, 224)


# def test_shape_2D_chan_diam_resize():
#     img = np.zeros((1, 224, 224))
#     model = models.CellposeModel()
#     masks, _, _ = model.eval(img, diameter=50)
#     assert masks.shape == (224, 224)


def test_shape_2D_chan_last():
    img = np.zeros((224, 224, 2))
    model = models.CellposeModel()
    masks, _, _ = model.eval(img)
    assert masks.shape == (224, 224)


# def test_shape_2D_chan_specify():
#     img = np.zeros((224, 224, 2))
#     model = models.CellposeModel()
#     masks, _, _ = model.eval(img, channel_axis=-1)
#     assert masks.shape == (224, 224)


def test_shape_2D_2chan_specify():
    img = np.zeros((224, 5, 224))
    model = models.CellposeModel()
    masks, flows, _ = model.eval(img, channels=[2, 1], channel_axis=1)
    assert masks.shape == (224, 224)
    

#################### 3D Tests ####################
def test_shape_stitch():
    img = np.zeros((5, 80, 80, 2)) # 5 layer 3d input, 2 channels
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    masks, _, _ = model.eval(img, channels=[0, 0],
                                    stitch_threshold=0.9, 
                                    channel_axis=3, z_axis=0, 
                                    do_3D=False)
    # Need do_3D=False to stitch
    assert masks.shape == (5, 80, 80)


# def test_shape_3D():
#     img = np.zeros((80, 80, 5, 1))
#     use_gpu = torch.cuda.is_available()
#     model = models.CellposeModel(gpu=use_gpu)
#     masks, _, _ = model.eval(img, channel_axis=3, z_axis=2, do_3D=True)
#     assert masks.shape == (5, 80, 80)


# def test_shape_3D_1ch():
#     img = np.zeros((5, 80, 80, 1))
#     use_gpu = torch.cuda.is_available()
#     model = models.CellposeModel(gpu=use_gpu)
#     masks, _, _ = model.eval(img, channel_axis=3, z_axis=0, do_3D=True)
#     assert masks.shape == (5, 80, 80)


# def test_shape_3D_1ch_3ndim():
#     img = np.zeros((5, 80, 80))
#     use_gpu = torch.cuda.is_available()
#     model = models.CellposeModel(gpu=use_gpu)
#     masks, _, _ = model.eval(img, channel_axis=None, z_axis=0, do_3D=True)
#     assert masks.shape == (5, 80, 80)


# def test_shape_3D_1ch_3ndim_diam():
#     img = np.zeros((5, 80, 80))
#     use_gpu = torch.cuda.is_available()
#     model = models.CellposeModel(gpu=use_gpu)
#     masks, _, _ = model.eval(img, channel_axis=None, diameter=50, z_axis=0, do_3D=True)
#     assert masks.shape == (5, 80, 80)


def test_shape_3D_2ch():
    img = np.zeros((80, 2, 80, 4))
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    masks, _, _ = model.eval(img, z_axis=-1, channel_axis=1, do_3D=True)
    assert masks.shape == (4, 80, 80)


# def test_shape_3D_rgb_diam():
#     img = np.zeros((5, 80, 80, 3))
#     use_gpu = torch.cuda.is_available()
#     model = models.CellposeModel(gpu=use_gpu)
#     masks, _, _ = model.eval(img, diameter=50, channels=[0, 0],
#                                     channel_axis=3, z_axis=0, do_3D=True)
#     assert masks.shape == (5, 80, 80)
    
# def test_shape_3D_rgb():
#     img = np.zeros((5, 80, 80, 3))
#     use_gpu = torch.cuda.is_available()
#     model = models.CellposeModel(gpu=use_gpu)
#     masks, _, _ = model.eval(img, channels=[0, 0],
#                                     channel_axis=3, z_axis=0, do_3D=True)
#     assert masks.shape == (5, 80, 80)