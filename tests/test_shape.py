from cellpose import io, models, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
import torch
import numpy as np


#################### 2D Tests ####################
def test_shape_2D_grayscale():
    img = np.zeros((224, 224))
    model = models.CellposeModel()
    masks, _, _ = model.eval(img)
    assert masks.shape == (224, 224)


def test_shape_2D_chan_first():
    img = np.zeros((1, 224, 224))
    model = models.CellposeModel()
    masks, _, _ = model.eval(img)
    assert masks.shape == (224, 224)


def test_shape_2D_chan_last():
    img = np.zeros((224, 224, 2))
    model = models.CellposeModel()
    masks, _, _ = model.eval(img)
    assert masks.shape == (224, 224)


def test_shape_2D_2chan():
    img = np.zeros((224, 224))
    model = models.CellposeModel()
    masks, flows, _ = model.eval(img, channels=[2, 1], channel_axis=1)
    assert masks.shape == (224, 224)
    

#################### 3D Tests ####################
def test_shape_stitch():
    img = np.zeros((5, 224, 224, 2)) # 5 layer 3d input, 2 channels
    model = models.CellposeModel()
    masks, _, _ = model.eval(img, channels=[0, 0],
                                    stitch_threshold=0.9)
    assert masks.shape == (5, 224, 224)


def test_shape_3D():
    # This fails because the input suggests a 3D image, but ndim suggests 2D
    img = np.zeros((224, 224, 5))
    model = models.CellposeModel()
    masks, flows, _ = model.eval(img, diameter=30, channels=[0, 0],
                                    channel_axis=None, z_axis=3)
    assert masks.shape == (5, 224, 224)


def test_shape_3D_1ch():
    # This fails because the and channel_axis z_axis args are deprecated
    img = np.zeros((5, 224, 224, 1))
    model = models.CellposeModel()
    masks, flows, _ = model.eval(img, diameter=30, channels=[0, 0],
                                    channel_axis=None, z_axis=3)
    assert masks.shape == (5, 224, 224)


def test_shape_3D_1ch_pass():
    # passes
    img = np.zeros((5, 224, 224, 1))
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    masks, flows, _ = model.eval(img, do_3D=True)
    assert masks.shape == (5, 224, 224)


def test_shape_3D_rgb():
    img = np.zeros((5, 224, 224, 3))
    model = models.CellposeModel()
    masks, flows, _ = model.eval(img, diameter=30, channels=[0, 0],
                                    channel_axis=None, z_axis=3)
    assert masks.shape == (5, 224, 224)