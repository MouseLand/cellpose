from cellpose import io, models, metrics, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
import numpy as np

def test_shape_2D():
    img = np.zeros((224,1,224))
    model = models.Cellpose(model_type='cyto')
    masks, flows, _, _ = model.eval(img, diameter=30, channels=[0,0], 
                                    net_avg=False, channel_axis=1)
    assert masks.shape==(224,224)

def test_shape_3D():
    img = np.zeros((224,224,1,5,1))
    model = models.Cellpose(model_type='cyto')
    masks, flows, _, _ = model.eval(img, diameter=30, channels=[0,0], 
                                    net_avg=False, channel_axis=None, z_axis=3)
    assert masks.shape==(5,224,224)

def test_shape_stitch():
    img = np.zeros((5,224,224))
    model = models.Cellpose(model_type='cyto')
    masks, flows, _, _ = model.eval(img, diameter=30, channels=[0,0], 
                                    net_avg=False, stitch_threshold=0.9)
    assert masks.shape==(5,224,224)

def test_shape_2D_2chan():
    img = np.zeros((224,3,224))
    model = models.Cellpose(model_type='cyto')
    masks, flows, _, _ = model.eval(img, diameter=30, channels=[2,1], 
                                    net_avg=False, channel_axis=1)
    assert masks.shape==(224,224)