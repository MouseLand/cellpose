"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import string
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from cellpose import utils

cmap_emb = ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95), 100))


kp_colors = np.array([[0.55,0.55,0.55],
                      [0.,0.,1],
                      [0.8,0,0],
                      [1.,0.4,0.2],
                      [0,0.6,0.4],
                      [0.2,1,0.5],
                      ])

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titlelocation"] = "left"
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font

ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"

def plot_label(ltr, il, ax, trans, fs_title=20):
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il

def outlines_img(imgi, maski, color=[1,0,0], weight=2):
    img = np.tile(np.clip(imgi.copy(), 0, 1)[:,:,np.newaxis], (1,1,3))
    out = np.nonzero(utils.masks_to_outlines(maski[1:-1,1:-1]))
    img[out[0], out[1]] = np.array(color)
    if weight>1:
        if weight==2:
            ix, iy = np.meshgrid(np.arange(0, 3), np.arange(0, 3))
        else:
            ix = np.array([-1, 1, 0, 0])
            iy = np.array([0, 0, 1, 1])
        ix, iy = ix.flatten(), iy.flatten()
        for i in range(len(ix)):
            img[out[0]+ix[i], out[1]+iy[i]] = np.array(color)
    return img