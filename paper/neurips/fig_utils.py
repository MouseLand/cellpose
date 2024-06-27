"""
Copyright © 2024 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import string
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from cellpose import utils

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