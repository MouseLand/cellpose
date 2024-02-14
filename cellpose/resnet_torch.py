"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime

from . import transforms, io, dynamics, utils

def batchconv(in_channels, out_channels, sz, conv_3D=False):
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum = 0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz//2),
    )  

def batchconv0(in_channels, out_channels, sz, conv_3D=False):
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum = 0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz//2),
    )  

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj  = batchconv0(in_channels, out_channels, 1, conv_3D)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, conv_3D))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, conv_3D))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class downsample(nn.Module):
    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        super().__init__()
        self.down = nn.Sequential()
        if max_pool:
            self.maxpool = nn.MaxPool3d(2, stride=2) if conv_3D else nn.MaxPool2d(2, stride=2)
        else:
            self.maxpool = nn.AvgPool3d(2, stride=2) if conv_3D else nn.AvgPool2d(2, stride=2)
        for n in range(len(nbase)-1):
            self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz, conv_3D))
            
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = self.maxpool(xd[n-1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
    
class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz, conv_3D)
        self.full = nn.Linear(style_channels, out_channels)
        
    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            x = x + y
        feat = self.full(style)
        for k in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y
    
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, 
                 concatenation=False, conv_3D=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, conv_3D=conv_3D))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, conv_3D=conv_3D))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz, conv_3D=conv_3D))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz, conv_3D=conv_3D))
        self.proj  = batchconv0(in_channels, out_channels, 1, conv_3D=conv_3D)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x
    
class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, 
                 concatenation=False, conv_3D=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, conv_3D))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, 
                                                concatenation=concatenation, conv_3D=conv_3D))
        
    def forward(self, x, y, style, mkldnn=False):
        x = self.conv[1](style, self.conv[0](x), y=y)
        return x
    
class make_style(nn.Module):
    def __init__(self, conv_3D=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d if conv_3D else F.avg_pool2d

    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style
    
class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, conv_3D=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1,len(nbase)):
            self.up.add_module('res_up_%d'%(n-1), 
                resup(nbase[n], nbase[n-1], nbase[-1], sz, conv_3D))
            
    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up)-2,-1,-1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x
    
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz, mkldnn=False,
                conv_3D=False, max_pool=True,
                diam_mean=30.):
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = True
        self.style_on = True
        self.concatenation = False
        self.conv_3D = conv_3D
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, conv_3D=conv_3D, max_pool=max_pool)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, conv_3D=conv_3D)
        self.make_style = make_style(conv_3D=conv_3D)
        self.output = batchconv(nbaseup[0], nout, 1, conv_3D=conv_3D)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0    = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense()) 
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T1 = self.upsample(style, T0, self.mkldnn)
        T1    = self.output(T1)
        if self.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()    
        return T1, style0, T0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        if (device is not None) and (device.type != 'cpu'):
            state_dict = torch.load(filename, map_location=device)
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.sz,
                          self.mkldnn,
                          self.conv_3D,
                          self.diam_mean)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
            
        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)
