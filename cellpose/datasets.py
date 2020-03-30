import sys
import numpy as np
import os
import cv2
import pickle

def diameters(masks):
    unique, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    return md, counts**0.5

def load_pickle(filename, filecell, cyto=True, specialist=False):        
    with open(filename, 'rb') as pickle_in:
        V=pickle.load(pickle_in)

    np.random.seed(101)
    r = np.random.rand(10000)
        
    if cyto:
        r[610] = 0. # put one of kenneth's images in test
        iscell = np.load(filecell)
        ls = [1,2,3,4,5,0]
        if specialist:
            ls = [1]
            iscell[140:] = -1
        print(ls)
        vf=[]
        type_cell=[]
        for k in range(len(ls)):
            irange = (iscell==ls[k]).nonzero()[0]
            type_cell.extend(list(np.ones(len(irange))*ls[k]))
            vf.extend([V[k] for k in irange])
        diam_mean = 27.
        print(len(vf))
    else:
        iscell = np.load(filecell)
        ls = [1,2,3,4]
        vf = []
        type_cell = []
        for k in range(len(ls)):
            irange = (iscell==ls[k]).nonzero()[0]
            type_cell.extend(list(np.ones(len(irange))*ls[k]))
            vf.extend([V[k] for k in irange])
        diam_mean = 15.
        
    type_cell = np.array(type_cell)
    train_cell = type_cell[r[:len(type_cell)]>.1]
    test_cell = type_cell[r[:len(type_cell)]<.1]
    print(train_cell.shape)

    vft = [vf[k] for k in (r[:len(vf)]<.1).nonzero()[0]]
    vf  = [vf[k] for k in (r[:len(vf)]>.1).nonzero()[0]]
    print(len(vft), len(vf))

    train_data = []
    train_labels = []
    train_flows = []
    nimg = len(vf)
    tcell = []
    for n in range(nimg):
        img = vf[n][0][:1]
        if cyto:
            if len(vf[n][1])>0:
                img = np.concatenate((img, vf[n][1][np.newaxis,...]), axis=0)
            else:
                img = np.concatenate((img, np.zeros_like(img)), axis=0)
        diam = diameters(vf[n][2])[0]
        if diam>0:
            train_data.append(img)
            if cyto:
                train_labels.append(vf[n][2][np.newaxis,...]+1)
            else:
                train_labels.append(vf[n][2][np.newaxis,...])
            train_flows.append(vf[n][0][[3,1,2]])
            tcell.append(train_cell[n])
    train_cell = np.array(tcell)
    test_data = []
    test_labels = []
    test_flows = []
    for n in range(len(vft)):
        img = vft[n][0][:1]
        if cyto:
            if len(vft[n][1])>0:
                img = np.concatenate((img, vft[n][1][np.newaxis,...]), axis=0)
            else:
                img = np.concatenate((img, np.zeros_like(img)), axis=0)
        test_data.append(img)
        test_labels.append(vft[n][2][np.newaxis,...])
        test_flows.append(vft[n][0][[3,1,2]])

    
    print(r[0], len(train_data), len(vft))

    return train_data, train_labels, train_flows, train_cell, test_data, test_labels, test_flows, test_cell
