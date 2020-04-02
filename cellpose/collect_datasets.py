from glob import glob
from tifffile import imread
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from matplotlib import pyplot as plt
from mxnet import gpu, cpu
import time, os
import cv2
from glob import glob
import sys

import cellpose
from cellpose import utils, datasets

def normalize99(img):
    X = img.copy()
    if img.ndim>2:
        for j in range(X.shape[0]):
            X[j] = normalize99(X[j])
    else:
        X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def make():
    V = []
    N = 0

    fmask = glob('H:/DATA/cellpose/proc/micronet/*mask.tif')
    Y = list(map(imread, fmask))
    fmask = glob('H:/DATA/cellpose/proc/micronet/img???.tif')
    X = list(map(imread, fmask))

    for k in range(len(X)):
        img = X[k].astype('float32')
        mask = Y[k].astype('int32')
        N = N + len(np.unique(mask))
        img = normalize99(img)
        V0 = datasets.img_to_flow(img[1], mask, img[0])
        V0 = np.transpose(V0, (2, 0, 1))
        V.append([V0, img[0], mask])
        if k%10==1:
            print(k)
    print(N)

    fmask = glob('H:/DATA/cellpose/proc/neurites/*mask.tif')
    Y = list(map(imread, fmask))
    fimg = glob('H:/DATA/cellpose/proc/neurites/img???.tif')
    X = list(map(imread, fimg))
    for k in range(len(X)):
        img = X[k].astype('float32')
        mask = Y[k].astype('int32')
        N = N + len(np.unique(mask))
        img = normalize99(img)
        V0 = datasets.img_to_flow(img[1], mask, img[0])
        V0 = np.transpose(V0, (2, 0, 1))
        V.append([V0, img[0], mask])
        if k%10==1:
            print(k)
    print(N)


    fnpy = glob('H:/DATA/cellpose/proc/BBBC007/*npy')
    fnpy.extend(glob('H:/DATA/cellpose/proc/BBBC020/*npy'))
    new_X = []
    for j in range(len(fnpy)):
        new_X.append(np.load(fnpy[j], allow_pickle=True).item())
        if 'img' not in new_X[j]:
            new_X[j]['img'] = imread(new_X[j]['filename'])
        if new_X[j]['img'].shape[-1]<5:
            new_X[j]['img'] = np.transpose(new_X[j]['img'], (2, 0, 1))
    for k in range(len(new_X)):
        img = new_X[k]['img'].astype('float32')
        ix = np.array(new_X[k]['mask_types'])=='cytoplasm'
        outline = [new_X[k]['outlines'][j] for j in ix.nonzero()[0]]
        mask = datasets.outlines_to_masks(outline, img.shape[-2:])
        V0 = []
        img = normalize99(img)
        V0 = datasets.img_to_flow(img[1], mask, img[0])
        V0 = np.transpose(V0, (2, 0, 1))
        N = N + len(np.unique(mask))
        V.append([V0, img[0], mask])
    print(N)


    flist = glob('H:/DATA/cellpose/proc/C2DL/*mask.tif')
    Y = list(map(imread, flist))
    flist = glob('H:/DATA/cellpose/proc/C2DL/img???.tif')
    X = list(map(imread, flist))
    for k in range(len(X)):
        img = X[k].astype('float32')
        mask = Y[k].astype('int32')
        N = N + len(np.unique(mask))
        img = normalize99(img)
        V0 = datasets.img_to_flow(img, mask)
        V0 = np.transpose(V0, (2, 0, 1))
        V.append([V0, [], mask])
        if k%10==1:
            print(k)
    print(len(V))

    fnpy = glob('Z:/datasets/segmentation/gcamp/*npy')
    new_X = []
    for j in range(len(fnpy)):
        new_X.append(np.load(fnpy[j], allow_pickle=True).item())
        if 'img' not in new_X[j]:
            new_X[j]['img'] = imread(new_X[j]['filename'])
        if type(new_X[j]['img']) is list:
            img = new_X[j]['img']
            I = np.float32(img[0])
            for t in range(1,len(img)):
                 I += np.float32(img[t])
            new_X[j]['img'] = I
        img = np.float32(new_X[j]['img'])
        if img.ndim>2:
            if img.shape[-1]<5:
                img = np.mean(img, axis=-1)
            else:
                img = np.mean(img, axis=0)
        new_X[j]['img'] = img
        img = new_X[j]['img'].astype('float32')
        outline = new_X[j]['outlines']
        masks = datasets.outlines_to_masks(outline, img.shape[-2:])
        img =  normalize99(img)
        V0 = datasets.img_to_flow(img, masks)
        V0 = np.concatenate((V0, np.expand_dims(masks,-1)), axis=-1)
        mask = cv2.resize(V0[:,:,-1], (1024, 1024), interpolation=cv2.INTER_NEAREST)
        V0 = cv2.resize(V0, (1024, 1024))
        V0[:,:,-1] = mask
        V0 = np.transpose(V0, (2, 0, 1))
        V0 = np.reshape(V0, (-1, 2, 512, 2,512))
        V0 = np.transpose(V0, (1,3,0,2,4))
        V0 = np.reshape(V0, (4, -1, 512, 512))
        V0[:,-1,:,:] = np.round(V0[:,-1,:,:])
        for t in range(4):
            V0[t][0] = normalize99(V0[t][0])
            V.append([V0[t][:-1], [], V0[t][-1]])

    fnpy = []
    fnpy = glob('Z:/datasets/segmentation/tim/*npy')
    N = 0
    new_X = []
    for j in range(len(fnpy)):
        new_X.append(np.load(fnpy[j], allow_pickle=True).item())
        img = imread(new_X[j]['filename'])
        img = np.float32(img)
        img = normalize99(img)
        outline = new_X[j]['outlines']
        masks = datasets.outlines_to_masks(outline, img.shape[-2:])
        V0 = datasets.img_to_flow(img[1], masks, img[0])
        V0 = np.transpose(V0, (2, 0, 1))
        V.append([V0, img[0], masks])
        N = N + len(new_X[j]['outlines'])
    print(N)


    fnpy = []
    fnpy.extend(glob('Z:\datasets\segmentation\week1/*npy'))
    fnpy.extend(glob('Z:\datasets\segmentation\week2/*npy'))
    fnpy.extend(glob('Z:\datasets\segmentation\week3/*npy'))
    fnpy.extend(glob('Z:\datasets\segmentation\week4/*npy'))
    N = 0
    new_X = []
    for j in range(len(fnpy)):
        new_X.append(np.load(fnpy[j], allow_pickle = True).item())
        #if 'img' not in new_X[j]:
         #   new_X[j]['img'] = imread(new_X[j]['filename'])
        if type(new_X[j]['img']) is list:
            print('image is list %d'%j)
            img = new_X[j]['img']
            I = np.float32(img[0])
            for t in range(1,len(img)):
                 I += np.float32(img[t])
            new_X[j]['img'] = I
        img = np.float32(new_X[j]['img'])
        if img.shape[0]==3 or img.shape[-1]==3:
            if img.shape[-1]<5:
                img = np.mean(img, axis=-1)
                print('image is 3-chan %d'%j)
            else:
                img = np.mean(img, axis=0)
                print('image is 3-chan, axis last %d'%j)
        img = img.astype('float32')
        if img.ndim<3:
            img  =np.expand_dims(img, 0)
        if 'masks' in new_X[j]:
            N = N + np.max(new_X[j]['masks'][0])
            masks = new_X[j]['masks'][0]
            print('masks found %d'%j)
        else:
            N = N + len(new_X[j]['outlines'])
            outline = new_X[j]['outlines']
            masks = datasets.outlines_to_masks(outline, img.shape[-2:])
        img =  normalize99(img)
        if img.shape[0]>1:
            V0 = datasets.img_to_flow(img[0], masks, img[1])
        else:
            V0 = datasets.img_to_flow(img[0], masks)
        V0 = np.transpose(V0, (2, 0, 1))
        if img.shape[0]>1:
            V.append([V0, img[1], masks])
            print(j)
        else:
            V.append([V0, [], masks])
    print(N)

    return V
