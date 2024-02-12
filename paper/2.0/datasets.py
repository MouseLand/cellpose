"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, argparse
from tifffile import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from cellpose import models
from cellpose.io import logger_setup
from cellpose.transforms import normalize_img
from cellpose import metrics
from tqdm import tqdm, trange
from natsort import natsorted

### FUNCTIONS FOR USING ALL DATASETS TOGETHER -------------- ###

def reshape_and_normalize(img, img_type):
    if img_type[0] == 'tn':
        img = img[::-1]
    if img_type[0] == 'cp':
        if img_type[1] == 'nuclei':
            img = np.stack((img[0], np.zeros_like(img[0])), axis=0)
        elif img_type[1] == 'cyto':
            img = img[:,:,[1,0]].transpose(2,0,1)   
    elif img.ndim < 3:
        img = np.stack((img, np.zeros_like(img)), axis=0)
    img = normalize_img(img, axis=0)
    return img

def get_all_files(root):
    train_files_all = []
    train_types_all = []
    test_files_all = []
    test_types_all = []

    dsets = ['cellpose_cyto_dataset',
             'livecell_dataset',
             'tissuenet_dataset']
    ctype = ['cyto']
    ext = ['.png', '.tif', '.tif']
    tfs = ['', 'lc', 'tn']
    train_str = ['train', 'train', 'train_full']
    for k, dset in enumerate(dsets):
        train_files = glob(os.path.join(root+dset, f'{train_str[k]}/*{ext[k]}'))
        train_files = [tf for tf in train_files if tf[-10:]!=f'_masks.png' and tf[-10:]!=f'_masks.tif' and tf[-10:]!='_flows.tif']
        train_files = natsorted(train_files)
        if k > 0:
            train_types = np.array([[tfs[k], *os.path.split(tf)[-1].split('_')[:2]] for tf in train_files], dtype=object)
        else:
            train_types = np.array([['cp', ctype[k]] for i in range(len(train_files))], dtype=object)
        train_files_all.extend(train_files)
        train_types_all.extend(train_types)

        test_files = glob(os.path.join(root+dset, f'test/*{ext[k]}'))
        test_files = [tf for tf in test_files if tf[-10:]!=f'_masks.png' and tf[-10:]!=f'_masks.tif' and tf[-10:]!='_flows.tif']
            
        test_files = natsorted(test_files)
        if k>0:
            test_types = np.array([[tfs[k], *os.path.split(tf)[-1].split('_')[:2]] for tf in test_files], dtype=object)
        else:
            test_types = np.array([['cp', ctype[k]] for i in range(len(test_files))], dtype=object)
        test_files_all.extend(test_files)
        test_types_all.extend(test_types)
        
    train_files = train_files_all
    train_types = train_types_all
    test_files = test_files_all
    test_types = test_types_all

    return train_files, train_types, test_files, test_types

def load_train_test_all(train_files, train_types, test_files, test_types):
    """ reshape train and test data with cyto chan 0 and nuclei chan 1 """
    from cellpose.io import imread
    train_data = [reshape_and_normalize(imread(train_files[i]), train_types[i]) 
                  for i in trange(len(train_files))]
    test_data = [reshape_and_normalize(imread(test_files[i]), test_types[i]) 
                  for i in trange(len(test_files))]
    return train_data, test_data


### FUNCTIONS FOR GETTING TISSUENET FILES + SUBSETS --------------- ###

def get_tissuenet_val(root, tissue_type='ALL', platform_type='ALL', 
                        nval=8, seed=1):    
    dat = np.load(root + 'npz/tissuenet_v1.0_val.npz')
    data = dat['X']
    labels = dat['y']
    if tissue_type != 'ALL':
        ix = (np.logical_and(dat['tissue_list']==tissue_type, dat['platform_list']==platform_type)).nonzero()[0]
    X_val, y_val = data[ix], labels[ix]
    return X_val[:nval], y_val[:nval, :, :, [0]]


def get_tissuenet_train(root, tissue_type='ALL', platform_type='ALL', 
                        ntrain=10, full_img=False, seed=1):
    train_files = glob(os.path.join(root, 'train_full/*.tif'))
    train_files = [tf for tf in train_files if tf[-10:]!='_masks.tif' and tf[-10:]!='_flows.tif']
    train_types = np.array([os.path.split(tf)[-1].split('_')[0] for tf in train_files])
    train_platforms = np.array([os.path.split(tf)[-1].split('_')[1] for tf in train_files])
    # get indices of files    
    if tissue_type=='ALL':
        ix_train = np.arange(0, len(train_files))
    else:
        ix_train = (train_types==tissue_type) * (train_platforms==platform_type)
        ix_train = ix_train.nonzero()[0]

    if ntrain is not None and ntrain > 0:
        np.random.seed(seed)
        iperm_train = np.random.permutation(len(ix_train))        
        itrain  = ix_train[iperm_train[:int(np.ceil(ntrain))]]
        train_files_full = [train_files[it] for it in itrain]
        if full_img:
            return train_files_full
        else:
            krand = np.random.randint(4, size=(len(ix_train),))
            train_files = []
            for i, tf in enumerate(train_files_full):
                fname = os.path.splitext(os.path.split(tf)[-1])[0]
                fnewname = os.path.join(root, f'train/{fname}_{krand[i]}.tif')
                train_files.append(fnewname)
    else:
        itrain = ix_train
        train_files_full = [train_files[it] for it in itrain]
        if full_img:
            return train_files_full 
        else:
            train_files = []
            for i, tf in enumerate(train_files_full):
                for k in range(4):
                    fname = os.path.splitext(os.path.split(tf)[-1])[0]
                    fnewname = os.path.join(root, f'train/{fname}_{k}.tif')
                    train_files.append(fnewname)
    return train_files 

def get_tissuenet_test(root, tissue_type='ALL', platform_type='ALL', ntest=0):
    test_files = glob(os.path.join(root, 'test/*.tif'))
    test_files = [tf for tf in test_files if tf[-10:]!='_masks.tif' and tf[-10:]!='_flows.tif']
    test_types = np.array([os.path.split(tf)[-1].split('_')[0] for tf in test_files])
    test_platforms = np.array([os.path.split(tf)[-1].split('_')[1] for tf in test_files])
    if tissue_type=='ALL':
        ix_test = np.arange(0, len(test_files))
    else:
        ix_test = (test_types==tissue_type) * (test_platforms==platform_type)
        ix_test = ix_test.nonzero()[0]

    if ntest is not None and ntest>0:
        np.random.seed(1)
        iperm_test = np.random.permutation(len(ix_test))
        itest   = ix_test[iperm_test[:ntest]]
    else:
        itest = ix_test
    test_files = [test_files[it] for it in itest]
    return test_files

### FUNCTIONS FOR GETTING LIVECELL FILES + SUBSETS --------------- ###

def get_livecell_train(root, cell_type='ALL', ntrain=10, seed=0):
    train_files = glob(os.path.join(root, 'train/*.tif'))
    train_files = [tf for tf in train_files if tf[-10:]!='_masks.tif' and tf[-10:]!='_flows.tif']
    train_types = np.array([os.path.split(tf)[-1].split('_')[0] for tf in train_files])
    
    # get indices of files    
    if cell_type=='ALL':
        ix_train = np.arange(0, len(train_files))
    else:
        ix_train = train_types==cell_type
        ix_train = ix_train.nonzero()[0]
    if ntrain is not None and ntrain>0:
        np.random.seed(seed)
        iperm_train = np.random.permutation(len(ix_train))        
        itrain  = ix_train[iperm_train[:int(np.ceil(ntrain))]]
    else:
        itrain = ix_train
    train_files = [train_files[it] for it in itrain]
    return train_files
    
def get_livecell_test(root, cell_type='ALL', ntest=0):
    test_files = glob(os.path.join(root, 'test/*.tif'))
    test_files = [tf for tf in test_files if tf[-10:]!='_masks.tif' and tf[-10:]!='_flows.tif']
    test_types = np.array([os.path.split(tf)[-1].split('_')[0] for tf in test_files])
    
    if cell_type=='ALL':
        ix_test = np.arange(0, len(test_files))
    else:
        ix_test = test_types==cell_type
        ix_test = ix_test.nonzero()[0]
    if ntest is not None and ntest>0:
        np.random.seed(1)
        iperm_test = np.random.permutation(len(ix_test))
        itest   = ix_test[iperm_test[:ntest]]
    else:
        itest = ix_test
    test_files = [test_files[it] for it in itest]
    return test_files

### FUNCTIONS FOR LIVECELL/TISSUENET LOADING ------------------------ ###
    
def load_data_masks(files, frac=1.0):
    data = [imread(file) for file in files]
    masks = [imread(os.path.splitext(file)[0] + '_masks.tif').astype(np.uint16) for file in files]
    if frac==0.5:
        data = [d[...,:d.shape[-1]//2] for d in data]
        masks = [m[...,:m.shape[-1]//2] for m in masks]
    elif frac==0.25:
        data = [d[...,:d.shape[-2]//2,:d.shape[-1]//2] for d in data]
        masks = [m[...,:m.shape[-2]//2,:m.shape[-1]//2] for m in masks]
    return data, masks

def get_train_files(root, cell_type, tissue_type, platform_type, ntrain=10,
                    full_img=False, seed=1):
    if cell_type is not None:
        train_files = get_livecell_train(root, cell_type, ntrain=ntrain, seed=seed)
        channels = [0,0]
        netstrf = f'_livecell_{cell_type}_ntrain_{ntrain}_seed_{seed}'
    elif tissue_type is not None:
        train_files = get_tissuenet_train(root, tissue_type, platform_type, ntrain=ntrain, 
                                          full_img=full_img, seed=seed)
        channels = [2,1]
        netstrf = f'_tissuenet_{tissue_type}_{platform_type}_ntrain_{ntrain}_seed_{seed}'
    if full_img:
        netstrf += '_FULL'
    return train_files, channels, netstrf

### FUNCTION FOR LIVECELL PREPROCESS ------------------------------ ### 

def remove_overlaps(masks, medians, overlap_threshold=0.75):
    """ replace overlapping mask pixels with mask id of closest mask
        if mask fully within another mask, remove it
        masks = Nmasks x Ly x Lx
    """
    cellpix = masks.sum(axis=0)
    igood = np.ones(masks.shape[0], 'bool')
    for i in masks.sum(axis=(1,2)).argsort():
        npix = float(masks[i].sum())
        noverlap = float(masks[i][cellpix > 1].sum())
        if noverlap / npix >= overlap_threshold:
            igood[i] = False
            cellpix[masks[i]>0] -= 1
            #print(cellpix.min())
    print(f'removing {(~igood).sum()} masks')
    masks = masks[igood]
    medians = medians[igood]
    cellpix = masks.sum(axis=0)
    overlaps = np.array(np.nonzero(cellpix>1.0)).T
    dists = ((overlaps[:,:,np.newaxis] - medians.T)**2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:,0], overlaps[:,1]] = 0
    masks[tocell, overlaps[:,0], overlaps[:,1]] = 1

    # labels should be 1 to mask.shape[0]
    masks = masks.astype(int) * np.arange(1,masks.shape[0]+1,1,int)[:,np.newaxis,np.newaxis]
    masks = masks.sum(axis=0)
    return masks

def ann_to_masks(annotations, anns, overlap_threshold=0.75):
    """ list of coco-format annotations with masks to single image"""
    masks = []
    k=0
    medians=[]
    for ann in anns:
        mask = annotations.annToMask(ann)
        masks.append(mask)
        ypix, xpix = mask.nonzero()
        medians.append(np.array([ypix.mean(), xpix.mean()]))
        k+=1
    masks=np.array(masks).astype('int')
    medians=np.array(medians)
    masks = remove_overlaps(masks, medians, overlap_threshold=overlap_threshold)
    return masks

def livecell_ann_to_masks(img_dir, annotation_file):
    from pycocotools.coco import COCO
    from tifffile import imsave
    img_dir_classes = glob(img_dir + '*/')
    classes = [img_dir_class.split(os.sep)[-2] for img_dir_class in img_dir_classes]
    print(classes)

    train_files = []
    train_class_files = []
    for cclass, img_dir_class in zip(classes, img_dir_classes):
        train_files.extend(glob(img_dir_class + '*.tif'))
        train_class_files.append(glob(img_dir_class + '*.tif'))

    annotations = COCO(annotation_file)
    imgIds = list(annotations.imgs.keys())

    for train_class_file in train_class_files:
        for i in range(len(train_class_file)):
            filename = train_class_file[i]
            fname = os.path.split(filename)[-1]
            loc = np.array([annotations.imgs[imgId]['file_name']==fname for imgId in imgIds]).nonzero()[0]
            if len(loc) > 0:
                imgId = imgIds[loc[0]]
                annIds = annotations.getAnnIds(imgIds=[imgId], iscrowd=None)
                anns = annotations.loadAnns(annIds)
                masks = ann_to_masks(annotations, anns, overlap_threshold=0.75)
                masks = masks.astype(np.uint16)
                maskname = os.path.splitext(filename)[0] + '_masks.tif'
                imsave(maskname, masks)
                print(f'saved masks at {maskname}')
