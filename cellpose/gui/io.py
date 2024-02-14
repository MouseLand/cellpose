"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, datetime, gc, warnings, glob, shutil, copy
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging
import fastremap 

from ..io import imread, imsave, outlines_to_text, add_model, remove_model, save_rois
from ..models import normalize_default, MODEL_DIR, MODEL_LIST_PATH, get_user_models
from ..utils import masks_to_outlines, outlines_list 

try:
    import qtpy
    from qtpy.QtWidgets import QFileDialog
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False
  
def _init_model_list(parent):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parent.model_list_path = MODEL_LIST_PATH
    parent.model_strings = get_user_models()
    
def _add_model(parent, filename=None, load_model=True):
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Add model to GUI"
            )
        filename = name[0]
    add_model(filename)
    fname = os.path.split(filename)[-1]
    parent.ModelChooseC.addItems([fname])
    parent.model_strings.append(fname)
    
    for ind, model_string in enumerate(parent.model_strings[:-1]):
        if model_string == fname:
            _remove_model(parent, ind=ind+1, verbose=False)

    parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))    
    if load_model:
        parent.model_choose(custom=True)

def _remove_model(parent, ind=None, verbose=True):
    if ind is None:
        ind = parent.ModelChooseC.currentIndex()
    if ind > 0:
        ind -= 1
        parent.ModelChooseC.removeItem(ind+1)
        del parent.model_strings[ind]
        # remove model from txt path
        modelstr = parent.ModelChooseC.currentText()
        remove_model(modelstr)
        if len(parent.model_strings) > 0:
            parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))
        else:
            parent.ModelChooseC.setCurrentIndex(0)
    else:
        print('ERROR: no model selected to delete')

def _get_train_set(image_names):
    """ get training data and labels for images in current folder image_names"""
    train_data, train_labels, train_files = [], [], []
    restore = None 
    normalize_params = normalize_default
    for image_name_full in image_names:
        image_name = os.path.splitext(image_name_full)[0]
        label_name = None
        if os.path.exists(image_name + '_seg.npy'):
            dat = np.load(image_name + '_seg.npy', allow_pickle=True).item()
            masks = dat['masks'].squeeze()
            if masks.ndim==2:
                fastremap.renumber(masks, in_place=True)
                label_name = image_name + '_seg.npy'
            else:
                print(f'GUI_INFO: _seg.npy found for {image_name} but masks.ndim!=2')
            if "img_restore" in dat:
                data = dat["img_restore"].squeeze()
                restore = dat["restore"]
            else:
                data = imread(image_name_full)
            normalize_params = dat["normalize_params"] if "normalize_params" in dat else normalize_default
        if label_name is not None:
            train_files.append(image_name_full)
            train_data.append(data)
            train_labels.append(masks)
    if restore:
        print(f"GUI_INFO: using {restore} images (dat['img_restore'])")
    return train_data, train_labels, train_files, restore, normalize_params

def _load_image(parent, filename=None, load_seg=True, load_3D=False):
    """ load image with filename; if None, open QFileDialog """
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load image"
            )
        filename = name[0]
    manual_file = os.path.splitext(filename)[0]+'_seg.npy'
    load_mask = False
    if load_seg:
        if os.path.isfile(manual_file) and not parent.autoloadMasks.isChecked():
            _load_seg(parent, manual_file, image=imread(filename), image_file=filename, load_3D=load_3D)
            return
        elif parent.autoloadMasks.isChecked():
            mask_file = os.path.splitext(filename)[0]+'_masks'+os.path.splitext(filename)[-1]
            mask_file = os.path.splitext(filename)[0]+'_masks.tif' if not os.path.isfile(mask_file) else mask_file
            load_mask = True if os.path.isfile(mask_file) else False
    try:
        print(f'GUI_INFO: loading image: {filename}')
        image = imread(filename)
        parent.loaded = True
    except Exception as e:
        print('ERROR: images not compatible')
        print(f'ERROR: {e}')

    if parent.loaded:
        parent.reset()
        parent.filename = filename
        filename = os.path.split(parent.filename)[-1]
        _initialize_images(parent, image, load_3D=load_3D)
        parent.loaded = True
        parent.enable_buttons()
        if load_mask:
            _load_masks(parent, filename=mask_file)

            
def _initialize_images(parent, image, load_3D=False):
    """ format image for GUI """
    parent.nchan = 3
    if image.ndim > 4:
        image = image.squeeze()
        if image.ndim > 4:
            raise ValueError("cannot load 4D stack, reduce dimensions")
    elif image.ndim==1:
        raise ValueError("cannot load 1D stack, increase dimensions")

    if image.ndim==4:
        if not load_3D:
            raise ValueError("cannot load 3D stack, run 'python -m cellpose --Zstack' for 3D GUI")
        else:
            # make tiff Z x channels x W x H
            if image.shape[0] < 4:
                # tiff is channels x Z x W x H
                image = image.transpose((1,2,3,0))
            image = np.transpose(image, (0,2,3,1))
    elif image.ndim==3:
        if not load_3D:
            # assume smallest dimension is channels and put last
            c = np.array(image.shape).argmin()
            image = image.transpose(((c+1)%3,(c+2)%3,c))
        elif load_3D:
            # assume smallest dimension is Z and put first
            z = np.array(image.shape).argmin()
            image = image.transpose((z, (z+1)%3,(z+2)%3))
            image = image[..., np.newaxis]
    elif image.ndim==2:
        if not load_3D:
            image = image[...,np.newaxis]
        else:
            raise ValueError("cannot load 2D stack in 3D mode, run 'python -m cellpose' for 2D GUI")

    if image.shape[-1] > 3:
        print("WARNING: image has more than 3 channels, keeping only first 3")
        image = image[...,:3]
    elif image.shape[-1]==2:
        # fill in with blank channels to make 3 channels
        shape = image.shape
        image = np.concatenate((image,
                    np.zeros((*shape[:-1], 3-shape[-1]), dtype=np.uint8)), axis=-1)
        parent.nchan = 2
    elif image.shape[-1] == 1:
        parent.nchan = 1
    
    parent.stack = image
    if load_3D:
        parent.NZ = len(parent.stack)
        parent.scroll.setMaximum(parent.NZ-1)
    else:
        parent.NZ = 1
        parent.stack = parent.stack[np.newaxis,...]
    
    img_min = image.min() 
    img_max = image.max()
    parent.stack = parent.stack.astype(np.float32)
    parent.stack -= img_min
    if img_max > img_min + 1e-3:
        parent.stack /= (img_max - img_min)
    parent.stack *= 255
    
    if load_3D:
        print('GUI_INFO: converted to float and normalized values to 0.0->255.0')
    
    del image
    gc.collect()

    parent.imask=0
    parent.Ly, parent.Lx = parent.stack.shape[-3:-1]
    parent.Ly0, parent.Lx0 = parent.stack.shape[-3:-1]
    parent.layerz = 255 * np.ones((parent.Ly,parent.Lx,4), 'uint8')
    if hasattr(parent, "stack_filtered"):
        parent.Lyr, parent.Lxr = parent.stack_filtered.shape[-3:-1]
    elif parent.restore and "upsample" in parent.restore:
        parent.Lyr, parent.Lxr = int(parent.Ly*parent.ratio), int(parent.Lx*parent.ratio)
    else:
        parent.Lyr, parent.Lxr = parent.Ly, parent.Lx 
    parent.clear_all()

    if not hasattr(parent, "stack_filtered") and parent.restore:
        print("GUI_INFO: no 'img_restore' found, applying current settings")
        parent.compute_restore()
    
    if parent.autobtn.isChecked():
        if parent.restore is None:
            print('GUI_INFO: normalization checked: computing saturation levels (and optionally filtered image)')
            parent.compute_saturation()
    elif len(parent.saturation) != parent.NZ:
        parent.saturation = []
        for r in range(3):
            parent.saturation.append([])
            for n in range(parent.NZ):
                parent.saturation[-1].append([0, 255])
            parent.sliders[r].setValue([0, 255])
    parent.compute_scale()
    parent.track_changes = []

    if load_3D:
        parent.currentZ = int(np.floor(parent.NZ/2))
        parent.scroll.setValue(parent.currentZ)
        parent.zpos.setText(str(parent.currentZ))
    else:
        parent.currentZ = 0

    
        

def _load_seg(parent, filename=None, image=None, image_file=None, load_3D=False):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load labelled data", filter="*.npy"
            )
        filename = name[0]
    try:
        dat = np.load(filename, allow_pickle=True).item()
        # check if there are keys in filename
        dat['outlines']
        parent.loaded = True
    except:
        parent.loaded = False
        print('ERROR: not NPY')
        return

    parent.reset()
    if image is None:
        found_image = False
        if 'filename' in dat:
            parent.filename = dat['filename']
            if os.path.isfile(parent.filename):
                parent.filename = dat['filename']
                found_image = True
            else:
                imgname = os.path.split(parent.filename)[1]
                root = os.path.split(filename)[0]
                parent.filename = root+'/'+imgname
                if os.path.isfile(parent.filename):
                    found_image = True
        if found_image:
            try:
                print(parent.filename)
                image = imread(parent.filename)
            except:
                parent.loaded = False
                found_image = False
                print('ERROR: cannot find image file, loading from npy')
        if not found_image:
            parent.filename = filename[:-8]
            print(parent.filename)
            if 'img' in dat:
                image = dat['img']
            else:
                print('ERROR: no image file found and no image in npy')
                return
    else:
        parent.filename = image_file
    
    if "normalize_params" in dat:
        parent.restore = None if "restore" not in dat else dat["restore"]
        print(f"GUI_INFO: restore: {parent.restore}")
        parent.set_normalize_params(dat["normalize_params"])
        parent.set_restore_button()
    
    if 'img_restore' in dat:
        parent.stack_filtered = dat["img_restore"]
        parent.restore = dat["restore"]
        parent.ViewDropDown.model().item(3).setEnabled(True)
        parent.view = 3
        if parent.restore and "upsample" in parent.restore:
            print(parent.stack_filtered.shape, image.shape)
            parent.ratio = dat["ratio"]
    
    _initialize_images(parent, image, load_3D=load_3D)
    if 'chan_choose' in dat:
        parent.ChannelChoose[0].setCurrentIndex(dat['chan_choose'][0])
        parent.ChannelChoose[1].setCurrentIndex(dat['chan_choose'][1])

    if 'outlines' in dat:
        if isinstance(dat['outlines'], list):
            # old way of saving files
            dat['outlines'] = dat['outlines'][::-1]
            for k, outline in enumerate(dat['outlines']):
                if 'colors' in dat:
                    color = dat['colors'][k]
                else:
                    col_rand = np.random.randint(1000)
                    color = parent.colormap[col_rand,:3]
                median = parent.add_mask(points=outline, color=color)
                if median is not None:
                    parent.cellcolors = np.append(parent.cellcolors, color[np.newaxis,:], axis=0)
                    parent.ncells+=1
        else:
            if dat['masks'].min()==-1:
                dat['masks'] += 1
                dat['outlines'] += 1
            parent.ncells = dat['masks'].max()
            if 'colors' in dat and len(dat['colors'])==dat['masks'].max():
                colors = dat['colors']
            else:
                colors = parent.colormap[:parent.ncells,:3]

            _masks_to_gui(parent, dat["masks"], outlines=dat["outlines"], colors=colors)

            parent.draw_layer()
            if 'est_diam' in dat:
                parent.Diameter.setText('%0.1f'%dat['est_diam'])
                parent.diameter = dat['est_diam']
                parent.compute_scale()

        if 'manual_changes' in dat: 
            parent.track_changes = dat['manual_changes']
            print('GUI_INFO: loaded in previous changes')    
        if 'zdraw' in dat:
            parent.zdraw = dat['zdraw']
        else:
            parent.zdraw = [None for n in range(parent.ncells)]
        parent.loaded = True
        #print(f'GUI_INFO: {parent.ncells} masks found in {filename}')
    else:
        parent.clear_all()

    parent.ismanual = np.zeros(parent.ncells, bool)
    if 'ismanual' in dat:
        if len(dat['ismanual']) == parent.ncells:
            parent.ismanual = dat['ismanual']

    if 'current_channel' in dat:
        parent.color = (dat['current_channel']+2)%5
        parent.RGBDropDown.setCurrentIndex(parent.color)

    if 'flows' in dat:
        parent.flows = dat['flows']
        try:
            if parent.flows[0].shape[-3]!=dat['masks'].shape[-2]:
                Ly, Lx = dat['masks'].shape[-2:]
                parent.flows[0] = cv2.resize(parent.flows[0].squeeze(), (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...]
                parent.flows[1] = cv2.resize(parent.flows[1].squeeze(), (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...]
            if parent.NZ==1:
                parent.recompute_masks = True
            else:
                parent.recompute_masks = False
                
        except:
            try:
                if len(parent.flows[0])>0:
                    parent.flows = parent.flows[0]
            except:
                parent.flows = [[],[],[],[],[[]]]
            parent.recompute_masks = False

    parent.enable_buttons()
    parent.update_layer()
    del dat
    gc.collect()

def _load_masks(parent, filename=None):
    """ load zeros-based masks (0=no cell, 1=cell 1, ...) """
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load masks (PNG or TIFF)"
            )
        filename = name[0]
    print(f'GUI_INFO: loading masks: {filename}')
    masks = imread(filename)
    outlines = None
    if masks.ndim>3:
        # Z x nchannels x Ly x Lx
        if masks.shape[-1]>5:
            parent.flows = list(np.transpose(masks[:,:,:,2:], (3,0,1,2)))
            outlines = masks[...,1]
            masks = masks[...,0]
        else:
            parent.flows = list(np.transpose(masks[:,:,:,1:], (3,0,1,2)))
            masks = masks[...,0]
    elif masks.ndim==3:
        if masks.shape[-1]<5:
            masks = masks[np.newaxis,:,:,0]
    elif masks.ndim<3:
        masks = masks[np.newaxis,:,:]
    # masks should be Z x Ly x Lx
    if masks.shape[0]!=parent.NZ:
        print('ERROR: masks are not same depth (number of planes) as image stack')
        return

    _masks_to_gui(parent, masks, outlines)
    if parent.ncells>0:
        parent.draw_layer()
        parent.toggle_mask_ops()
    del masks 
    gc.collect()
    parent.update_layer()
    parent.update_plot()

def _masks_to_gui(parent, masks, outlines=None, colors=None):
    """ masks loaded into GUI """
    # get unique values
    shape = masks.shape
    masks = masks.flatten()
    fastremap.renumber(masks, in_place=True)
    masks = masks.reshape(shape)
    masks = masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32)
    if parent.restore and "upsample" in parent.restore:
        parent.cellpix_resize = masks.copy()
        parent.cellpix = parent.cellpix_resize.copy()
        parent.cellpix_orig = cv2.resize(masks.squeeze(), (parent.Lx0, parent.Ly0), 
                                         interpolation=cv2.INTER_NEAREST)[np.newaxis,:,:]
        parent.resize = True
    else:
        parent.cellpix = masks
    if parent.cellpix.ndim == 2:
        parent.cellpix = parent.cellpix[np.newaxis,:,:]
        if parent.restore and "upsample" in parent.restore:
            if parent.cellpix_resize.ndim==2:
                parent.cellpix_resize = parent.cellpix_resize[np.newaxis, :, :]
            if parent.cellpix_orig.ndim==2:
                parent.cellpix_orig = parent.cellpix_orig[np.newaxis, :, :]
            
    print(f'GUI_INFO: {masks.max()} masks found')

    # get outlines
    if outlines is None: # parent.outlinesOn
        parent.outpix = np.zeros_like(parent.cellpix)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
        for z in range(parent.NZ):
            outlines = masks_to_outlines(parent.cellpix[z])
            parent.outpix[z] = outlines * parent.cellpix[z]
            if parent.restore and "upsample" in parent.restore:
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
            if z%50==0 and parent.NZ > 1:
                print('GUI_INFO: plane %d outlines processed'%z)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy() 
    else:
        parent.outpix = outlines
        shape = parent.outpix.shape
        fastremap.renumber(parent.outpix, in_place=True)
        parent.outpix = np.reshape(parent.outpix, shape)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy()
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
            for z in range(parent.NZ):
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
                if z%50==0 and parent.NZ > 1:
                    print('GUI_INFO: plane %d outlines processed'%z)
        
    if parent.outpix.ndim==2:
        parent.outpix = parent.outpix[np.newaxis,:,:]
        if parent.restore and "upsample" in parent.restore:
            if parent.outpix_resize.ndim==2:
                parent.outpix_resize = parent.outpix_resize[np.newaxis, :, :]
            if parent.outpix_orig.ndim==2:
                parent.outpix_orig = parent.outpix_orig[np.newaxis, :, :]

    parent.ncells = parent.cellpix.max()
    colors = parent.colormap[:parent.ncells, :3] if colors is None else colors
    print('GUI_INFO: creating cellcolors and drawing masks')
    parent.cellcolors = np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8)
    if parent.ncells>0:
        parent.draw_layer()
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells, bool)
    parent.zdraw = list(-1*np.ones(parent.ncells, np.int16))
    
    if hasattr(parent, "stack_filtered"):
        parent.ViewDropDown.setCurrentIndex(3)
        print("set denoised/filtered view")
    else:
        parent.ViewDropDown.setCurrentIndex(0)

def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        if parent.cellpix[0].max() > 65534:
            print('GUI_INFO: saving 2D masks to tif (too many masks for PNG)')
            imsave(base + '_cp_masks.tif', parent.cellpix[0])
        else:
            print('GUI_INFO: saving 2D masks to png')
            imsave(base + '_cp_masks.png', parent.cellpix[0].astype(np.uint16))
    else:
        print('GUI_INFO: saving 3D masks to tiff')
        imsave(base + '_cp_masks.tif', parent.cellpix)

def _save_flows(parent):
    """ save flows and cellprob to tiff """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if len(parent.flows) > 0:
        imsave(base + '_cp_flows.tif', parent.flows[4][:-1])
        imsave(base + '_cp_cellprob.tif', parent.flows[4][-1])

def _save_rois(parent):
    """ save masks as rois in .zip file for ImageJ """
    filename = parent.filename
    if parent.NZ == 1:
        print(f'GUI_INFO: saving {parent.cellpix[0].max()} ImageJ ROIs to .zip archive.')
        save_rois(parent.cellpix[0], parent.filename)
    else:
        print('ERROR: cannot save 3D outlines')

def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('GUI_INFO: saving 2D outlines to text file, see docs for info to load into ImageJ')    
        outlines = outlines_list(parent.cellpix[0])
        outlines_to_text(base, outlines)
    else:
        print('ERROR: cannot save 3D outlines')
    
def _save_sets_with_check(parent):
    """ Save masks and update *_seg.npy file. Use this function when saving should be optional
     based on the disableAutosave checkbox. Otherwise, use _save_sets """
    if not parent.disableAutosave.isChecked():
        _save_sets(parent)


def _save_sets(parent):
    """ save masks to *_seg.npy. This function should be used when saving
    is forced, e.g. when clicking the save button. Otherwise, use _save_sets_with_check
    """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    flow_threshold, cellprob_threshold = parent.get_thresholds()
    if parent.NZ > 1 and parent.is_stack:
        np.save(base + '_seg.npy',
                {'outlines': parent.outpix,
                 'colors': parent.cellcolors[1:],
                 'masks': parent.cellpix,
                 'current_channel': (parent.color-2)%5,
                 'filename': parent.filename,
                 'flows': parent.flows,
                 'zdraw': parent.zdraw,
                 'model_path': parent.current_model_path if hasattr(parent, 'current_model_path') else 0,
                 'flow_threshold': flow_threshold,
                 'cellprob_threshold': cellprob_threshold
                 })
    else:
        dat = {'outlines': parent.outpix.squeeze() if parent.restore is None or not "upsample" in parent.restore else parent.outpix_resize.squeeze(),
                'colors': parent.cellcolors[1:],
                'masks': parent.cellpix.squeeze() if parent.restore is None or not "upsample" in parent.restore else parent.cellpix_resize.squeeze(),
                'chan_choose': [parent.ChannelChoose[0].currentIndex(),
                                parent.ChannelChoose[1].currentIndex()],
                'filename': parent.filename,
                'flows': parent.flows,
                'ismanual': parent.ismanual,
                'manual_changes': parent.track_changes,
                'model_path': parent.current_model_path if hasattr(parent, 'current_model_path') else 0,
                'flow_threshold': flow_threshold,
                'cellprob_threshold': cellprob_threshold,
                'normalize_params': parent.get_normalize_params(),
                'restore': parent.restore,
                'ratio': parent.ratio,
                'diameter': parent.diameter}
        print(dat["masks"].shape)
        if parent.restore is not None:
            dat["img_restore"] = parent.stack_filtered
        np.save(base + '_seg.npy',
                dat)
        del dat
    #print(parent.point_sets)
    print('GUI_INFO: %d ROIs saved to %s'%(parent.ncells, base + '_seg.npy'))