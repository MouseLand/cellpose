import os, datetime, gc, warnings, glob, shutil, copy
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging
import fastremap 

from .. import utils, plot, transforms, models
from ..io import imread, imsave, outlines_to_text
from ..transforms import normalize99

try:
    from PyQt5.QtWidgets import QFileDialog
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

NCOLOR = False 
# WIP to make GUI use N-color masks. Tricky thing is that only the display should be 
# reduced to N colors; selection and editing should act on unique labels. 
    
def _init_model_list(parent):
    models.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parent.model_list_path = os.fspath(models.MODEL_DIR.joinpath('gui_models.txt'))
    parent.model_strings = []
    if not os.path.exists(parent.model_list_path):
        textfile = open(parent.model_list_path, 'w')
        textfile.close()
    else:
        with open(parent.model_list_path, 'r') as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                parent.model_strings.extend(lines)
    
def _add_model(parent, filename=None, load_model=True):
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Add model to GUI"
            )
        filename = name[0]
    fname = os.path.split(filename)[-1]
    try:
        shutil.copyfile(filename, os.fspath(models.MODEL_DIR.joinpath(fname)))
    except shutil.SameFileError:
        pass
    print(f'GUI_INFO: {filename} copied to models folder {os.fspath(models.MODEL_DIR)}')
    with open(parent.model_list_path, 'a') as textfile:
        textfile.write(fname + '\n')
    parent.ModelChoose.addItems([fname])
    parent.model_strings.append(fname)
    if len(parent.model_strings) > 0:
        parent.ModelButton.setStyleSheet(parent.styleUnpressed)
        parent.ModelButton.setEnabled(True)
    
    for ind, model_string in enumerate(parent.model_strings[:-1]):
        if model_string == fname:
            _remove_model(parent, ind=ind+1, verbose=False)

    parent.ModelChoose.setCurrentIndex(len(parent.model_strings))
    if load_model:
        parent.model_choose(len(parent.model_strings))

def _remove_model(parent, ind=None, verbose=True):
    if ind is None:
        ind = parent.ModelChoose.currentIndex()
    if ind > 0:
        ind -= 1
        if verbose:
            print(f'GUI_INFO: deleting {parent.model_strings[ind]} from GUI')
        parent.ModelChoose.removeItem(ind+1)
        del parent.model_strings[ind]
        custom_strings = parent.model_strings
        if len(custom_strings) > 0:
            with open(parent.model_list_path, 'w') as textfile:
                for fname in custom_strings:
                    textfile.write(fname + '\n')
            parent.ModelChoose.setCurrentIndex(len(parent.model_strings))
        else:
            # write empty file
            textfile = open(parent.model_list_path, 'w')
            textfile.close()
            parent.ModelChoose.setCurrentIndex(0)
            parent.ModelButton.setStyleSheet(parent.styleInactive)
            parent.ModelButton.setEnabled(False)
    else:
        print('ERROR: no model selected to delete')

    

def _get_train_set(image_names):
    """ get training data and labels for images in current folder image_names"""
    train_data, train_labels, train_files = [], [], []
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
        if label_name is not None:
            train_files.append(image_name_full)
            train_data.append(imread(image_name_full))
            train_labels.append(masks)
    return train_data, train_labels, train_files

def _load_image(parent, filename=None, load_seg=True):
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
            _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
            return
        elif os.path.isfile(os.path.splitext(filename)[0]+'_manual.npy'):
            manual_file = os.path.splitext(filename)[0]+'_manual.npy'
            _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
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
        _initialize_images(parent, image, resize=parent.resize, X2=0)
        parent.clear_all()
        parent.loaded = True
        parent.enable_buttons()
        if load_mask:
            _load_masks(parent, filename=mask_file)
            


def _initialize_images(parent, image, resize, X2):
    """ format image for GUI """
    parent.onechan=False
    if image.ndim > 3:
        # make tiff Z x channels x W x H
        if image.shape[0]<4:
            # tiff is channels x Z x W x H
            image = np.transpose(image, (1,0,2,3))
        elif image.shape[-1]<4:
            # tiff is Z x W x H x channels
            image = np.transpose(image, (0,3,1,2))
        # fill in with blank channels to make 3 channels
        if image.shape[1] < 3:
            shape = image.shape
            image = np.concatenate((image,
                            np.zeros((shape[0], 3-shape[1], shape[2], shape[3]), dtype=np.uint8)), axis=1)
            if 3-shape[1]>1:
                parent.onechan=True
        image = np.transpose(image, (0,2,3,1))
    elif image.ndim==3:
        if image.shape[0] < 5:
            image = np.transpose(image, (1,2,0))
        if image.shape[-1] < 3:
            shape = image.shape
            #if parent.autochannelbtn.isChecked():
            #    image = normalize99(image) * 255
            image = np.concatenate((image,np.zeros((shape[0], shape[1], 3-shape[2]),dtype=type(image[0,0,0]))), axis=-1)
            if 3-shape[2]>1:
                parent.onechan=True
            image = image[np.newaxis,...]
        elif image.shape[-1]<5 and image.shape[-1]>2:
            image = image[:,:,:3]
            #if parent.autochannelbtn.isChecked():
            #    image = normalize99(image) * 255
            image = image[np.newaxis,...]
    else:
        image = image[np.newaxis,...]
    
    img_min = image.min() 
    img_max = image.max()
    parent.stack = image
    parent.NZ = len(parent.stack)
    parent.scroll.setMaximum(parent.NZ-1)
    parent.stack = parent.stack.astype(np.float32)
    parent.stack -= img_min
    if img_max > img_min + 1e-3:
        parent.stack /= (img_max - img_min)
    parent.stack *= 255
    if parent.NZ>1:
        print('GUI_INFO: converted to float and normalized values to 0.0->255.0')
    del image
    gc.collect()

    #parent.stack = list(parent.stack)

    if parent.stack.ndim < 4:
        parent.onechan=True
        parent.stack = parent.stack[:,:,:,np.newaxis]
    
    parent.imask=0
    parent.Ly, parent.Lx = parent.stack.shape[1:3]
    parent.layerz = 255 * np.ones((parent.Ly,parent.Lx,4), 'uint8')
    print(parent.layerz.shape)
    if parent.autobtn.isChecked():
        parent.compute_saturation()
    elif len(parent.saturation) != parent.NZ:
        parent.saturation = []
        for n in range(parent.NZ):
            parent.saturation.append([0, 255])
        parent.slider.setLow(0)
        parent.slider.setHigh(255)
    parent.compute_scale()
    parent.currentZ = int(np.floor(parent.NZ/2))
    parent.scroll.setValue(parent.currentZ)
    parent.zpos.setText(str(parent.currentZ))
    parent.track_changes = []

def _load_seg(parent, filename=None, image=None, image_file=None):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load labelled data", filter="*.npy"
            )
        filename = name[0]
    try:
        dat = np.load(filename, allow_pickle=True).item()
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
                image = imread(parent.filename)
            except:
                parent.loaded = False
                found_image = False
                print('ERROR: cannot find image file, loading from npy')
        if not found_image:
            parent.filename = filename[:-11]
            if 'img' in dat:
                image = dat['img']
            else:
                print('ERROR: no image file found and no image in npy')
                return
    else:
        parent.filename = image_file
    
    if 'X2' in dat:
        parent.X2 = dat['X2']
    else:
        parent.X2 = 0
    if 'resize' in dat:
        parent.resize = dat['resize']
    elif 'img' in dat:
        if max(image.shape) > max(dat['img'].shape):
            parent.resize = max(dat['img'].shape)
    else:
        parent.resize = -1
    _initialize_images(parent, image, resize=parent.resize, X2=parent.X2)
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
            if dat['masks'].ndim==2:
                dat['masks'] = dat['masks'][np.newaxis,:,:]
                dat['outlines'] = dat['outlines'][np.newaxis,:,:]
            if dat['masks'].min()==-1:
                dat['masks'] += 1
                dat['outlines'] += 1
            parent.ncells = dat['masks'].max()
            if 'colors' in dat and len(dat['colors'])==dat['masks'].max():
                colors = dat['colors']
            else:
                colors = parent.colormap[:parent.ncells,:3]
            parent.cellpix = dat['masks']
            parent.outpix = dat['outlines']
            parent.cellcolors = np.append(parent.cellcolors, colors, axis=0)
            
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
        print(f'GUI_INFO: {parent.ncells} masks found in {filename}')
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
    del masks 
    gc.collect()
    parent.update_layer()
    parent.update_plot()

def _masks_to_gui(parent, masks, outlines=None):
    """ masks loaded into GUI """
    # get unique values
    shape = masks.shape
    masks = masks.flatten()
    fastremap.renumber(masks, in_place=True)
    masks = masks.reshape(shape)
    masks = masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32)
    parent.cellpix = masks
    if parent.cellpix.ndim == 2:
        parent.cellpix = parent.cellpix[np.newaxis,:,:]
    print(f'GUI_INFO: {masks.max()} masks found')

    # get outlines
    if outlines is None: # parent.outlinesOn
        parent.outpix = np.zeros_like(masks)
        for z in range(parent.NZ):
            outlines = utils.masks_to_outlines(masks[z])
            parent.outpix[z] = outlines * masks[z]
            if z%50==0 and parent.NZ > 1:
                print('GUI_INFO: plane %d outlines processed'%z)
    else:
        parent.outpix = outlines
        shape = parent.outpix.shape
        _,parent.outpix = np.unique(parent.outpix, return_inverse=True)
        parent.outpix = np.reshape(parent.outpix, shape)

    parent.ncells = parent.cellpix.max()
    colors = parent.colormap[:parent.ncells, :3]
    print('GUI_INFO: creating cellcolors and drawing masks')
    parent.cellcolors = np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8)
    parent.draw_layer()
    if parent.ncells>0:
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells, bool)
    parent.zdraw = list(-1*np.ones(parent.ncells, np.int16))
    parent.update_layer()
    parent.update_plot()

def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('GUI_INFO: saving 2D masks to png')
        imsave(base + '_cp_masks.png', parent.cellpix[0])
    else:
        print('GUI_INFO: saving 3D masks to tiff')
        imsave(base + '_cp_masks.tif', parent.cellpix)

def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('GUI_INFO: saving 2D outlines to text file, see docs for info to load into ImageJ')    
        outlines = utils.outlines_list(parent.cellpix[0])
        outlines_to_text(base, outlines)
    else:
        print('ERROR: cannot save 3D outlines')
    

def _save_sets(parent):
    """ save masks to *_seg.npy """
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
        image = parent.chanchoose(parent.stack[parent.currentZ].copy())
        if image.ndim < 4:
            image = image[np.newaxis,...]
        np.save(base + '_seg.npy',
                {'outlines': parent.outpix.squeeze(),
                 'colors': parent.cellcolors[1:],
                 'masks': parent.cellpix.squeeze(),
                 'chan_choose': [parent.ChannelChoose[0].currentIndex(),
                                 parent.ChannelChoose[1].currentIndex()],
                 'img': image.squeeze(),
                 'filename': parent.filename,
                 'flows': parent.flows,
                 'ismanual': parent.ismanual,
                 'manual_changes': parent.track_changes,
                 'model_path': parent.current_model_path if hasattr(parent, 'current_model_path') else 0,
                 'flow_threshold': flow_threshold,
                 'cellprob_threshold': cellprob_threshold})
    #print(parent.point_sets)
    print('GUI_INFO: %d ROIs saved to %s'%(parent.ncells, base + '_seg.npy'))
