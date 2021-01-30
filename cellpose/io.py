import os, datetime, gc, warnings, glob
from natsort import natsorted
import numpy as np
import cv2
import tifffile

from . import utils, plot, transforms

try:
    from PyQt5 import QtGui, QtCore, Qt, QtWidgets
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False
    print('matplotlib not installed')

try:
    from google.cloud import storage
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False


def outlines_to_text(base, outlines):
    with open(base + '_cp_outlines.txt', 'w') as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ','.join(map(str, xy))
            f.write(xy_str)
            f.write('\n')

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='tiff':
        img = tifffile.imread(filename)
        return img
    else:
        try:
            img = cv2.imread(filename, -1)#cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            print('ERROR: could not read file, %s'%e)
            return None

def imsave(filename, arr):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='tiff':
        tifffile.imsave(filename, arr)
    else:
        cv2.imwrite(filename, arr)

def get_image_files(folder, mask_filter, imf=None):
    mask_filters = ['_cp_masks', '_cp_output', '_flows', mask_filter]
    image_names = []
    if imf is None:
        imf = ''
    image_names.extend(glob.glob(folder + '/*%s.png'%imf))
    image_names.extend(glob.glob(folder + '/*%s.jpg'%imf))
    image_names.extend(glob.glob(folder + '/*%s.jpeg'%imf))
    image_names.extend(glob.glob(folder + '/*%s.tif'%imf))
    image_names.extend(glob.glob(folder + '/*%s.tiff'%imf))
    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and imfile[-len(mask_filter):] != mask_filter) or len(imfile) < len(mask_filter) 
                        for mask_filter in mask_filters])
        if len(imf)>0:
            igood &= imfile[-len(imf):]==imf
        if igood:
            imn.append(im)
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names
        
def get_label_files(image_names, mask_filter, imf=None):
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0
        
    # check for flows
    if os.path.exists(label_names0[0] + '_flows.tif'):
        flow_names = [label_names0[n] + '_flows.tif' for n in range(nimg)]
    else:
        flow_names = [label_names[n] + '_flows.tif' for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        flow_names = None
    
    # check for masks
    if os.path.exists(label_names[0] + mask_filter + '.tif'):
        label_names = [label_names[n] + mask_filter + '.tif' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.png'):
        label_names = [label_names[n] + mask_filter + '.png' for n in range(nimg)]
    else:
        raise ValueError('labels not provided with correct --mask_filter')
    if not all([os.path.exists(label) for label in label_names]):
        raise ValueError('labels not provided for all images in train and/or test set')

    return label_names, flow_names


def load_train_test_data(train_dir, test_dir=None, image_filter=None, mask_filter='_masks', unet=False):
    image_names = get_image_files(train_dir, mask_filter, imf=image_filter)
    nimg = len(image_names)
    images = [imread(image_names[n]) for n in range(nimg)]

    # training data
    label_names, flow_names = get_label_files(image_names, mask_filter, imf=image_filter)
    nimg = len(image_names)
    labels = [imread(label_names[n]) for n in range(nimg)]
    if flow_names is not None and not unet:
        for n in range(nimg):
            flows = imread(flow_names[n])
            if flows.shape[0]<4:
                labels[n] = np.concatenate((labels[n][np.newaxis,:,:], flows), axis=0) 
            else:
                labels[n] = flows
            
    # testing data
    test_images, test_labels, image_names_test = None, None, None
    if test_dir is not None:
        image_names_test = get_image_files(test_dir, mask_filter, imf=image_filter)
        label_names_test, flow_names_test = get_label_files(image_names_test, mask_filter, imf=image_filter)
        nimg = len(image_names_test)
        test_images = [imread(image_names_test[n]) for n in range(nimg)]
        test_labels = [imread(label_names_test[n]) for n in range(nimg)]
        if flow_names_test is not None and not unet:
            for n in range(nimg):
                flows = imread(flow_names_test[n])
                if flows.shape[0]<4:
                    test_labels[n] = np.concatenate((test_labels[n][np.newaxis,:,:], flows), axis=0) 
                else:
                    test_labels[n] = flows
    return images, labels, image_names, test_images, test_labels, image_names_test



def masks_flows_to_seg(images, masks, flows, diams, file_names, channels=None):
    """ save output of model eval to be loaded in GUI 

    can be list output (run on multiple images) or single output (run on single image)

    saved to file_names[k]+'_seg.npy'
    
    Parameters
    -------------

    images: (list of) 2D or 3D arrays
        images input into cellpose

    masks: (list of) 2D arrays, int
        masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from Cellpose.eval

    diams: float array
        diameters used to run Cellpose

    file_names: (list of) str
        names of files of images

    channels: list of int (optional, default None)
        channels used to run Cellpose    
    
    """
    
    if channels is None:
        channels = [0,0]
    
    if isinstance(masks, list):
        for k, [image, mask, flow, diam, file_name] in enumerate(zip(images, masks, flows, diams, file_names)):
            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(image, mask, flow, diam, file_name, channels_img)
        return

    if len(channels)==1:
        channels = channels[0]

    flowi = []
    if flows[0].ndim==3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...])
    else:
        flowi.append(flows[0])
    if flows[0].ndim==3:
        cellprob = (np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis,...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis,...]
    else:
        flowi.append((np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8))
        flowi.append((flows[1][0]/10 * 127 + 127).astype(np.uint8))
    if len(flows)>2:
        flowi.append(flows[3])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis,...]), axis=0))
    outlines = masks * utils.masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]
    if masks.ndim==3:
        np.save(base+ '_seg.npy',
                    {'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                        'masks': masks.astype(np.uint16) if outlines.max()<2**16-1 else masks.astype(np.uint32),
                        'chan_choose': channels,
                        'img': images,
                        'ismanual': np.zeros(masks.max(), np.bool),
                        'filename': file_names,
                        'flows': flowi,
                        'est_diam': diams})
    else:
        if images.shape[0]<8:
            np.transpose(images, (1,2,0))
        np.save(base+ '_seg.npy',
                    {'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                     'masks': masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32),
                     'chan_choose': channels,
                     'ismanual': np.zeros(masks.max(), np.bool),
                     'filename': file_names,
                     'flows': flowi,
                     'est_diam': diams})    

def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True) 
    
        does not work for 3D images
    
    """
    save_masks(images, masks, flows, file_names, png=True)

def save_masks(images, masks, flows, file_names, png=True, tif=False):
    """ save masks + nicely plotted segmentation image to png and/or tiff

    if png, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.png'

    if tif, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.tif'

    if png and matplotlib installed, full segmentation figure is saved to file_names[k]+'_cp.png'

    only tif option works for 3D data
    
    Parameters
    -------------

    images: (list of) 2D, 3D or 4D arrays
        images input into cellpose

    masks: (list of) 2D arrays, int
        masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from Cellpose.eval

    file_names: (list of) str
        names of files of images
    
    """
    
    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif)
        return
    
    if masks.ndim > 2 and not tif:
        raise ValueError('cannot save 3D outputs as PNG, use tif option instead')
    base = os.path.splitext(file_names)[0]
    exts = []
    if masks.ndim > 2 or masks.max()>2**16-1:
        png = False
        tif = True
    if png:    
        exts.append('.png')
    if tif:
        exts.append('.tif')

    # convert to uint16 if possible so can save as PNG if needed
    masks = masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32)
    
    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:
            imsave(base + '_cp_masks' + ext, masks)

    if png and MATPLOTLIB and not min(images.shape) > 3:
        img = images.copy()
        if img.ndim<3:
            img = img[:,:,np.newaxis]
        elif img.shape[0]<8:
            np.transpose(img, (1,2,0))
        
        fig = plt.figure(figsize=(12,3))
        # can save images (set save_dir=None if not)
        plot.show_segmentation(fig, img, masks, flows[0])
        fig.savefig(base + '_cp_output.png', dpi=300)
        plt.close(fig)

    if masks.ndim < 3: 
        outlines = utils.outlines_list(masks)
        outlines_to_text(base, outlines)

def save_server(parent=None, filename=None):
    """ Uploads a *_seg.npy file to the bucket.
    
    Parameters
    ----------------

    parent: PyQt.MainWindow (optional, default None)
        GUI window to grab file info from

    filename: str (optional, default None)
        if no GUI, send this file to server

    """
    if parent is not None:
        q = QtGui.QMessageBox.question(
                                    parent,
                                    "Send to server",
                                    "Are you sure? Only send complete and fully manually segmented data.\n (do not send partially automated segmentations)",
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
                                  )
        if q != QtGui.QMessageBox.Yes:
            return
        else:
            filename = parent.filename

    if filename is not None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        'key/cellpose-data-writer.json')
        bucket_name = 'cellpose_data'
        base = os.path.splitext(filename)[0]
        source_file_name = base + '_seg.npy'
        print(source_file_name)
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
        filestring = time + '.npy'
        print(filestring)
        destination_blob_name = filestring
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

def _load_image(parent, filename=None):
    """ load image with filename; if None, open QFileDialog """
    if filename is None:
        name = QtGui.QFileDialog.getOpenFileName(
            parent, "Load image"
            )
        filename = name[0]
    manual_file = os.path.splitext(filename)[0]+'_seg.npy'
    if os.path.isfile(manual_file):
        print(manual_file)
        _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
        return
    elif os.path.isfile(os.path.splitext(filename)[0]+'_manual.npy'):
        manual_file = os.path.splitext(filename)[0]+'_manual.npy'
        _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
        return
    try:
        image = imread(filename)
        image.shape
        parent.loaded = True
    except:
        print('images not compatible')

    if parent.loaded:
        parent.reset()
        parent.filename = filename
        print(filename)
        filename = os.path.split(parent.filename)[-1]
        _initialize_images(parent, image, resize=parent.resize, X2=0)
        parent.clear_all()
        parent.loaded = True
        parent.enable_buttons()
        parent.threshslider.setEnabled(False)
        parent.probslider.setEnabled(False)
            

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
            image = np.concatenate((image,
                                       np.zeros((shape[0], shape[1], 3-shape[2]),
                                        dtype=type(image[0,0,0]))), axis=-1)
            if 3-shape[2]>1:
                parent.onechan=True
            image = image[np.newaxis,...]
        elif image.shape[-1]<5 and image.shape[-1]>2:
            image = image[:,:,:3]
            image = image[np.newaxis,...]
    else:
        image = image[np.newaxis,...]

    parent.stack = image
    parent.NZ = len(parent.stack)
    parent.scroll.setMaximum(parent.NZ-1)
    if parent.stack.max()>255 or parent.stack.min()<0.0 or parent.stack.max()<=50.0:
        parent.stack = parent.stack.astype(np.float32)
        parent.stack -= parent.stack.min()
        parent.stack /= parent.stack.max()
        parent.stack *= 255
    del image
    gc.collect()

    parent.stack = list(parent.stack)
    for k,img in enumerate(parent.stack):
        # if grayscale make 3D
        if resize != -1:
            img = transforms._image_resizer(img, resize=resize, to_uint8=False)
        if img.ndim==2:
            img = np.tile(img[:,:,np.newaxis], (1,1,3))
            parent.onechan=True
        if X2!=0:
            img = transforms._X2zoom(img, X2=X2)
        parent.stack[k] = img

    parent.imask=0
    print(parent.NZ, parent.stack[0].shape)
    parent.Ly, parent.Lx = img.shape[0], img.shape[1]
    parent.stack = np.array(parent.stack)
    parent.layers = 0*np.ones((parent.NZ,parent.Ly,parent.Lx,4), np.uint8)
    if parent.autobtn.isChecked() or len(parent.saturation)!=parent.NZ:
        parent.compute_saturation()
    parent.compute_scale()
    parent.currentZ = int(np.floor(parent.NZ/2))
    parent.scroll.setValue(parent.currentZ)
    parent.zpos.setText(str(parent.currentZ))

def _load_seg(parent, filename=None, image=None, image_file=None):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    if filename is None:
        name = QtGui.QFileDialog.getOpenFileName(
            parent, "Load labelled data", filter="*.npy"
            )
        filename = name[0]
    try:
        dat = np.load(filename, allow_pickle=True).item()
        dat['outlines']
        parent.loaded = True
    except:
        parent.loaded = False
        print('not NPY')
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
    print(parent.filename)

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
                    parent.cellcolors.append(color)
                    parent.ncells+=1
        else:
            if dat['masks'].ndim==2:
                dat['masks'] = dat['masks'][np.newaxis,:,:]
                dat['outlines'] = dat['outlines'][np.newaxis,:,:]
            if dat['masks'].min()==-1:
                dat['masks'] += 1
                dat['outlines'] += 1
            if 'colors' in dat:
                colors = dat['colors']
            else:
                col_rand = np.random.randint(0, 1000, (dat['masks'].max(),))
                colors = parent.colormap[col_rand,:3]
            parent.cellpix = dat['masks']
            parent.outpix = dat['outlines']
            parent.cellcolors.extend(colors)
            parent.ncells = parent.cellpix.max()
            parent.draw_masks()
            if 'est_diam' in dat:
                parent.Diameter.setText('%0.1f'%dat['est_diam'])
                parent.diameter = dat['est_diam']
                parent.compute_scale()

            if parent.masksOn or parent.outlinesOn and not (parent.masksOn and parent.outlinesOn):
                parent.redraw_masks(masks=parent.masksOn, outlines=parent.outlinesOn)
        if 'zdraw' in dat:
            parent.zdraw = dat['zdraw']
        else:
            parent.zdraw = [None for n in range(parent.ncells)]
        parent.loaded = True
        print('%d masks found'%(parent.ncells))
    else:
        parent.clear_all()

    parent.ismanual = np.zeros(parent.ncells, np.bool)
    if 'ismanual' in dat:
        if len(dat['ismanual']) == parent.ncells:
            parent.ismanual = dat['ismanual']

    if 'current_channel' in dat:
        parent.color = (dat['current_channel']+2)%5
        parent.RGBDropDown.setCurrentIndex(parent.color)

    if 'flows' in dat:
        parent.flows = dat['flows']
        if parent.flows[0].shape[-3]!=dat['masks'].shape[-2]:
            Ly, Lx = dat['masks'].shape[-2:]
            parent.flows[0] = cv2.resize(parent.flows[0][0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...]
            parent.flows[1] = cv2.resize(parent.flows[1][0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...]
        try:
            if parent.NZ==1:
                parent.threshslider.setEnabled(True)
                parent.probslider.setEnabled(True)
            else:
                parent.threshslider.setEnabled(False)
                parent.probslider.setEnabled(False)
        except:
            try:
                if len(parent.flows[0])>0:
                    parent.flows = parent.flows[0]
            except:
                parent.flows = [[],[],[],[],[[]]]
            parent.threshslider.setEnabled(False)
            parent.probslider.setEnabled(False)
            
    parent.enable_buttons()
    del dat
    gc.collect()

def _load_masks(parent, filename=None):
    """ load zeros-based masks (0=no cell, 1=cell 1, ...) """
    if filename is None:
        name = QtGui.QFileDialog.getOpenFileName(
            parent, "Load masks (PNG or TIFF)"
            )
        filename = name[0]
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
    print('%d masks found'%(len(np.unique(masks))-1))

    _masks_to_gui(parent, masks, outlines)

    parent.update_plot()

def _masks_to_gui(parent, masks, outlines=None):
    """ masks loaded into GUI """
    # get unique values
    shape = masks.shape
    _, masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape)
    masks = masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32)
    parent.cellpix = masks

    # get outlines
    if outlines is None:
        parent.outpix = np.zeros_like(masks)
        for z in range(parent.NZ):
            outlines = utils.masks_to_outlines(masks[z])
            parent.outpix[z] = outlines * masks[z]
            if z%50==0:
                print('plane %d outlines processed'%z)
    else:
        parent.outpix = outlines
        shape = parent.outpix.shape
        _,parent.outpix = np.unique(parent.outpix, return_inverse=True)
        parent.outpix = np.reshape(parent.outpix, shape)

    parent.ncells = parent.cellpix.max()
    colors = parent.colormap[np.random.randint(0,1000,size=parent.ncells), :3]
    parent.cellcolors = list(np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8))
    parent.draw_masks()
    if parent.ncells>0:
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells, np.bool)
    parent.zdraw = list(-1*np.ones(parent.ncells, np.int16))
    parent.update_plot()

def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('saving 2D masks to png')
        imsave(base + '_cp_masks.png', parent.cellpix[0])
    else:
        print('saving 3D masks to tiff')
        imsave(base + '_cp_masks.tif', parent.cellpix)

def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('saving 2D outlines to text file, see docs for info to load into ImageJ')    
        outlines = utils.outlines_list(parent.cellpix[0])
        outlines_to_text(base, outlines)
    else:
        print('ERROR: cannot save 3D outlines')
    

def _save_sets(parent):
    """ save masks to *_seg.npy """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ > 1 and parent.is_stack:
        np.save(base + '_seg.npy',
                {'outlines': parent.outpix,
                 'colors': parent.cellcolors[1:],
                 'masks': parent.cellpix,
                 'current_channel': (parent.color-2)%5,
                 'filename': parent.filename,
                 'flows': parent.flows,
                 'zdraw': parent.zdraw})
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
                 'ismanual': parent.ismanual,
                 'X2': parent.X2,
                 'filename': parent.filename,
                 'flows': parent.flows})
    #print(parent.point_sets)
    print('--- %d ROIs saved chan1 %s, chan2 %s'%(parent.ncells,
                                                  parent.ChannelChoose[0].currentText(),
                                                  parent.ChannelChoose[1].currentText()))