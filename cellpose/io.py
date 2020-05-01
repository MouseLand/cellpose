import os, datetime, gc, warnings
import numpy as np
import skimage.io 
import matplotlib.pyplot as plt

from . import plot, transforms

try:
    from PyQt5 import QtGui, QtCore, Qt, QtWidgets
    GUI = False
except:
    GUI = True

try:
    from google.cloud import storage
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        'key/cellpose-data-writer.json')
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False


def masks_flows_to_seg(images, masks, flows, diams, file_names, channels=None):
    """ save output of model eval to be loaded in GUI 

    saved to file_names[k]+'_seg.npy'
    
    Parameters
    -------------

    images: list of 2D or 3D arrays
        images input into cellpose

    masks: list of 2D arrays, int
        masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flows: list of lists of ND arrays 
        flows output from Cellpose.eval

    diams: float array
        diameters used to run Cellpose

    file_names: list of str
        names of files of images

    channels: list of int (optional, default None)
        channels used to run Cellpose    
    
    """
    nimg = len(masks)
    if channels is None:
        channels = [0,0]
    for n in range(nimg):
        flowi = []
        if flows[n][0].ndim==3:
            flowi.append(flows[n][0][np.newaxis,...])
        else:
            flowi.append(flows[n][0])
        print(flowi[0].shape)
        flowi.append((np.clip(transforms.normalize99(flows[n][2]),0,1) * 255).astype(np.uint8)[np.newaxis,...])
        if flows[n][0].ndim==3:
            flowi.append(np.zeros(flows[1][0].shape, dtype=np.uint8))
            flowi[-1] = flowi[-1][np.newaxis,...]
        else:
            flowi.append((flows[n][1][0]/10 * 127 + 127).astype(np.uint8))
        if len(flows[n])>2:
            flowi.append(flows[n][3])
            flowi.append(np.concatenate((flows[n][1], flows[n][2][np.newaxis,...]), axis=0))
        outlines = masks[n] * plot.masks_to_outlines(masks[n])
        base = os.path.splitext(file_names[n])[0]
        if images[n].shape[0]<8:
            np.transpose(images[n], (1,2,0))
        np.save(base+ '_seg.npy',
                    {'outlines': outlines.astype(np.uint16),
                     'masks': masks[n].astype(np.uint16),
                     'chan_choose': channels,
                     'img': images[n],
                     'ismanual': np.zeros(masks[n].max(), np.bool),
                     'filename': file_names[n],
                     'flows': flowi,
                     'est_diam': diams[n]})

def save_to_png(images, masks, flows, file_names):
    """ save masks + nicely plotted segmentation image to png 

    masks[k] for images[k] are saved to file_names[k]+'_cp_masks.png'

    full segmentation figure is saved to file_names[k]+'_cp.png'
    
    Parameters
    -------------

    images: list of 2D or 3D arrays
        images input into cellpose

    masks: list of 2D arrays, int
        masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flows: list of lists of ND arrays 
        flows output from Cellpose.eval

    file_names: list of str
        names of files of images
    
    """
    nimg = len(images)
    for n in range(nimg):
        img = images[n].copy()
        if img.ndim<3:
            img = img[:,:,np.newaxis]
        elif img.shape[0]<8:
            np.transpose(img, (1,2,0))
        base = os.path.splitext(file_names[n])[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(base+'_cp_masks.png', masks[n].astype(np.uint16))
        maski = masks[n]
        flowi = flows[n][0]
        fig = plt.figure(figsize=(12,3))
        # can save images (set save_dir=None if not)
        plot.show_segmentation(fig, img, maski, flowi)
        fig.savefig(base+'_cp.png', dpi=300)
        plt.close(fig)

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
        _load_seg(parent, manual_file, image=skimage.io.imread(filename), image_file=filename)
        return
    elif os.path.isfile(os.path.splitext(filename)[0]+'_manual.npy'):
        manual_file = os.path.splitext(filename)[0]+'_manual.npy'
        _load_seg(parent, manual_file, image=skimage.io.imread(filename), image_file=filename)
        return
    try:
        image = skimage.io.imread(filename)
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

def _initialize_images(parent, image, resize, X2):
    """ format image for GUI """
    parent.onechan=False
    if image.ndim > 3:
        # tiff is Z x channels x W x H
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
                image = skimage.io.imread(parent.filename)
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
            parent.ncells = np.uint16(parent.cellpix.max())
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
        try:
            print(parent.flows[0].shape)
        except:
            parent.flows = parent.flows[0]

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
    masks = skimage.io.imread(filename)
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
    parent.cellpix = masks.astype(np.uint16)
    # get outlines
    if outlines is None:
        parent.outpix = np.zeros(masks.shape, np.uint16)
        for z in range(parent.NZ):
            outlines = plot.masks_to_outlines(masks[z])
            parent.outpix[z] = ((outlines * masks[z])).astype(np.uint16)
            if z%50==0:
                print('plane %d outlines processed'%z)
    else:
        parent.outpix = outlines
        shape = parent.outpix.shape
        _,parent.outpix = np.unique(parent.outpix, return_inverse=True)
        parent.outpix = np.reshape(parent.outpix, shape)

    parent.ncells = np.uint16(parent.cellpix.max())
    colors = parent.colormap[np.random.randint(0,1000,size=parent.ncells), :3]
    parent.cellcolors = list(np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8))
    parent.draw_masks()
    if parent.ncells>0:
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells, np.bool)
    parent.update_plot()

def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('saving 2D masks to png')
        skimage.io.imsave(base + '_masks.png', parent.cellpix[0])
    else:
        print('saving 3D masks to tiff')
        skimage.io.imsave(base + '_masks.tif', parent.cellpix)

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