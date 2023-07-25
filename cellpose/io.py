import os, datetime, gc, warnings, glob
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys
from tqdm import tqdm
from pathlib import Path

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

try:
    from google.cloud import storage
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False
  
io_logger = logging.getLogger(__name__)

def logger_setup():
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ]
                )
    logger = logging.getLogger(__name__)
    logger.info(f'WRITING LOG OUTPUT TO {log_file}')
    #logger.handlers[1].stream = sys.stdout

    return logger, log_file

from . import utils, plot, transforms

# helper function to check for a path; if it doesn't exist, make it 
def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def outlines_to_text(base, outlines):
    with open(base + '_cp_outlines.txt', 'w') as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ','.join(map(str, xy))
            f.write(xy_str)
            f.write('\n')

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]['shape']
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                io_logger.info(f'reading tiff with {ltif} planes')
                img = np.zeros((ltif, *shape), dtype=dtype)
                for i,page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)            
        return img
    elif ext != '.npy':
        try:
            img = cv2.imread(filename, -1)#cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            io_logger.critical('ERROR: could not read file, %s'%e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat['masks']
            return masks
        except Exception as e:
            io_logger.critical('ERROR: could not read masks from file, %s'%e)
            return None


def imsave(filename, arr):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        tifffile.imsave(filename, arr)
    else:
        if len(arr.shape)>2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, arr)
#         skimage.io.imsave(filename, arr.astype()) #cv2 doesn't handle transparency

def get_image_files(folder, mask_filter, imf=None, look_one_level_down=False):
    """ find all images in a folder and if look_one_level_down all subfolders """
    mask_filters = ['_cp_masks', '_cp_output', '_flows', '_masks', mask_filter]
    image_names = []
    if imf is None:
        imf = ''
    
    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*/")))
    folders.append(folder)

    for folder in folders:
        image_names.extend(glob.glob(folder + '/*%s.png'%imf))
        image_names.extend(glob.glob(folder + '/*%s.jpg'%imf))
        image_names.extend(glob.glob(folder + '/*%s.jpeg'%imf))
        image_names.extend(glob.glob(folder + '/*%s.tif'%imf))
        image_names.extend(glob.glob(folder + '/*%s.tiff'%imf))
    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and imfile[-len(mask_filter):] != mask_filter) or len(imfile) <= len(mask_filter) 
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
        io_logger.info('not all flows are present, running flow generation for all images')
        flow_names = None
    
    # check for masks
    if mask_filter =='_seg.npy':
        label_names = [label_names[n] + mask_filter for n in range(nimg)]
        return label_names, None

    if os.path.exists(label_names[0] + mask_filter + '.tif'):
        label_names = [label_names[n] + mask_filter + '.tif' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.tiff'):
        label_names = [label_names[n] + mask_filter + '.tiff' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.png'):
        label_names = [label_names[n] + mask_filter + '.png' for n in range(nimg)]
    # todo, allow _seg.npy
    #elif os.path.exists(label_names[0] + '_seg.npy'):
    #    io_logger.info('labels found as _seg.npy files, converting to tif')
    else:
        raise ValueError('labels not provided with correct --mask_filter')
    if not all([os.path.exists(label) for label in label_names]):
        raise ValueError('labels not provided for all images in train and/or test set')

    return label_names, flow_names


def load_images_labels(tdir, mask_filter='_masks', image_filter=None, look_one_level_down=False, unet=False):
    image_names = get_image_files(tdir, mask_filter, image_filter, look_one_level_down)
    nimg = len(image_names)

    # training data
    label_names, flow_names = get_label_files(image_names, mask_filter, imf=image_filter)
    
    images = []
    labels = []
    k = 0
    for n in range(nimg):
        if os.path.isfile(label_names[n]):
            image = imread(image_names[n])
            label = imread(label_names[n])
            if not unet:
                if flow_names is not None and not unet:
                    flow = imread(flow_names[n])
                    if flow.shape[0]<4:
                        label = np.concatenate((label[np.newaxis,:,:], flow), axis=0) 
                    else:
                        label = flow
            images.append(image)
            labels.append(label)
            k+=1
    io_logger.info(f'{k} / {nimg} images in {tdir} folder have labels')
    return images, labels, image_names

def load_train_test_data(train_dir, test_dir=None, image_filter=None, mask_filter='_masks', unet=False, look_one_level_down=False):
    images, labels, image_names = load_images_labels(train_dir, mask_filter, image_filter, look_one_level_down, unet)
                    
    # testing data
    test_images, test_labels, test_image_names = None, None, None
    if test_dir is not None:
        test_images, test_labels, test_image_names = load_images_labels(test_dir, mask_filter, image_filter, look_one_level_down, unet)

    return images, labels, image_names, test_images, test_labels, test_image_names



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
        if not isinstance(diams, (list, np.ndarray)):
            diams = diams * np.ones(len(masks), np.float32)
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
                        'ismanual': np.zeros(masks.max(), bool),
                        'filename': file_names,
                        'flows': flowi,
                        'est_diam': diams})
    else:
        if images.shape[0]<8:
            np.transpose(images, (1,2,0))
        np.save(base+ '_seg.npy',
                    {'img': images,
                        'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                     'masks': masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32),
                     'chan_choose': channels,
                     'ismanual': np.zeros(masks.max(), bool),
                     'filename': file_names,
                     'flows': flowi,
                     'est_diam': diams})    

def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True) 
    
        does not work for 3D images
    
    """
    save_masks(images, masks, flows, file_names, png=True)

# Now saves flows, masks, etc. to separate folders.
def save_masks(images, masks, flows, file_names, png=True, tif=False, channels=[0,0],
               suffix='',save_flows=False, save_outlines=False, save_ncolor=False, 
               dir_above=False, in_folders=False, savedir=None, save_txt=True):
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
        
    savedir: str
        absolute path where images will be saved. Default is none (saves to image directory)
    
    save_flows, save_outlines, save_ncolor, save_txt: bool
        Can choose which outputs/views to save.
        ncolor is a 4 (or 5, if 4 takes too long) index version of the labels that
        is way easier to visualize than having hundreds of unique colors that may
        be similar and touch. Any color map can be applied to it (0,1,2,3,4,...).
    
    """

    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif, suffix=suffix,dir_above=dir_above,
                       save_flows=save_flows,save_outlines=save_outlines,save_ncolor=save_ncolor,
                       savedir=savedir,save_txt=save_txt,in_folders=in_folders)
        return
    
    if masks.ndim > 2 and not tif:
        raise ValueError('cannot save 3D outputs as PNG, use tif option instead')
#     base = os.path.splitext(file_names)[0]
    
    if savedir is None: 
        if dir_above:
            savedir = Path(file_names).parent.parent.absolute() #go up a level to save in its own folder
        else:
            savedir = Path(file_names).parent.absolute()
    
    check_dir(savedir) 
            
    basename = os.path.splitext(os.path.basename(file_names))[0]
    if in_folders:
        maskdir = os.path.join(savedir,'masks')
        outlinedir = os.path.join(savedir,'outlines')
        txtdir = os.path.join(savedir,'txt_outlines')
        ncolordir = os.path.join(savedir,'ncolor_masks')
        flowdir = os.path.join(savedir,'flows')
    else:
        maskdir = savedir
        outlinedir = savedir
        txtdir = savedir
        ncolordir = savedir
        flowdir = savedir
        
    check_dir(maskdir) 

    exts = []
    if masks.ndim > 2:
        png = False
        tif = True
    if png:    
        if masks.max() < 2**16:
            masks = masks.astype(np.uint16) 
            exts.append('.png')
        else:
            png = False 
            tif = True
            io_logger.warning('found more than 65535 masks in each image, cannot save PNG, saving as TIF')
    if tif:
        exts.append('.tif')

    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:
            
            imsave(os.path.join(maskdir,basename + '_cp_masks' + suffix + ext), masks)
            
    if png and MATPLOTLIB and not min(images.shape) > 3:
        img = images.copy()
        if img.ndim<3:
            img = img[:,:,np.newaxis]
        elif img.shape[0]<8:
            np.transpose(img, (1,2,0))
        
        fig = plt.figure(figsize=(12,3))
        plot.show_segmentation(fig, img, masks, flows[0])
        fig.savefig(os.path.join(savedir,basename + '_cp_output' + suffix + '.png'), dpi=300)
        plt.close(fig)

    # ImageJ txt outline files 
    if masks.ndim < 3 and save_txt:
        check_dir(txtdir)
        outlines = utils.outlines_list(masks)
        outlines_to_text(os.path.join(txtdir,basename), outlines)
    
    # RGB outline images
    if masks.ndim < 3 and save_outlines: 
        check_dir(outlinedir) 
        outlines = utils.masks_to_outlines(masks)
        outX, outY = np.nonzero(outlines)
        img0 = transforms.normalize99(images)
        if img0.shape[0] < 4:
            img0 = np.transpose(img0, (1,2,0))
        if img0.shape[-1] < 3 or img0.ndim < 3:
            img0 = plot.image_to_rgb(img0, channels=channels)
        else:
            if img0.max()<=50.0:
                img0 = np.uint8(np.clip(img0*255, 0, 1))
        imgout= img0.copy()
        imgout[outX, outY] = np.array([255,0,0]) #pure red 
        imsave(os.path.join(outlinedir, basename + '_outlines' + suffix + '.png'),  imgout)
    
    # save RGB flow picture
    if masks.ndim < 3 and save_flows:
        check_dir(flowdir)
        imsave(os.path.join(flowdir, basename + '_flows' + suffix + '.tif'), (flows[0]*(2**16 - 1)).astype(np.uint16))
        #save full flow data
        imsave(os.path.join(flowdir, basename + '_dP' + suffix + '.tif'), flows[1]) 

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
        io_logger.info(f'sending {source_file_name} to server')
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
        filestring = time + '.npy'
        io_logger.info(f'name on server: {filestring}')
        destination_blob_name = filestring
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        io_logger.info(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )