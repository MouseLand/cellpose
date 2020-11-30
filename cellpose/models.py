import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2

from mxnet import gluon, nd
import mxnet as mx

from . import transforms, dynamics, utils, resnet_style, plot, metrics
import __main__

def use_gpu(gpu_number=0):
    """ check if mxnet gpu works """
    try:
        _ = mx.ndarray.array([1, 2, 3], ctx=mx.gpu(gpu_number))
        print('** CUDA version installed and working. **')
        return True
    except mx.MXNetError:
        print('CUDA version not installed/working, will use CPU version.')
        return False

def check_mkl():
    print('Running test snippet to check if MKL running (https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_readme.html#4)')
    process = subprocess.Popen(['python', 'test_mkl.py'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                cwd=os.path.dirname(os.path.abspath(__file__)))
    stdout, stderr = process.communicate()
    if len(stdout)>0:
        print('** MKL version working - CPU version is fast. **')
        mkl_enabled = True
    else:
        print('WARNING: MKL version not working/installed - CPU version will be SLOW!')
        mkl_enabled = False
    return mkl_enabled

def dx_to_circ(dP):
    """ dP is 2 x Y x X => 'optic' flow representation """
    sc = max(np.percentile(dP[0], 99), np.percentile(dP[0], 1))
    Y = np.clip(dP[0] / sc, -1, 1)
    sc = max(np.percentile(dP[1], 99), np.percentile(dP[1], 1))
    X = np.clip(dP[1] / sc, -1, 1)
    H = (np.arctan2(Y, X) + np.pi) / (2*np.pi)
    S = utils.normalize99(dP[0]**2 + dP[1]**2)
    V = np.ones_like(S)
    HSV = np.concatenate((H[:,:,np.newaxis], S[:,:,np.newaxis], S[:,:,np.newaxis]), axis=-1)
    HSV = np.clip(HSV, 0.0, 1.0)
    flow = (utils.hsv_to_rgb(HSV)*255).astype(np.uint8)
    return flow

class Cellpose():
    """ main model which combines SizeModel and CellposeModel

    Parameters
    ----------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available

    model_type: str (optional, default 'cyto')
        'cyto'=cytoplasm model; 'nuclei'=nucleus model

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))

    """
    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None):
        super(Cellpose, self).__init__()
        # assign device (GPU or CPU)
        if device is not None:
            self.device = device
        elif gpu and use_gpu():
            self.device = mx.gpu()
            print('>>>> using GPU')
        else:
            self.device = mx.cpu()
            print('>>>> using CPU')

        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')
        if model_type is None:
            model_type = 'cyto'

        self.pretrained_model = [os.fspath(model_dir.joinpath('%s_%d'%(model_type,j))) for j in range(4)]
        self.pretrained_size = os.fspath(model_dir.joinpath('size_%s_0.npy'%(model_type)))
        if model_type=='cyto':
            self.diam_mean = 30.
        else:
            self.diam_mean = 17.
        if not os.path.isfile(self.pretrained_model[0]):
            download_model_weights()
        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]

        self.cp = CellposeModel(device=self.device,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean)
        self.cp.model_type = model_type
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=None, invert=False, normalize=True, diameter=30., do_3D=False, anisotropy=None,
             net_avg=True, augment=False, tile=True, tile_overlap=0.1, resample=False, flow_threshold=0.4, cellprob_threshold=0.0,
             min_size=15, stitch_threshold=0.0, rescale=None, progress=None):
        """ run cellpose and get masks

        Parameters
        ----------
        x: list or array of images
            can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

        batch_size: int (optional, default 8)
            number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)

        channels: list (optional, default None)
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
            For instance, to segment grayscale images, input [0,0]. To segment images with cells
            in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
            image with cells in green and nuclei in blue, input [[0,0], [2,3]].

        invert: bool (optional, default False)
            invert image pixel intensity before running network (if True, image is also normalized)

        normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

        diameter: float (optional, default 30.)
            if set to None, then diameter is automatically estimated if size model is loaded

        do_3D: bool (optional, default False)
            set to True to run 3D segmentation on 4D image input

        anisotropy: float (optional, default None)
            for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        resample: bool (optional, default False)
            run dynamics at original image size (will be slower but create more accurate boundaries)

        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

        cellprob_threshold: float (optional, default 0.0)
            cell probability threshold (all pixels with prob above threshold kept for masks)

        min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

        stitch_threshold: float (optional, default 0.0)
            if stitch_threshold>0.0 and not do_3D and equal image sizes, masks are stitched in 3D to return volume segmentation

        rescale: float (optional, default None)
            if diameter is set to None, and rescale is not None, then rescale is used instead of diameter for resizing image

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI

        Returns
        -------
        masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

        flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
            flows[k][0] = XY flow in HSV 0-255
            flows[k][1] = flows at each pixel
            flows[k][2] = the cell probability centered at 0.0

        styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
            style vector summarizing each image, also used to estimate size of objects in image

        diams: list of diameters, or float (if do_3D=True)

        """

        if not isinstance(x,list):
            nolist = True
            if x.ndim < 2 or x.ndim > 5:
                raise ValueError('%dD images not supported'%x.ndim)
            if x.ndim==4:
                if do_3D:
                    x = [x]
                else:
                    x = list(x)
                    nolist = False
            elif x.ndim==5: 
                if do_3D:
                    x = list(x)
                    nolist = False
                else:
                    raise ValueError('4D images must be processed using 3D')
            else:
                x = [x]
        else:
            nolist = False
            for xi in x:
                if xi.ndim < 2 or xi.ndim > 5:
                    raise ValueError('%dD images not supported'%xi.ndim)
            
        tic0 = time.time()

        nimg = len(x)
        print('processing %d image(s)'%nimg)
        # make rescale into length of x
        if diameter is not None and diameter!=0:
            if not isinstance(diameter, list) or len(diameter)==1 or len(diameter)<nimg:
                diams = diameter * np.ones(nimg, np.float32)
            else:
                diams = diameter
            rescale = self.diam_mean / diams
        else:
            if rescale is not None and (not isinstance(rescale, list) or len(rescale)==1):
                rescale = rescale * np.ones(nimg, np.float32)
            if self.pretrained_size is not None and rescale is None and not do_3D:
                tic = time.time()
                diams, _ = self.sz.eval(x, channels=channels, invert=invert, batch_size=batch_size, augment=augment, tile=tile)
                rescale = self.diam_mean / diams
                print('estimated cell diameters for %d image(s) in %0.2f sec'%(nimg, time.time()-tic))
            else:
                if rescale is None:
                    if do_3D:
                        rescale = np.ones(1)
                    else:
                        rescale = np.ones(nimg, np.float32)
                diams = self.diam_mean / rescale

        tic = time.time()
        masks, flows, styles = self.cp.eval(x, 
                                            batch_size=batch_size, 
                                            invert=invert, 
                                            rescale=rescale, 
                                            anisotropy=anisotropy, 
                                            channels=channels, 
                                            augment=augment, 
                                            tile=tile, 
                                            do_3D=do_3D, 
                                            net_avg=net_avg, progress=progress,
                                            tile_overlap=tile_overlap,
                                            resample=resample,
                                            flow_threshold=flow_threshold, 
                                            cellprob_threshold=cellprob_threshold,
                                            min_size=min_size, 
                                            stitch_threshold=stitch_threshold)
        print('estimated masks for %d image(s) in %0.2f sec'%(nimg, time.time()-tic))
        print('>>>> TOTAL TIME %0.2f sec'%(time.time()-tic0))
        
        if nolist:
            masks, flows, styles, diams = masks[0], flows[0], styles[0], diams[0]
        
        return masks, flows, styles, diams

def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        print('parsing model string to get unet options')
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        print('parsing model string to get cellpose options')
        nclasses = 3
    else:
        return None
    ostrs = model_str.split('_')[2::2]
    residual_on = ostrs[0]=='on'
    style_on = ostrs[1]=='on'
    concatenation = ostrs[2]=='on'
    return nclasses, residual_on, style_on, concatenation

class UnetModel():
    def __init__(self, gpu=False, pretrained_model=False,
                    diam_mean=30., net_avg=True, device=None,
                    residual_on=False, style_on=False, concatenation=True,
                    nclasses = 3):
        self.unet = True
        if device is not None:
            self.device = device
        elif gpu and use_gpu():
            self.device = mx.gpu()
            print('>>>> using GPU')
        else:
            self.device = mx.cpu()
            print('>>>> using CPU')

        self.pretrained_model = pretrained_model
        self.diam_mean = diam_mean

        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params
        
        ostr = ['off', 'on']
        self.net_type = 'unet{}_residual_{}_style_{}_concatenation_{}'.format(nclasses,
                                                                                ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])                                             
        if pretrained_model:
            print(self.net_type)
        # create network
        self.nclasses = nclasses
        nbase = [32,64,128,256]
        self.net = resnet_style.CPnet(nbase, nout=self.nclasses,
                                      residual_on=residual_on, 
                                      style_on=style_on,
                                      concatenation=concatenation)
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.initialize(ctx = self.device)

        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_parameters(pretrained_model)

    def eval(self, x, batch_size=8, channels=None, invert=False, normalize=True,
             rescale=None, do_3D=False, anisotropy=None, net_avg=True, augment=False,
             tile=True, cell_threshold=None, boundary_threshold=None, min_size=15):
        """ segment list of images x

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            cell_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            boundary_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        x, nolist = convert_images(x, channels, do_3D, normalize, invert)
        nimg = len(x)
        self.batch_size = batch_size

        styles = []
        flows = []
        masks = []
        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        if nimg > 1:
            iterator = trange(nimg)
        else:
            iterator = range(nimg)

        if isinstance(self.pretrained_model, list):
            model_path = self.pretrained_model[0]
            if not net_avg:
                self.net.load_parameters(self.pretrained_model[0])
                self.net.collect_params().grad_req = 'null'
        else:
            model_path = self.pretrained_model

        if cell_threshold is None or boundary_threshold is None:
            try:
                thresholds = np.load(model_path+'_cell_boundary_threshold.npy')
                cell_threshold, boundary_threshold = thresholds
                print('>>>> found saved thresholds from validation set')
            except:
                print('WARNING: no thresholds found, using default / user input')

        cell_threshold = 2.0 if cell_threshold is None else cell_threshold
        boundary_threshold = 0.5 if boundary_threshold is None else boundary_threshold

        if not do_3D:
            for i in iterator:
                img = x[i].copy()
                shape = img.shape
                # rescale image for flow computation
                imgs = transforms.resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg, augment=augment, tile=tile)
                
                maski = utils.get_masks_unet(y, cell_threshold, boundary_threshold)
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                maski = transforms.resize_image(maski, shape[-3], shape[-2], 
                                                    interpolation=cv2.INTER_NEAREST)
                masks.append(maski)
                styles.append(style)
        else:
            for i in iterator:
                tic=time.time()
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy, 
                                         net_avg=net_avg, augment=augment, tile=tile)
                yf = yf.mean(axis=0)
                print('probabilities computed %2.2fs'%(time.time()-tic))
                maski = utils.get_masks_unet(yf.transpose((1,2,3,0)), cell_threshold, boundary_threshold)
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                masks.append(maski)
                styles.append(style)
                print('masks computed %2.2fs'%(time.time()-tic))
                flows.append(yf)

        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]

        return masks, flows, styles

                
                
    def _run_nets(self, img, net_avg=True, augment=False, tile=True, tile_overlap=0.1, bsize=224, progress=None):
        """ run network (if more than one, loop over networks and average results

        Parameters
        --------------

        img: float, [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

        Returns
        ------------------

        y: array [3 x Ly x Lx] or [3 x Lz x Ly x Lx]
            y is output (averaged over networks);
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles,
            but not averaged over networks.

        """
        if isinstance(self.pretrained_model, str) or not net_avg:  
            y, style = self._run_net(img, augment=augment, tile=tile, bsize=bsize)
        else:  
            for j in range(len(self.pretrained_model)):
                self.net.load_parameters(self.pretrained_model[j])
                self.net.collect_params().grad_req = 'null'
                y0, style = self._run_net(img, augment=augment, tile=tile, 
                                          tile_overlap=tile_overlap, bsize=bsize)

                if j==0:
                    y = y0
                else:
                    y += y0
                if progress is not None:
                    progress.setValue(10 + 10*j)
            y = y / len(self.pretrained_model)
        return y, style

    def _run_net(self, imgs, augment=False, tile=True, tile_overlap=0.1, bsize=224):
        """ run network on image or stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """   
        if imgs.ndim==4:  
            # make image Lz x nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (0,3,1,2))  
            detranspose = (0,2,3,1)
        else:
            # make image nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (2,0,1))
            detranspose = (1,2,0)

        # pad image for net so Ly and Lx are divisible by 4
        imgs, ysub, xsub = transforms.pad_image_ND(imgs)
        # slices from padding
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)

        # run network
        if tile or augment or imgs.ndim==4:
            y,style = self._run_tiled(imgs, augment=augment, bsize=bsize, tile_overlap=tile_overlap)
        else:
            imgs = nd.array(np.expand_dims(imgs, axis=0), ctx=self.device)
            y,style = self.net(imgs)
            y = y[0].asnumpy()
            imgs = imgs.asnumpy()
            style = style.asnumpy()[0]
        style /= (style**2).sum()**0.5

        # slice out padding
        y = y[slc]

        # transpose so channels axis is last again
        y = np.transpose(y, detranspose)
         
        return y, style
    
    def _run_tiled(self, imgi, augment=False, bsize=224, tile_overlap=0.1):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].
        If augment, tiles have 50% overlap and are flipped at overlaps.
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
         
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        Returns
        ------------------

        yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles

        """

        if imgi.ndim==4:
            batch_size = self.batch_size 
            Lz, nchan = imgi.shape[:2]
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize, 
                                                            augment=augment, tile_overlap=tile_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            batch_size *= max(4, (bsize**2 // (ly*lx))**0.5)
            yf = np.zeros((Lz, self.nclasses, imgi.shape[-2], imgi.shape[-1]), np.float32)
            styles = []
            if ny*nx > batch_size:
                ziterator = trange(Lz)
                for i in ziterator:
                    yfi, stylei = self._run_tiled(imgi[i], augment=augment, 
                                                  bsize=bsize, tile_overlap=tile_overlap)
                    yf[i] = yfi
                    styles.append(stylei)
            else:
                # run multiple slices at the same time
                ntiles = ny*nx
                nimgs = max(2, int(np.round(batch_size / ntiles)))
                niter = int(np.ceil(Lz/nimgs))
                ziterator = trange(niter)
                for k in ziterator:
                    IMGa = np.zeros((ntiles*nimgs, nchan, ly, lx), np.float32)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[k*nimgs+i], bsize=bsize, 
                                                                        augment=augment, tile_overlap=tile_overlap)
                        IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, ly, lx))
                    y0, style = self.net(nd.array(IMGa, ctx=self.device))
                    ya = y0.asnumpy()
                    stylea = style.asnumpy()
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        y = ya[i*ntiles:(i+1)*ntiles]
                        if augment:
                            y = np.reshape(y, (ny, nx, 3, ly, lx))
                            y = transforms.unaugment_tiles(y, self.unet)
                            y = np.reshape(y, (-1, 3, ly, lx))
                        yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                        yfi = yfi[:,:imgi.shape[2],:imgi.shape[3]]
                        yf[k*nimgs+i] = yfi
                        stylei = stylea[i*ntiles:(i+1)*ntiles].sum(axis=0)
                        stylei /= (stylei**2).sum()**0.5
                        styles.append(stylei)
            return yf, np.array(styles)
        else:
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, 
                                                            augment=augment, tile_overlap=tile_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
            batch_size = self.batch_size
            niter = int(np.ceil(IMG.shape[0] / batch_size))
            y = np.zeros((IMG.shape[0], self.nclasses, ly, lx))
            for k in range(niter):
                irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
                y0, style = self.net(nd.array(IMG[irange], ctx=self.device))
                y0 = y0.asnumpy()
                y[irange] = y0
                if k==0:
                    styles = style.asnumpy()[0]
                styles += style.asnumpy().sum(axis=0)
            styles /= IMG.shape[0]
            if augment:
                y = np.reshape(y, (ny, nx, self.nclasses, bsize, bsize))
                y = transforms.unaugment_tiles(y, self.unet)
                y = np.reshape(y, (-1, self.nclasses, bsize, bsize))
            
            yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
            yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
            styles /= (styles**2).sum()**0.5
            return yf, styles

    def _run_3D(self, imgs, rsz=1.0, anisotropy=None, net_avg=True, 
                augment=False, tile=True, tile_overlap=0.1, 
                bsize=224, progress=None):
        """ run network on stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI


        Returns
        ------------------

        yf: array [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """ 
        sstr = ['YX', 'ZY', 'ZX']
        if anisotropy is not None:
            rescaling = [[rsz, rsz],
                         [rsz*anisotropy, rsz],
                         [rsz*anisotropy, rsz]]
        else:
            rescaling = [rsz] * 3
        pm = [(0,1,2,3), (1,0,2,3), (2,0,1,3)]
        ipm = [(3,0,1,2), (3,1,0,2), (3,1,2,0)]
        yf = np.zeros((3, self.nclasses, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
        for p in range(3 - 2*self.unet):
            xsl = imgs.copy().transpose(pm[p])
            # rescale image for flow computation
            shape = xsl.shape
            xsl = transforms.resize_image(xsl, rsz=rescaling[p])    
            # per image
            print('\n running %s: %d planes of size (%d, %d) \n\n'%(sstr[p], shape[0], shape[1], shape[2]))
            y, style = self._run_nets(xsl, net_avg=net_avg, augment=augment, tile=tile, 
                                      bsize=bsize, tile_overlap=tile_overlap)
            y = transforms.resize_image(y, shape[1], shape[2])    
            yf[p] = y.transpose(ipm[p])
            if progress is not None:
                progress.setValue(25+15*p)
        return yf, style

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        criterion = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
        # if available set boundary pixels to 2
        if lbl.shape[1]>1 and self.nclasses>2:
            boundary = lbl[:,1]<=4
            lbl = lbl[:,0]
            lbl[boundary] *= 2
        else:
            lbl = lbl[:,0]
        lbl = nd.array(lbl.astype(np.uint8), ctx=self.device)
        loss = 8 * 1./self.nclasses * criterion(y, lbl)
        return loss

    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None, save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, weight_decay=0.00001, batch_size=8, rescale=False):
        """ train function uses 0-1 mask label and boundary pixels for training """

        nimg = len(train_data)

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize)

        # add dist_to_bound to labels
        if self.nclasses==3:
            print('computing boundary pixels')
            train_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
                                for label in tqdm(train_labels)]
        else:
            train_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                for label in tqdm(train_labels)]
        if run_test:
            if self.nclasses==3:
                test_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels)]
            else:
                test_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels)]
        
        # split train data into train and val
        val_data = train_data[::8]
        val_classes = train_classes[::8]
        val_labels = train_labels[::8]
        del train_data[::8], train_classes[::8], train_labels[::8]

        model_path = self._train_net(train_data, train_classes, 
                                     test_data, test_classes,
                                     pretrained_model, save_path, save_every,
                                     learning_rate, n_epochs, weight_decay, 
                                     batch_size, rescale)


        # find threshold using validation set
        print('>>>> finding best thresholds using validation set')
        cell_threshold, boundary_threshold = self.threshold_validation(val_data, val_labels)
        np.save(model_path+'_cell_boundary_threshold.npy', np.array([cell_threshold, boundary_threshold]))

    def threshold_validation(self, val_data, val_labels):
        cell_thresholds = np.arange(-4.0, 4.25, 0.5)
        if self.nclasses==3:
            boundary_thresholds = np.arange(-2, 2.25, 1.0)
        else:
            boundary_thresholds = np.zeros(1)
        aps = np.zeros((cell_thresholds.size, boundary_thresholds.size, 3))
        for j,cell_threshold in enumerate(cell_thresholds):
            for k,boundary_threshold in enumerate(boundary_thresholds):
                masks = []
                for i in range(len(val_data)):
                    output,style = self._run_net(val_data[i].transpose(1,2,0), augment=False)
                    masks.append(utils.get_masks_unet(output, cell_threshold, boundary_threshold))
                ap = metrics.average_precision(val_labels, masks)[0]
                ap0 = ap.mean(axis=0)
                aps[j,k] = ap0
            if self.nclasses==3:
                kbest = aps[j].mean(axis=-1).argmax()
            else:
                kbest = 0
            if j%4==0:
                print('best threshold at cell_threshold = {} => boundary_threshold = {}, ap @ 0.5 = {}'.format(cell_threshold, boundary_thresholds[kbest], 
                                                                        aps[j,kbest,0]))   
        if self.nclasses==3: 
            jbest, kbest = np.unravel_index(aps.mean(axis=-1).argmax(), aps.shape[:2])
        else:
            jbest = aps.squeeze().mean(axis=-1).argmax()
            kbest = 0
        cell_threshold, boundary_threshold = cell_thresholds[jbest], boundary_thresholds[kbest]
        print('>>>> best overall thresholds: (cell_threshold = {}, boundary_threshold = {}); ap @ 0.5 = {}'.format(cell_threshold, boundary_threshold, 
                                                          aps[jbest,kbest,0]))
        return cell_threshold, boundary_threshold

    def _train_net(self, train_data, train_labels, 
              test_data=None, test_labels=None,
              pretrained_model=None, save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, weight_decay=0.00001, 
              batch_size=8, rescale=True, netstr='cellpose'):
        """ train function uses loss function self.loss_fn """

        d = datetime.datetime.now()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = 0.9

        nimg = len(train_data)

        # compute average cell diameter
        if rescale:
            diam_train = np.array([utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
            diam_train[diam_train<5] = 5.
            if test_data is not None:
                diam_test = np.array([utils.diameters(test_labels[k][0])[0] for k in range(len(test_labels))])
                diam_test[diam_test<5] = 5.
            scale_range = 0.5
        else:
            scale_range = 1.0

        nchan = train_data[0].shape[0]
        print('>>>> training network with %d channel input <<<<'%nchan)
        print('>>>> saving every %d epochs'%save_every)
        print('>>>> median diameter = %d'%self.diam_mean)
        print('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate, self.batch_size, self.weight_decay))
        print('>>>> ntrain = %d'%nimg)
        if test_data is not None:
            print('>>>> ntest = %d'%len(test_data))
        print(train_data[0].shape)

        trainer = gluon.Trainer(self.net.collect_params(), 'sgd',{'learning_rate': self.learning_rate,
                                'momentum': self.momentum, 'wd': self.weight_decay})

        eta = np.linspace(0, self.learning_rate, 10)
        tic = time.time()

        lavg, nsum = 0, 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            print('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg)
            if iepoch<len(eta):
                LR = eta[iepoch]
                trainer.set_learning_rate(LR)
            for ibatch in range(0,nimg,batch_size):
                if rescale:
                    diam_batch = diam_train[rperm[ibatch:ibatch+batch_size]]
                    rsc = diam_batch / self.diam_mean
                else:
                    rsc = np.ones(len(rperm[ibatch:ibatch+batch_size]), np.float32)

                imgi, lbl, scale = transforms.random_rotate_and_resize(
                                        [train_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                        Y=[train_labels[i][1:] for i in rperm[ibatch:ibatch+batch_size]],
                                        rescale=rsc, scale_range=scale_range, unet=self.unet)
                if self.unet and lbl.shape[1]>1 and rescale:
                    #lbl[:,1] *= scale[0]**2
                    lbl[:,1] /= diam_batch[:,np.newaxis,np.newaxis]**2
                X = nd.array(imgi, ctx=self.device)
                with mx.autograd.record():
                    y, style = self.net(X)
                    loss = self.loss_fn(lbl, y)

                loss.backward()
                train_loss = nd.sum(loss).asscalar()
                lavg += train_loss
                nsum+=len(loss)
                if iepoch>0:
                    trainer.step(batch_size)
            if iepoch>self.n_epochs-100 and iepoch%10==1:
                LR = LR/2
                trainer.set_learning_rate(LR)

            if iepoch%10==0 or iepoch<10:
                lavg = lavg / nsum
                if test_data is not None:
                    lavgt = 0
                    nsum = 0
                    np.random.seed(42)
                    rperm = np.arange(0, len(test_data), 1, int)
                    for ibatch in range(0,len(test_data),batch_size):
                        if rescale:
                            rsc = diam_test[rperm[ibatch:ibatch+batch_size]] / self.diam_mean
                        else:
                            rsc = np.ones(len(rperm[ibatch:ibatch+batch_size]), np.float32)
                        imgi, lbl, scale = transforms.random_rotate_and_resize(
                                            [test_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                            Y=[test_labels[i][1:] for i in rperm[ibatch:ibatch+batch_size]],
                                            scale_range=0., rescale=rsc, unet=self.unet)
                        if self.unet and lbl.shape[1]>1:
                            lbl[:,1] *= scale[0]**2
                        X    = nd.array(imgi, ctx=self.device)
                        y, style = self.net(X)
                        loss = self.loss_fn(lbl, y)
                        lavgt += nd.sum(loss).asscalar()
                        nsum+=len(loss)
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, lavgt/nsum, LR))
                else:
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, LR))
                lavg, nsum = 0, 0

            if save_path is not None:
                if iepoch==self.n_epochs-1 or iepoch%save_every==1:
                    # save model at the end
                    file = '{}_{}_{}'.format(self.net_type, file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                    ksave += 1
                    print('saving network parameters')
                    self.net.save_parameters(os.path.join(file_path, file))
        return os.path.join(file_path, file)

class CellposeModel(UnetModel):
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available

    pretrained_model: str or list of strings (optional, default False)
        path to pretrained cellpose model(s), if False, no model loaded;
        if None, built-in 'cyto' model loaded

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    
    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model

    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))

    """

    def __init__(self, gpu=False, pretrained_model=False,
                    diam_mean=30., net_avg=True, device=None,
                    residual_on=True, style_on=True, concatenation=False):
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        nclasses = 3 # 3 prediction maps (dY, dX and cellprob)
        self.nclasses = nclasses 
        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')
        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params
        # load default cyto model if pretrained_model is None
        elif pretrained_model is None:
            if net_avg:
                pretrained_model = [os.fspath(model_dir.joinpath('cyto_%d'%j)) for j in range(4)]
                if not os.path.isfile(pretrained_model[0]):
                    download_model_weights()
            else:
                pretrained_model = os.fspath(model_dir.joinpath('cyto_0'))
                if not os.path.isfile(pretrained_model):
                    download_model_weights()
            self.diam_mean = 30.
            residual_on = True 
            style_on = True 
            concatenation = False
        
        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         nclasses=nclasses)
        self.unet = False
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None and isinstance(self.pretrained_model, str):
            self.net.load_parameters(self.pretrained_model)

        ostr = ['off', 'on']
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])
        if pretrained_model:
            print(self.net_type)


    def eval(self, imgs, batch_size=8, channels=None, normalize=True, invert=False, rescale=None, 
             do_3D=False, anisotropy=None, net_avg=True, augment=False, tile=True, tile_overlap=0.1,
             resample=False, flow_threshold=0.4, cellprob_threshold=0.0, compute_masks=True, 
             min_size=15, stitch_threshold=0.0, progress=None):
        """
            segment list of images imgs, or 4D array - Z x nchan x Y x X

            Parameters
            ----------
            imgs: list or array of images
                can be list of 2D/3D/4D images, or array of 2D/3D images

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            tile_overlap: float (optional, default 0.1)
                fraction of overlap of tiles when computing flows

            resample: bool (optional, default False)
                run dynamics at original image size (will be slower but create more accurate boundaries)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            cellprob_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            stitch_threshold: float (optional, default 0.0)
                if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        x, nolist = convert_images(imgs.copy(), channels, do_3D, normalize, invert)
        
        nimg = len(x)
        self.batch_size = batch_size
        
        styles = []
        flows = []
        masks = []

        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)

        if nimg > 1:
            iterator = trange(nimg)
        else:
            iterator = range(nimg)

        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_parameters(self.pretrained_model[0])
            self.net.collect_params().grad_req = 'null'

        if not do_3D:
            flow_time = 0
            net_time = 0
            for i in iterator:
                img = x[i].copy()
                Ly,Lx = img.shape[:2]

                tic = time.time()
                shape = img.shape
                # rescale image for flow computation
                img = transforms.resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg, 
                                            augment=augment, tile=tile,
                                            tile_overlap=tile_overlap)
                net_time += time.time() - tic
                if progress is not None:
                    progress.setValue(55)
                styles.append(style)
                if compute_masks:
                    tic=time.time()
                    if resample:
                        y = transforms.resize_image(y, shape[-3], shape[-2])
                    cellprob = y[:,:,-1]
                    dP = y[:,:,:2].transpose((2,0,1))
                    niter = 1 / rescale[i] * 200
                    p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5., 
                                                niter=niter)
                    if progress is not None:
                        progress.setValue(65)
                    maski = dynamics.get_masks(p, iscell=(cellprob>cellprob_threshold),
                                                flows=dP, threshold=flow_threshold)
                    maski = utils.fill_holes_and_remove_small_masks(maski)
                    maski = transforms.resize_image(maski, shape[-3], shape[-2], 
                                                    interpolation=cv2.INTER_NEAREST)
                    if progress is not None:
                        progress.setValue(75)
                    #dP = np.concatenate((dP, np.zeros((1,dP.shape[1],dP.shape[2]), np.uint8)), axis=0)
                    flows.append([dx_to_circ(dP), dP, cellprob, p])
                    masks.append(maski)
                    flow_time += time.time() - tic
                else:
                    flows.append([None]*3)
                    masks.append([])
            print('time spent: running network %0.2fs; flow+mask computation %0.2f'%(net_time, flow_time))

            if stitch_threshold > 0.0 and nimg > 1 and all([m.shape==masks[0].shape for m in masks]):
                print('stitching %d masks using stitch_threshold=%0.3f to make 3D masks'%(nimg, stitch_threshold))
                masks = utils.stitch3D(np.array(masks), stitch_threshold=stitch_threshold)
        else:
            for i in iterator:
                tic=time.time()
                shape = x[i].shape
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy, 
                                         net_avg=net_avg, augment=augment, tile=tile, 
                                         tile_overlap=tile_overlap, progress=progress)
                cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1]
                dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]), 
                                axis=0) # (dZ, dY, dX)
                print('flows computed %2.2fs'%(time.time()-tic))
                # ** mask out values using cellprob to increase speed and reduce memory requirements **
                yout = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.)
                print('dynamics computed %2.2fs'%(time.time()-tic))
                maski = dynamics.get_masks(yout, iscell=(cellprob>cellprob_threshold))
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                print('masks computed %2.2fs'%(time.time()-tic))
                flow = np.array([dx_to_circ(dP[1:,i]) for i in range(dP.shape[1])])
                flows.append([flow, dP, cellprob, yout])
                masks.append(maski)
                styles.append(style)
        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]
        return masks, flows, styles

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        criterion  = gluon.loss.L2Loss()
        criterion2 = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        veci = 5. * nd.array(lbl[:,1:], ctx=self.device)
        lbl  = nd.array(lbl[:,0]>.5, ctx=self.device)
        loss = criterion(y[:,:-1] , veci) + criterion2(y[:,-1] , lbl)
        return loss

    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None, 
              save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, weight_decay=0.00001, batch_size=8, rescale=True):

        """ train network with images train_data 
        
            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            train_files: list of strings
                file names for images in train_data (to save flows for future runs)

            test_data: list of arrays (2D or 3D)
                images for testing

            test_labels: list of arrays (2D or 3D)
                labels for test_data, where 0=no masks; 1,2,...=mask labels; 
                can include flows as additional images
        
            test_files: list of strings
                file names for images in test_data (to save flows for future runs)

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            pretrained_model: string (default, None)
                path to pretrained_model to start from, if None it is trained from scratch

            save_path: string (default, None)
                where to save trained model, if None it is not saved

            save_every: int (default, 100)
                save network every [save_every] epochs

            learning_rate: float (default, 0.2)
                learning rate for training

            n_epochs: int (default, 500)
                how many times to go through whole training set during training

            weight_decay: float (default, 0.00001)

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            rescale: bool (default, True)
                whether or not to rescale images to diam_mean during training, 
                if True it assumes you will fit a size model after training or resize your images accordingly,
                if False it will try to train the model to be scale-invariant (works worse)


        """

        nimg = len(train_data)

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize)

        # check if train_labels have flows
        train_flows = dynamics.labels_to_flows(train_labels, files=train_files)
        if run_test:
            test_flows = dynamics.labels_to_flows(test_labels, files=test_files)
        else:
            test_flows = None
        
        model_path = self._train_net(train_data, train_flows, 
                                     test_data, test_flows,
                                     pretrained_model, save_path, save_every,
                                     learning_rate, n_epochs, weight_decay, batch_size, rescale)
        return model_path

class SizeModel():
    """ linear regression model for determining the size of objects in image
        used to rescale before input to cp_model
        uses styles from cp_model

        Parameters
        -------------------

        cp_model: UnetModel or CellposeModel
            model from which to get styles

        device: mxnet device (optional, default mx.cpu())
            where cellpose model is saved (mx.gpu() or mx.cpu())

        pretrained_size: str
            path to pretrained size model

    """
    def __init__(self, cp_model, device=mx.cpu(), pretrained_size=None, **kwargs):
        super(SizeModel, self).__init__(**kwargs)

        self.device = device
        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.diam_mean = self.cp.diam_mean
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params['diam_mean']
        if not hasattr(self.cp, 'pretrained_model'):
            raise ValueError('provided model does not have a pretrained_model')
        
    def eval(self, imgs=None, styles=None, channels=None, normalize=True, invert=False, augment=False, tile=True,
                batch_size=8, progress=None):
        """ use images imgs to produce style or use style input to predict size of objects in image

            Object size estimation is done in two steps:
            1. use a linear regression model to predict size from style in image
            2. resize image to predicted size and run CellposeModel to get output masks.
                Take the median object size of the predicted masks as the final predicted size.

            Parameters
            -------------------

            imgs: list or array of images (optional, default None)
                can be list of 2D/3D images, or array of 2D/3D images

            styles: list or array of styles (optional, default None)
                styles for images x - if x is None then styles must not be None

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            Returns
            -------
            diam: array, float
                final estimated diameters from images x or styles style after running both steps

            diam_style: array, float
                estimated diameters from style alone

        """
        if styles is None and imgs is None:
            raise ValueError('no image or features given')
        
        if progress is not None:
            progress.setValue(10)
        
        if imgs is not None:
            x, nolist = convert_images(imgs.copy(), channels, False, normalize, invert)
            nimg = len(x)
        
        if styles is None:
            styles = self.cp.eval(x, net_avg=False, augment=augment, tile=tile, compute_masks=False)[-1]
            if progress is not None:
                progress.setValue(30)
            diam_style = self._size_estimation(np.array(styles))
            if progress is not None:
                progress.setValue(50)
        else:
            if isinstance(styles, list):
                styles = np.array(styles)
            diam_style = self._size_estimation(styles)
        diam_style[np.isnan(diam_style)] = self.diam_mean

        if imgs is not None:
            masks = self.cp.eval(x, rescale=self.diam_mean/diam_style, net_avg=False, 
                                augment=augment, tile=tile)[0]
            diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
            if progress is not None:
                progress.setValue(100)
            if hasattr(self, 'model_type') and (self.model_type=='nuclei' or self.model_type=='cyto'):
                diam_style /= (np.pi**0.5)/2
                diam[diam==0] = self.diam_mean / ((np.pi**0.5)/2)
                diam[np.isnan(diam)] = self.diam_mean / ((np.pi**0.5)/2)
            else:
                diam[diam==0] = self.diam_mean
                diam[np.isnan(diam)] = self.diam_mean
        else:
            diam = diam_style
            print('no images provided, using diameters estimated from styles alone')
        if nolist:
            return diam[0], diam_style[0]
        else:
            return diam, diam_style

    def _size_estimation(self, style):
        """ linear regression from style to size 
        
            sizes were estimated using "diameters" from square estimates not circles; 
            therefore a conversion factor is included (to be removed)
        
        """
        szest = np.exp(self.params['A'] @ (style - self.params['smean']).T +
                        np.log(self.diam_mean) + self.params['ymean'])
        szest = np.maximum(5., szest)
        return szest

    def train(self, train_data, train_labels,
              test_data=None, test_labels=None,
              channels=None, normalize=True, 
              learning_rate=0.2, n_epochs=10, 
              l2_regularization=1.0, batch_size=8):
        """ train size model with images train_data to estimate linear model from styles to diameters
        
            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            n_epochs: int (default, 10)
                how many times to go through whole training set (taking random patches) for styles for diameter estimation

            l2_regularization: float (default, 1.0)
                regularize linear model from styles to diameters

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)
        """
        self.cp.batch_size = batch_size
        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize)
        if isinstance(self.cp.pretrained_model, list) and len(self.cp.pretrained_model)>1:
            cp_model_path = self.cp.pretrained_model[0]
            self.cp.net.load_parameters(cp_model_path)
            self.cp.net.collect_params().grad_req = 'null'
        else:
            cp_model_path = self.cp.pretrained_model

        diam_train = np.array([utils.diameters(lbl)[0] for lbl in train_labels])
        if run_test: 
            diam_test = np.array([utils.diameters(lbl)[0] for lbl in test_labels])
        
        nimg = len(train_data)
        styles = np.zeros((n_epochs*nimg, 256), np.float32)
        diams = np.zeros((n_epochs*nimg,), np.float32)
        tic = time.time()
        for iepoch in range(n_epochs):
            iall = np.arange(0,nimg,1,int)
            for ibatch in range(0,nimg,batch_size):
                inds = iall[ibatch:ibatch+batch_size]
                imgi,lbl,scale = transforms.random_rotate_and_resize(
                            [train_data[i] for i in inds],
                            Y=[train_labels[i].astype(np.int16) for i in inds], scale_range=1, xy=(512,512))
                feat = self.cp.net(nd.array(imgi, ctx=self.device))[-1].asnumpy()
                styles[inds+nimg*iepoch] = feat
                diams[inds+nimg*iepoch] = np.log(diam_train[inds]) - np.log(self.diam_mean) + np.log(scale)
            del feat
            if (iepoch+1)%2==0:
                print('ran %d epochs in %0.3f sec'%(iepoch+1, time.time()-tic))

        # create model
        smean = styles.mean(axis=0)
        X = ((styles - smean).T).copy()
        ymean = diams.mean()
        y = diams - ymean

        A = np.linalg.solve(X@X.T + l2_regularization*np.eye(X.shape[0]), X @ y)
        ypred = A @ X
        print('train correlation: %0.4f'%np.corrcoef(y, ypred)[0,1])
            
        if run_test:
            nimg_test = len(test_data)
            styles_test = np.zeros((nimg_test, 256), np.float32)
            for i in range(nimg_test):
                styles_test[i] = self.cp._run_net(test_data[i].transpose((1,2,0)))[1]
            diam_test_pred = np.exp(A @ (styles_test - smean).T + np.log(self.diam_mean) + ymean)
            diam_test_pred = np.maximum(5., diam_test_pred)
            print('test correlation: %0.4f'%np.corrcoef(diam_test, diam_test_pred)[0,1])

        self.pretrained_size = cp_model_path+'_size.npy'
        self.params = {'A': A, 'smean': smean, 'diam_mean': self.diam_mean, 'ymean': ymean}
        np.save(self.pretrained_size, self.params)
        return self.params

def convert_images(x, channels, do_3D, normalize, invert):
    """ return list of images with channels last and normalized intensities """
    if not isinstance(x,list) and not (x.ndim>3 and not do_3D):
        nolist = True
        x = [x]
    else:
        nolist = False
    
    nimg = len(x)
    if do_3D:
        for i in range(len(x)):
            if x[i].ndim<3:
                raise ValueError('ERROR: cannot process 2D images in 3D mode') 
            elif x[i].ndim<4:
                x[i] = x[i][...,np.newaxis]
            if x[i].shape[1]<4:
                x[i] = x[i].transpose((0,2,3,1))
            elif x[i].shape[0]<4:
                x[i] = x[i].transpose((1,2,3,0))
            print('multi-stack tiff read in as having %d planes %d channels'%
                    (x[i].shape[0], x[i].shape[-1]))

    if channels is not None:
        if len(channels)==2:
            if not isinstance(channels[0], list):
                channels = [channels for i in range(nimg)]
        x = [transforms.reshape(x[i], channels=channels[i]) for i in range(nimg)]
    elif do_3D:
        for i in range(len(x)):
            # code above put channels last
            if x[i].shape[-1]>2:
                print('WARNING: more than 2 channels given, use "channels" input for specifying channels - just using first two channels to run processing')
                x[i] = x[i][...,:2]
    else:
        for i in range(len(x)):
            if x[i].ndim>3:
                raise ValueError('ERROR: cannot process 4D images in 2D mode')
            elif x[i].ndim==2:
                x[i] = np.stack((x[i], np.zeros_like(x[i])), axis=2)
            elif x[i].shape[0]<8:
                x[i] = x[i].transpose((1,2,0))
            if x[i].shape[-1]>2:
                print('WARNING: more than 2 channels given, use "channels" input for specifying channels - just using first two channels to run processing')
                x[i] = x[i][:,:,:2]

    if normalize or invert:
        x = [transforms.normalize_img(x[i], invert=invert) for i in range(nimg)]
    return x, nolist

urls = ['http://www.cellpose.org/models/cyto_0',
        'http://www.cellpose.org/models/cyto_1',
        'http://www.cellpose.org/models/cyto_2',
        'http://www.cellpose.org/models/cyto_3',
        'http://www.cellpose.org/models/size_cyto_0.npy',
        'http://www.cellpose.org/models/nuclei_0',
        'http://www.cellpose.org/models/nuclei_1',
        'http://www.cellpose.org/models/nuclei_2',
        'http://www.cellpose.org/models/nuclei_3',
        'http://www.cellpose.org/models/size_nuclei_0.npy']


def download_model_weights(urls=urls):
    # cellpose directory
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    model_dir = cp_dir.joinpath('models')
    model_dir.mkdir(exist_ok=True)

    for url in urls:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            utils.download_url_to_file(url, cached_file, progress=True)
