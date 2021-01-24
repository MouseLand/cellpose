import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2

from . import transforms, dynamics, utils, plot, metrics, core
from .core import UnetModel, assign_device, check_mkl, use_gpu, convert_images, MXNET_ENABLED, parse_model_string

urls = ['https://www.cellpose.org/models/cyto_0',
        'https://www.cellpose.org/models/cyto_1',
        'https://www.cellpose.org/models/cyto_2',
        'https://www.cellpose.org/models/cyto_3',
        'https://www.cellpose.org/models/size_cyto_0.npy',
        'https://www.cellpose.org/models/cytotorch_0',
        'https://www.cellpose.org/models/cytotorch_1',
        'https://www.cellpose.org/models/cytotorch_2',
        'https://www.cellpose.org/models/cytotorch_3',
        'https://www.cellpose.org/models/size_cytotorch_0.npy',
        'https://www.cellpose.org/models/nuclei_0',
        'https://www.cellpose.org/models/nuclei_1',
        'https://www.cellpose.org/models/nuclei_2',
        'https://www.cellpose.org/models/nuclei_3',
        'https://www.cellpose.org/models/size_nuclei_0.npy',
        'https://www.cellpose.org/models/nucleitorch_0',
        'https://www.cellpose.org/models/nucleitorch_1',
        'https://www.cellpose.org/models/nucleitorch_2',
        'https://www.cellpose.org/models/nucleitorch_3',
        'https://www.cellpose.org/models/size_nucleitorch_0.npy']


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

download_model_weights()
model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')

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
        whether or not to use GPU, will check if GPU available

    model_type: str (optional, default 'cyto')
        'cyto'=cytoplasm model; 'nuclei'=nucleus model

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    device: gpu device (optional, default None)
        where model is saved (e.g. mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4) or torch.cuda.device(4))

    torch: bool (optional, default False)
        run model using torch if available

    """
    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, torch=True):
        super(Cellpose, self).__init__()
        if not torch:
            if not MXNET_ENABLED:
                print('WARNING: mxnet not installed, using torch')
                torch = True
        self.torch = torch
        torch_str = ['','torch'][self.torch]
        
        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        model_type = 'cyto' if model_type is None else model_type
        self.pretrained_model = [os.fspath(model_dir.joinpath('%s%s_%d'%(model_type,torch_str,j))) for j in range(4)]
        self.pretrained_size = os.fspath(model_dir.joinpath('size_%s%s_0.npy'%(model_type,torch_str)))
        self.diam_mean = 30. if model_type=='cyto' else 17.
        
        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]

        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean, torch=self.torch)
        self.cp.model_type = model_type

        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=None, invert=False, normalize=True, diameter=30., do_3D=False, anisotropy=None,
             net_avg=True, augment=False, tile=True, tile_overlap=0.1, resample=False, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0, min_size=15, 
              stitch_threshold=0.0, rescale=None, progress=None):
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
            First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
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

        interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)

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
        if diameter is not None and not (not isinstance(diameter, (list, np.ndarray)) and 
                (diameter==0 or (diameter==30. and rescale is not None))):    
            if not isinstance(diameter, (list, np.ndarray)) or len(diameter)==1 or len(diameter)<nimg:
                diams = diameter * np.ones(nimg, np.float32)
            else:
                diams = diameter
            rescale = self.diam_mean / diams
        else:
            if rescale is not None and (not isinstance(rescale, (list, np.ndarray)) or len(rescale)==1):
                rescale = rescale * np.ones(nimg, np.float32)
            if self.pretrained_size is not None and rescale is None and not do_3D:
                tic = time.time()
                diams, _ = self.sz.eval(x, channels=channels, invert=invert, batch_size=batch_size, 
                                        augment=augment, tile=tile)
                rescale = self.diam_mean / diams
                print('estimated cell diameters for %d image(s) in %0.2f sec'%(nimg, time.time()-tic))
                print('>>> diameter(s) = ', diams)
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
                                            interp=interp,
                                            flow_threshold=flow_threshold, 
                                            cellprob_threshold=cellprob_threshold,
                                            min_size=min_size, 
                                            stitch_threshold=stitch_threshold)
        print('estimated masks for %d image(s) in %0.2f sec'%(nimg, time.time()-tic))
        print('>>>> TOTAL TIME %0.2f sec'%(time.time()-tic0))
        
        if nolist:
            masks, flows, styles, diams = masks[0], flows[0], styles[0], diams[0]
        
        return masks, flows, styles, diams

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

    def __init__(self, gpu=False, pretrained_model=False, torch=True,
                    diam_mean=30., net_avg=True, device=None,
                    residual_on=True, style_on=True, concatenation=False):
        if not torch:
            if not MXNET_ENABLED:
                print('WARNING: mxnet not installed, using torch')
                torch = True
        self.torch = torch
        
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        nclasses = 3 # 3 prediction maps (dY, dX and cellprob)
        self.nclasses = nclasses 
        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params
        # load default cyto model if pretrained_model is None
        elif pretrained_model is None:
            torch_str = ['','torch'][self.torch]
            pretrained_model = [os.fspath(model_dir.joinpath('cyto%s_%d'%(torch_str,j))) for j in range(4)] if net_avg else os.fspath(model_dir.joinpath('cyto_0'))
            self.diam_mean = 30.
            residual_on, style_on, concatenation = True, True, False
        
        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         nclasses=nclasses, torch=torch)
        self.unet = False
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None and isinstance(self.pretrained_model, str):
            self.net.load_model(self.pretrained_model, cpu=(not self.gpu))
        ostr = ['off', 'on']
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])
                                                                                
    def eval(self, imgs, batch_size=8, channels=None, normalize=True, invert=False, 
             rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=True, 
             augment=False, tile=True, tile_overlap=0.1,
             resample=False, interp=True, flow_threshold=0.4, cellprob_threshold=0.0, compute_masks=True, 
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
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            diameter: float (optional, default None)
                diameter for each image (only used if rescale is None), 
                if diameter is None, set to diam_mean

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

            interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)

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
            if diameter is not None:
                if not isinstance(diameter, (list, np.ndarray)):
                    diameter = diameter * np.ones(nimg)
                rescale = self.diam_mean / diameter
            else:
                rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)

        iterator = trange(nimg) if nimg>1 else range(nimg)
        
        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
            if not self.torch:
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
                if resample:
                    y = transforms.resize_image(y, shape[-3], shape[-2])
                cellprob = y[:,:,-1]
                dP = y[:,:,:2].transpose((2,0,1))
                if compute_masks:
                    tic=time.time()
                    niter = 1 / rescale[i] * 200
                    p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5., 
                                                niter=niter, interp=interp, use_gpu=self.gpu)
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
                    flows.append([dx_to_circ(dP), dP, cellprob, []])
                    masks.append([])
            if compute_masks:
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
        
        veci = 5. * self._to_device(lbl[:,1:])
        lbl  = self._to_device(lbl[:,0]>.5)
        loss = self.criterion(y[:,:2] , veci) 
        if self.torch:
            loss /= 2.
        loss2 = self.criterion2(y[:,-1] , lbl)
        loss = loss + loss2
        return loss


    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None, 
              save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8, rescale=True):

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
                                     learning_rate, n_epochs, momentum, weight_decay, batch_size, rescale)
        self.pretrained_model = model_path
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
    def __init__(self, cp_model, device=None, pretrained_size=None, **kwargs):
        super(SizeModel, self).__init__(**kwargs)

        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.device = self.cp.device
        self.diam_mean = self.cp.diam_mean
        self.torch = self.cp.torch
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
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
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
            print('computing styles from images')
            styles = self.cp.eval(x, net_avg=False, augment=augment, tile=tile, compute_masks=False)[-1]
            if progress is not None:
                progress.setValue(30)
            diam_style = self._size_estimation(np.array(styles))
            if progress is not None:
                progress.setValue(50)
        else:
            styles = np.array(styles) if isinstance(styles, list) else styles
            diam_style = self._size_estimation(styles)
        diam_style[np.isnan(diam_style)] = self.diam_mean

        if imgs is not None:
            masks = self.cp.eval(x, rescale=self.diam_mean/diam_style, net_avg=False, 
                                augment=augment, tile=tile, interp=False)[0]
            diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
            if progress is not None:
                progress.setValue(100)
            if hasattr(self, 'model_type') and (self.model_type=='nuclei' or self.model_type=='cyto') and not self.torch:
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
        batch_size /= 2 # reduce batch_size by factor of 2 to use larger tiles
        batch_size = int(max(1, batch_size))
        self.cp.batch_size = batch_size
        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize)
        if isinstance(self.cp.pretrained_model, list) and len(self.cp.pretrained_model)>1:
            cp_model_path = self.cp.pretrained_model[0]
            self.cp.net.load_model(cp_model_path, cpu=(not self.gpu))
            if not self.torch:
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
                feat = self.cp.network(imgi)[-1]
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
