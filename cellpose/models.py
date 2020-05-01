import os, sys, time, shutil, tempfile, datetime, pathlib, gc
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2

from mxnet import gluon, nd
import mxnet as mx

from . import transforms, dynamics, utils, resnet_style, plot
import __main__

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

    batch_size: int (optional, default 8)
        number of 224x224 patches to run simultaneously on the GPU 
        (can make smaller or bigger depending on GPU memory usage)

    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))

    """
    def __init__(self, gpu=False, model_type='cyto', net_avg=True, batch_size=8, device=None):
        super(Cellpose, self).__init__()
        # assign device (GPU or CPU)
        if device is not None:
            self.device = device
        elif gpu and utils.use_gpu():
            self.device = mx.gpu()
            print('>>>> using GPU')
        else:
            self.device = mx.cpu()
            print('>>>> using CPU')
        
        self.batch_size=batch_size
        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')
        if model_type is None:
            model_type = 'cyto'
        
        self.pretrained_model = [os.fspath(model_dir.joinpath('%s_%d'%(model_type,j))) for j in range(4)]
        self.pretrained_size = os.fspath(model_dir.joinpath('size_%s_0.npy'%(model_type)))
        if model_type=='cyto':
            self.diam_mean = 27.
        else:
            self.diam_mean = 15.
        if not os.path.isfile(self.pretrained_model[0]):
            download_model_weights()
        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]
        
        self.cp = CellposeModel(device=self.device,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean)
        
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)

    def eval(self, x, channels=None, diameter=30., invert=False, do_3D=False,
             net_avg=True, tile=True, flow_threshold=0.4, cellprob_threshold=0.0,
             rescale=None, progress=None):
        """ run cellpose and get masks

        Parameters
        ----------
        x: list or array of images
            can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

        channels: list (optional, default None) 
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
            For instance, to segment grayscale images, input [0,0]. To segment images with cells
            in green and nuclei in blue, input [2,3]. To segment one grayscale image and one 
            image with cells in green and nuclei in blue, input [[0,0], [2,3]].

        diameter: float (optional, default 30.)
            if set to None, then diameter is automatically estimated if size model is loaded

        invert: bool (optional, default False)
            invert image pixel intensity before running network

        do_3D: bool (optional, default False)
            set to True to run 3D segmentation on 4D image input

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        tile: bool (optional, default True) 
            tiles image for test time augmentation and to ensure GPU memory usage limited (recommended)

        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

        cellprob_threshold: float (optional, default 0.0)
            cell probability threshold (all pixels with prob above threshold kept for masks)

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
            x = [x]
        else:
            nolist = False

        if do_3D:
            for i in range(len(x)):
                if x[i].ndim<3:
                    raise ValueError('ERROR: cannot process 2D images in 3D mode') 
                elif x[i].ndim<4:
                    x[i] = x[i][...,np.newaxis]
                if x[i].shape[1]<4:
                    x[i] = np.transpose(x[i], (0,2,3,1))
            print('multi-stack tiff read in as having is %d planes %d channels'%
                    (x[0].shape[0], x[0].shape[-1]))

        print('processing %d images'%len(x))
        # make rescale into length of x
        if diameter is not None and diameter!=0:
            if not isinstance(diameter, list) or len(diameter)==1 or len(diameter)<len(x):
                diams = diameter * np.ones(len(x), np.float32)
            else:
                diams = diameter
            rescale = self.diam_mean / (diams.copy() * (np.pi**0.5/2))
        else:
            if rescale is not None and (not isinstance(rescale, list) or len(rescale)==1):
                rescale = rescale * np.ones(len(x), np.float32)
            if self.pretrained_size is not None and rescale is None and not do_3D:
                diams, diams_style = self.sz.eval(x, channels=channels, invert=invert, batch_size=self.batch_size, tile=tile)
                rescale = self.diam_mean / diams.copy()
                diams /= (np.pi**0.5/2) # convert to circular
                print('estimated cell diameters for all images')
            else:
                if rescale is None:
                    if do_3D:
                        rescale = 1.0
                    else:
                        rescale = np.ones(len(x), np.float32)
                diams = self.diam_mean / rescale.copy() / (np.pi**0.5/2)
            
        masks, flows, styles = self.cp.eval(x, invert=invert, rescale=rescale, channels=channels, tile=tile,
                                            do_3D=do_3D, net_avg=net_avg, progress=progress,
                                            flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
        if nolist:
            masks, flows, styles, diams = masks[0], flows[0], styles[0], diams[0]
        
        return masks, flows, styles, diams

class CellposeModel():
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

    batch_size: int (optional, default 8)
        number of 224x224 patches to run simultaneously on the GPU 
        (can make smaller or bigger depending on GPU memory usage)

    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model

    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))

    """
    
    def __init__(self, gpu=False, pretrained_model=False, batch_size=8,
                    diam_mean=27., net_avg=True, device=None, unet=False):
        super(CellposeModel, self).__init__()
        
        if device is not None:
            self.device = device
        elif gpu and utils.use_gpu():
            self.device = mx.gpu()
            print('>>>> using GPU')
        else:
            self.device = mx.cpu()
            print('>>>> using CPU')

        self.unet = unet
        if unet:
            nout = 1
        else:
            nout = 3

        self.pretrained_model = pretrained_model
        self.batch_size=batch_size
        self.diam_mean = diam_mean

        nbase = [32,64,128,256]
        self.net = resnet_style.CPnet(nbase, nout=nout)
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.initialize(ctx = self.device)#, grad_req='null')

        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')

        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_parameters(pretrained_model)
        elif pretrained_model is None and not unet:
            if net_avg:
                pretrained_model = [os.fspath(model_dir.joinpath('cyto_%d'%j)) for j in range(4)]
                if not os.path.isfile(pretrained_model[0]):
                    download_model_weights()
            else:
                pretrained_model = os.fspath(model_dir.joinpath('cyto_0'))
                if not os.path.isfile(pretrained_model):
                    download_model_weights()
                self.net.load_parameters(pretrained_model)
            self.diam_mean = 27.
            self.pretrained_model = pretrained_model

    def eval(self, x, channels=None, invert=False, rescale=None, do_3D=False, net_avg=True, 
             tile=True, flow_threshold=0.4, cellprob_threshold=0.0, compute_masks=True, progress=None):
        """
            segment list of images x, or 4D array - Z x nchan x Y x X
        
            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

            channels: list (optional, default None) 
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one 
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            tile: bool (optional, default True) 
                tiles image for test time augmentation and to ensure GPU memory usage limited (recommended)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            cellprob_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

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
        nimg = len(x)
        if channels is not None:
            if len(channels)==2:
                if not isinstance(channels[0], list):
                    channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i], invert=invert) for i in range(nimg)]
        elif do_3D:
            x = [np.transpose(x[i], (3,0,1,2)) for i in range(len(x))]
            
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
            for i in iterator:
                img = x[i].copy()
                if img.shape[0]<3:
                    img = np.transpose(img, (1,2,0))
                Ly,Lx = img.shape[:2]
                if img.shape[-1]==1:
                    img = np.concatenate((img, 0.*img), axis=-1)
                #tic=time.time()
                if isinstance(self.pretrained_model, str) or not net_avg:
                    y, style = self._run_net(img, rescale[i], tile)
                else:
                    y, style = self._run_many(img, rescale[i], tile)
                if progress is not None:
                    progress.setValue(55)
                styles.append(style)
                if compute_masks:
                    cellprob = y[...,-1]
                    if not self.unet:
                        dP = np.stack((y[...,0], y[...,1]), axis=0)
                        niter = 1 / rescale[i] * 200
                        p = dynamics.follow_flows(-1 * dP  / 5. , niter=niter)
                        if progress is not None:
                            progress.setValue(65)
                        maski = dynamics.get_masks(p, iscell=(cellprob>cellprob_threshold),
                                                   flows=dP, threshold=flow_threshold)
                        if progress is not None:
                            progress.setValue(75)
                        dZ = np.zeros((1,Ly,Lx), np.uint8)
                        dP = np.concatenate((dP, dZ), axis=0)
                        flow = plot.dx_to_circ(dP)
                        flows.append([flow, dP, cellprob, p])
                        maski = dynamics.fill_holes(maski)
                        masks.append(maski)
                else:
                    flows.append([None]*3)
                    masks.append([])
        else:
            for i in iterator:
                sstr = ['XY', 'XZ', 'YZ']
                if x[i].shape[-1] < 3:
                    x[i] = np.transpose(x[i], (3,0,1,2))
                pm = [(1,2,3,0), (2,1,3,0), (3,1,2,0)]
                ipm = [(0,1,2,3), (0,2,1,3), (0,2,3,1)]
                tic=time.time()
                flowi=[]
                for p in range(3):
                    xsl = np.transpose(x[i].copy(), pm[p])
                    print(xsl.shape)
                    flowi.append(np.zeros(((3,xsl.shape[0],xsl.shape[1],xsl.shape[2])), np.float32))
                    # per image
                    ziterator = trange(xsl.shape[0])
                    print('running %s (%d, %d)\n'%(sstr[p], xsl.shape[1], xsl.shape[2]))
                    for z in ziterator:
                        if isinstance(self.pretrained_model, str) or not net_avg:
                            y, style = self._run_net(xsl[z], rescale[0], tile=tile)
                        else:
                            y, style = self._run_many(xsl[z], rescale[0], tile=tile)
                        y = np.transpose(y[:,:,[1,0,2]], (2,0,1))
                        flowi[p][:,z] = y
                    flowi[p] = np.transpose(flowi[p], ipm[p])
                    if progress is not None:
                        progress.setValue(25+15*p)
                dX = flowi[0][0] + flowi[1][0]
                dY = flowi[0][1] + flowi[2][0]
                dZ = flowi[1][1] + flowi[2][1]
                cellprob = flowi[0][-1] + flowi[1][-1] + flowi[2][-1]
                dP = np.concatenate((dZ[np.newaxis,...], dY[np.newaxis,...], dX[np.newaxis,...]), axis=0)
                print('flows computed %2.2fs'%(time.time()-tic))
                yout = dynamics.follow_flows(-1 * dP / 5.)
                print('dynamics computed %2.2fs'%(time.time()-tic))
                maski = dynamics.get_masks(yout, iscell=(cellprob>cellprob_threshold))
                print('masks computed %2.2fs'%(time.time()-tic))
                flow = np.array([plot.dx_to_circ(dP[1:,i]) for i in range(dP.shape[1])])
                flows.append([flow, dP, cellprob, yout])
                masks.append(maski)
                styles.append([])
        return masks, flows, styles

    def _run_many(self, img, rsz=1.0, tile=True):
        """ loop over netwroks in pretrained_model and average results

        Parameters
        --------------

        img: float, [Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient for image

        tile: bool (optional, default True)
            tiles image for test time augmentation and to ensure GPU memory usage limited (recommended)
            
        Returns
        ------------------

        yup: array [3 x Ly x Lx]
            yup is output averaged over networks;
            yup[0] is Y flow; yup[1] is X flow; yup[2] is cell probability

        style: array [64]
            1D array summarizing the style of the image, 
            if tiled it is averaged over tiles, 
            but not averaged over networks.

        """
        for j in range(len(self.pretrained_model)):
            self.net.load_parameters(self.pretrained_model[j])
            self.net.collect_params().grad_req = 'null'
            yup0, style = self._run_net(img, rsz, tile)
            if j==0:
                yup = yup0
            else:
                yup += yup0
        yup = yup / len(self.pretrained_model)
        return yup, style

    def _run_tiled(self, imgi, bsize=224):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].        
        Then 4 versions of each tile are created:
            * original
            * flipped vertically
            * flipped horizontally
            * flipped vertically and horizontally
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx]

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        yf: array [3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles
            
        """
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize, augment=True)
        IMG = nd.array(IMG, ctx=self.device)
        nbatch = self.batch_size
        niter = int(np.ceil(IMG.shape[0]/nbatch))
        y = np.zeros((IMG.shape[0], 3, bsize, bsize))
        for k in range(niter):
            irange = np.arange(nbatch*k, min(IMG.shape[0], nbatch*k+nbatch))
            y0, style = self.net(IMG[irange])
            y[irange] = y0.asnumpy()
            if k==0:
                styles = style.asnumpy()[0]
            styles += style.asnumpy().sum(axis=0)
        styles /= IMG.shape[0]
        y = transforms.unaugment_tiles(y)
        yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
        styles /= (styles**2).sum()**0.5
        del IMG 
        gc.collect()
        return yf, styles

    def _run_net(self, img, rsz=1.0, tile=True, bsize=224):
        """ run network on image

        Parameters
        --------------

        img: array [Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient for image

        tile: bool (optional, default True)
            tiles image for test time augmentation and to ensure GPU memory usage limited (recommended)

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        y: array [3 x Ly x Lx]
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability

        style: array [64]
            1D array summarizing the style of the image, 
            if tiled it is averaged over tiles
            
        """
        shape = img.shape
        if abs(rsz - 1.0) < 0.03:
            rsz = 1.0
            Ly,Lx = img.shape[:2]
        else:
            Ly = int(img.shape[0] * rsz)
            Lx = int(img.shape[1] * rsz)
            img = cv2.resize(img, (Lx, Ly))

        # make image nchan x Ly x Lx for net
        if img.ndim<3:
            img = np.expand_dims(img, axis=-1)
        img = np.transpose(img, (2,0,1))
        
        # pad for net so divisible by 4
        img, ysub, xsub = transforms.pad_image_ND(img)
        if tile:
            y,style = self._run_tiled(img, bsize)
            y = np.transpose(y[:3], (1,2,0))
        else:
            img = nd.array(np.expand_dims(img, axis=0), ctx=self.device)
            y,style = self.net(img)
            img = img.asnumpy()
            y = np.transpose(y[0].asnumpy(), (1,2,0))
            style = style.asnumpy()[0]
            style = np.ones(10)
        
        y = y[np.ix_(ysub, xsub, np.arange(3))]
        style /= (style**2).sum()**0.5     
        if rsz!=1.0:
            y = cv2.resize(y, (shape[1], shape[0]))
        return y, style

    def train(self, train_data, train_labels, test_data=None, test_labels=None, channels=None,
              pretrained_model=None, save_path=None, save_every=100, 
              learning_rate=0.2, n_epochs=500, weight_decay=0.00001, batch_size=8, rescale=True):

        d = datetime.datetime.now()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = 0.9
        
        nimg = len(train_data)
        
        # check that arrays are correct size
        if nimg != len(train_labels):
            raise ValueError('train data and labels not same length')
            return
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            raise ValueError('training data or labels are not at least two-dimensional')
            return

        # check if test_data correct length
        if not (test_data is not None and test_labels is not None and 
                len(test_data) > 0 and len(test_data)==len(test_labels)):
            test_data = None
        
        # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
        train_data, test_data, run_test = transforms.reshape_data(train_data, test_data=test_data, channels=channels)
        if train_data is None:
            raise ValueError('training data do not all have the same number of channels')
            return
        nchan = train_data[0].shape[0]
        
        if not run_test:
            print('NOTE: test data not provided OR labels incorrect OR not same number of channels as train data')        

        # check if train_labels have flows
        if not self.unet:
            train_flows = dynamics.labels_to_flows(train_labels)
            if run_test:
                test_flows = dynamics.labels_to_flows(test_labels)
        else:
            train_flows = list(map(np.uint16, train_labels))
            test_flows = list(map(np.uint16, test_labels))
                
        # compute average cell diameter
        if rescale:
            diam_train = np.array([utils.diameters(train_labels[k])[0] for k in range(len(train_labels))])
            diam_train[diam_train<5] = 5.
            if run_test:
                diam_test = np.array([utils.diameters(test_labels[k])[0] for k in range(len(test_labels))])
                diam_test[diam_test<5] = 5.
            scale_range = 0.5
        else:
            scale_range = 1.0

        print('>>>> training network with %d channel input <<<<'%nchan)
        print('>>>> saving every %d epochs'%save_every)
        print('>>>> median diameter = %d'%self.diam_mean)
        print('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate, self.batch_size, self.weight_decay))
        print('>>>> ntrain = %d'%nimg)
        if run_test:
            print('>>>> ntest = %d'%len(test_data))
        print(train_data[0].shape)

        criterion  = gluon.loss.L2Loss()
        criterion2 = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        trainer = gluon.Trainer(self.net.collect_params(), 'sgd',{'learning_rate': self.learning_rate,
                                'momentum': self.momentum, 'wd': self.weight_decay})

        eta = np.linspace(0, self.learning_rate, 10)
        tic = time.time()

        lavg, nsum = 0, 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')
        else:
            print('WARNING: no save_path given, model not saving')
        ksave = 0
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        rsc = 1.0
        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg)
            if iepoch<len(eta):
                LR = eta[iepoch]
                trainer.set_learning_rate(LR)
            for ibatch in range(0,nimg,batch_size):
                if rescale:
                    rsc = diam_train[rperm[ibatch:ibatch+batch_size]] / self.diam_mean
                else:
                    rsc = np.ones(len(rperm[ibatch:ibatch+batch_size]), np.float32)
                
                imgi, lbl, _ = transforms.random_rotate_and_resize(
                                        [train_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                        Y=[train_flows[i] for i in rperm[ibatch:ibatch+batch_size]],
                                        rescale=rsc, scale_range=scale_range)
                X    = nd.array(imgi, ctx=self.device)
                if not self.unet:
                    veci = 5. * nd.array(lbl[:,1:], ctx=self.device)
                lbl  = nd.array(lbl[:,0]>.5, ctx=self.device)
                with mx.autograd.record():
                    y, style = self.net(X)
                    if self.unet:
                        loss = criterion2(y[:,-1] , lbl)
                    else:
                        loss = criterion(y[:,:-1] , veci) + criterion2(y[:,-1] , lbl)

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
                if run_test:
                    lavgt = 0
                    nsum = 0
                    np.random.seed(42)
                    rperm = np.arange(0, len(test_data), 1, int)
                    for ibatch in range(0,len(test_data),batch_size):
                        if rescale:
                            rsc = diam_test[rperm[ibatch:ibatch+batch_size]] / self.diam_mean
                        else:
                            rsc = np.ones(len(rperm[ibatch:ibatch+batch_size]), np.float32)
                        imgi, lbl, _ = transforms.random_rotate_and_resize(
                                            [test_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                            Y=[test_flows[i] for i in rperm[ibatch:ibatch+batch_size]],
                                            scale_range=0., rescale=rsc)
                        X    = nd.array(imgi, ctx=self.device)
                        if not self.unet:
                            veci = 5. * nd.array(lbl[:,1:], ctx=self.device)
                        lbl  = nd.array(lbl[:,0]>.5, ctx=self.device)
                        y, style = self.net(X)
                        if self.unet:
                            loss = criterion2(y[:,-1] , lbl)
                        else:
                            loss = criterion(y[:,:-1] , veci) + criterion2(y[:,-1] , lbl)
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
                    file = 'cellpose_{}_{}_{}'.format(self.unet, file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))                       
                    ksave += 1
                    print('saving network parameters')
                    self.net.save_parameters(os.path.join(file_path, file))

class SizeModel():
    """ linear regression model for determining the size of objects in image
        used to rescale before input to CellposeModel
        uses styles from CellposeModel 

        Parameters
        -------------------

        cp_model: CellposeModel
            cellpose model from which to get styles

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

    def eval(self, x=None, style=None, channels=None, invert=False, tile=True,
                batch_size=8, progress=None):
        """ use images x to produce style or use style input to predict size of objects in image

        Object size estimation is done in two steps:
        1. use a linear regression model to predict size from style in image
        2. resize image to predicted size and run CellposeModel to get output masks.
            Take the median object size of the predicted masks as the final predicted size.

        Parameters
        -------------------

        cp_model: CellposeModel
            cellpose model from which to get styles
        device: mxnet device (optional, default mx.cpu())
            where cellpose model is saved (mx.gpu() or mx.cpu())
        pretrained_size: str
            path to pretrained size model

        """
        if style is None and x is None:
            print('Error: no image or features given')
            return
        
        nimg = len(x)
        if channels is not None:
            if len(channels)==2:
                if not isinstance(channels[0], list):
                    channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i], invert=invert) for i in range(nimg)]
        diam_style = np.zeros(nimg, np.float32)
        if progress is not None:
            progress.setValue(10)
        if style is None:
            for i in trange(nimg):
                img = x[i]
                style = self.cp.eval([img], net_avg=False, tile=tile, compute_masks=False)[-1]
                if progress is not None:
                    progress.setValue(30)
                diam_style[i] = self._size_estimation(style)
            if progress is not None:
                progress.setValue(50)
        else:
            for i in range(len(style)):
                diam_style[i] = self._size_estimation(style[i])
        diam_style[diam_style==0] = self.diam_mean
        diam_style[np.isnan(diam_style)] = self.diam_mean
        masks = self.cp.eval(x, rescale=self.diam_mean/diam_style, net_avg=False, tile=tile)[0]
        diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
        diam[diam==0] = self.diam_mean
        diam[np.isnan(diam)] = self.diam_mean
        if progress is not None:
            progress.setValue(100)
        return diam, diam_style

    def _size_estimation(self, style):
        """ linear regression from style to size """
        szest = np.exp(self.params['A'] @ (style - self.params['smean']).T +
                        np.log(self.diam_mean) + self.params['ymean'])
        szest = np.maximum(5., szest)
        return szest

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
