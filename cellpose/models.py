import numpy as np
import os, sys, time, shutil, tempfile
from tqdm import trange, tqdm
from urllib.request import urlopen
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2

from mxnet import gluon, nd
from mxnet.gluon import nn
import mxnet as mx

from . import transforms, dynamics, utils, resnet_style, plot
import __main__

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

def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.
            THANKS TO TORCH, SLIGHTLY MODIFIED
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def download_model_weights(urls=urls):
    model_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'models/'
    )
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print('>>> Models not downloaded, please wait <<<')
    for url in urls:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            download_url_to_file(url, cached_file, progress=True)    

class Cellpose():
    """ main model which combines size and cellpose model """
    def __init__(self, device=mx.cpu(), model_type=None, pretrained_model=None,
                    pretrained_size=None, diam_mean=27., net_avg=True):
        super(Cellpose, self).__init__()
        self.batch_size=8
        self.diam_mean = diam_mean
        if model_type is not None and pretrained_model is None:
            pretrained_model = [os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/%s_%d'%(model_type,j))) for j in range(4)]
            pretrained_size = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    'models/size_%s_0.npy'%model_type))
            if not os.path.isfile(pretrained_model[0]):
                download_model_weights()
        elif pretrained_model is None:
            if net_avg:
                pretrained_model = [os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_%d'%j)) for j in range(4)]
                if not os.path.isfile(pretrained_model[0]):
                    download_model_weights()
            else:
                pretrained_model = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_0'))
                if not os.path.isfile(pretrained_model):
                    download_model_weights()
            if pretrained_size is None:
                pretrained_size = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    'models/size_cyto_0.npy'))
        if device==mx.gpu() and utils.use_gpu():
            self.device = mx.gpu()
        else:
            self.device = mx.cpu()
        self.pretrained_model = pretrained_model
        self.pretrained_size = pretrained_size
        self.cp = CellposeModel(device=self.device,
                                    pretrained_model=self.pretrained_model)
        if self.pretrained_size is not None:
            self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                                cp_model=self.cp, diam_mean=diam_mean)
            self.diam_mean = self.sz.diam_mean

    def eval(self, x, channels=None, rescale=1.0, do_3D=False,
                net_avg=True, progress=None, tile=True, threshold=0.4):
        # make rescale into length of x
        if rescale is not None and (not isinstance(rescale, list) or len(rescale)==1):
            rescale = rescale * np.ones(len(x), np.float32)

        if self.pretrained_size is not None and rescale is None and not do_3D:
            diams, diams_style = self.sz.eval(x, channels=channels, batch_size=self.batch_size, tile=tile)
            rescale = self.diam_mean / diams.copy()
            print('estimated cell diameters for all images')
        else:
            if rescale is None:
                rescale = np.ones(len(x), np.float32)
            diams = self.diam_mean / rescale.copy()
        masks, flows, styles = self.cp.eval(x, rescale=rescale, channels=channels, tile=tile,
                                            do_3D=do_3D, net_avg=net_avg, progress=progress,
                                            threshold=0.4)
        return masks, flows, styles, diams

class SizeModel():
    """ linear regression model for determining the size of objects in image
        used to rescale before input to CellposeModel
        uses styles from CellposeModel
    """
    def __init__(self, device=mx.cpu(), pretrained_size=None,
                    pretrained_model=None, cp_model=None, diam_mean=27., **kwargs):
        super(SizeModel, self).__init__(**kwargs)
        self.device = device
        self.diam_mean = diam_mean # avg diameter in pixels
        self.pretrained_model = pretrained_model
        self.pretrained_size = pretrained_size
        self.cp = cp_model
        if pretrained_model is not None and cp_model is None:
            self.cp = CellposeModel(device=self.device,
                                    pretrained_model=self.pretrained_model)
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params['diam_mean']

    def eval(self, x=None, feat=None, channels=None,
                batch_size=8, progress=None, tile=True):
        if feat is None and x is None:
            print('Error: no image or features given')
            return
        elif (feat is None and
                (self.pretrained_model is None and not hasattr(self,'cp'))):
            print('Error: no cellpose model or features given')
            return
        nimg = len(x)
        if channels is not None:
            if len(channels)==2:
                if not isinstance(channels[0], list):
                    channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i]) for i in range(nimg)]
        diam_style = np.zeros(nimg, np.float32)
        if progress is not None:
            progress.setValue(10)
        if feat is None:
            for i in trange(nimg):
                img = x[i]
                feat = self.cp.eval([img], net_avg=False, tile=tile)[-1]
                if progress is not None:
                    progress.setValue(30)
                diam_style[i] = self.size_estimation(feat)
            if progress is not None:
                progress.setValue(50)
        else:
            for i in range(len(feat)):
                szest = self.net(feat[i])
                diam_style[i] = np.exp(szest.asnumpy()[:,0] + np.log(self.diam_mean))
        diam_style[diam_style==0] = self.diam_mean
        diam_style[np.isnan(diam_style)] = self.diam_mean
        masks = self.cp.eval(x, rescale=self.diam_mean/diam_style, net_avg=False, tile=tile)[0]
        diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
        diam[diam==0] = self.diam_mean
        diam[np.isnan(diam)] = self.diam_mean
        if progress is not None:
            progress.setValue(100)
        return diam, diam_style

    def size_estimation(self, feat):
        szest = np.exp(self.params['A'] @ (feat - self.params['smean']).T +
                        np.log(self.diam_mean) + self.params['ymean'])
        szest = np.maximum(5., szest)
        return szest

class CellposeModel():
    def __init__(self, device, pretrained_model=None, batch_size=8,
                    diam_mean=27., net_avg=True):
        super(CellposeModel, self).__init__()
        if device==mx.gpu() and utils.use_gpu():
            self.device = mx.gpu()
        else:
            self.device = mx.cpu()
        self.pretrained_model = pretrained_model
        self.batch_size=batch_size
        self.diam_mean = diam_mean
        nbase = [32,64,128,256]
        self.net = resnet_style.CPnet(nbase, nout=3)
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.initialize(ctx = self.device)#, grad_req='null')
        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_parameters(pretrained_model)
            self.net.collect_params().setattr('grad_req', 'null')
        elif pretrained_model is None:
            if net_avg:
                pretrained_model = [os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_%d'%j)) for j in range(4)]
                if not os.path.isfile(pretrained_model[0]):
                    download_model_weights()
            else:
                pretrained_model = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_0'))
                if not os.path.isfile(pretrained_model):
                    download_model_weights()
                self.net.load_parameters(pretrained_model)
                self.net.collect_params().grad_req = 'null'
            self.pretrained_model = pretrained_model

    def eval(self, x, rescale=None, tile=True, net_avg=True, channels=None,
                do_3D=False, progress=None, threshold=0.4):
        """
            segment list of images x
        """
        nimg = len(x)
        if channels is not None:
            if len(channels)==2:
                if not isinstance(channels[0], list):
                    channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i]) for i in range(nimg)]
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

        for i in iterator:
            img = x[i].copy()
            if img.shape[0]<3:
                img = np.transpose(img, (1,2,0))
            Ly,Lx = img.shape[:2]
            if img.shape[-1]==1:
                img = np.concatenate((img, 0.*img), axis=-1)
            #tic=time.time()
            if isinstance(self.pretrained_model, str) or not net_avg:
                y, style = self.run_net(img, rescale[i], tile)
            else:
                y, style = self.run_many(img, rescale[i], tile)
            if progress is not None:
                progress.setValue(55)
            styles.append(style)
            cellprob = y[...,2]
            dP = np.stack((y[...,0], y[...,1]), axis=0)
            p = dynamics.follow_flows(-1 * dP * (cellprob>0) / 5.)
            if progress is not None:
                progress.setValue(65)
            maski = dynamics.get_masks(p, flows=dP, threshold=threshold)
            if progress is not None:
                progress.setValue(75)
            dZ = np.zeros((1,Ly,Lx), np.uint8)
            dP = np.concatenate((dP, dZ), axis=0)
            flow = plot.dx_to_circ(dP)
            flows.append([flow, dP, cellprob])
            masks.append(maski)

        return masks, flows, styles

    def run_many(self, img, rsz=1.0, tile=True):
        for j in range(len(self.pretrained_model)):
            self.net.load_parameters(self.pretrained_model[j])
            self.net.collect_params().grad_req = 'null'
            yup0, style = self.run_net(img, rsz, tile)
            if j==0:
                yup = yup0
            else:
                yup += yup0
        yup = yup / len(self.pretrained_model)
        return yup, style

    def run_tiled(self, imgi, bsize=224):
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize, augment=True)
        X = nd.array(IMG, ctx=self.device)
        nbatch = self.batch_size
        niter = int(np.ceil(IMG.shape[0]/nbatch))
        y = np.zeros((IMG.shape[0], 3, bsize, bsize))
        for k in range(niter):
            irange = np.arange(nbatch*k, min(IMG.shape[0], nbatch*k+nbatch))
            y0, style = self.net(X[irange])
            y[irange] = y0.asnumpy()
            if k==0:
                styles = style.asnumpy()[0]
            styles += style.asnumpy().sum(axis=0)
        styles /= IMG.shape[0]
        y = transforms.unaugment_tiles(y)

        # taper edges of tiles
        Navg = np.zeros((Ly,Lx))
        ytiled = np.zeros((3, Ly, Lx), np.float32)
        mask = utils.taper_mask()
        for j in range(len(ysub)):
            ytiled[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += y[j] * mask
            Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += mask
        ytiled /=Navg
        ytiled = ytiled[:,:imgi.shape[1], :imgi.shape[2]]
        styles /= (styles**2).sum()**0.5
        return ytiled, styles

    def run_net(self, img, rsz=1.0, tile=True, bsize=224):
        shape = img.shape
        if rsz==1.0:
            Ly,Lx = img.shape[:2]
        else:
            Ly = int(img.shape[0] * rsz)
            Lx = int(img.shape[1] * rsz)
            img = cv2.resize(img, (Lx, Ly))

        # make image 1 x nchan x Ly x Lx for net
        if img.ndim<3:
            img = np.expand_dims(img, axis=-1)
        img = np.transpose(img, (2,0,1))
        # pad for net so divisible by 4
        img, ysub, xsub = utils.pad_image(img)

        if tile:
            y,style = self.run_tiled(img, bsize)
            y = np.transpose(y[:3], (1,2,0))
        else:
            img = nd.array(np.expand_dims(img, axis=0), ctx=self.device)
            y,style = self.net(img)
            img = img.asnumpy()
            #y = nd.array((img.shape[1], img.shape[2],3), ctx=self.device, dtype='float')
            #y = np.zeros((img.shape[1], img.shape[2],3), np.float32)
            y = np.transpose(y[0].asnumpy(), (1,2,0))
            style = style.asnumpy()[0]
            style = np.ones(10)
        style /= (style**2).sum()**0.5

        y = y[np.ix_(ysub, xsub, np.arange(3))]

        if rsz!=1.0:
            y = cv2.resize(y, (shape[1], shape[0]))
        return y, style
