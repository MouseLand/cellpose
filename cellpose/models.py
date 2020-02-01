import numpy as np
import os
from datetime import datetime
import time
from glob import glob
from tqdm import trange
from scipy.ndimage import median_filter
import cv2
from mxnet import gluon, nd
from mxnet.gluon import nn
import mxnet as mx
from . import transforms, dynamics, utils, nets, resnet_style, plot
import __main__

class Cellpose():
    """ main model which combines size and cellpose model """
    def __init__(self, device=mx.cpu(), pretrained_model=None, 
                    pretrained_size=None, diam_mean=27., net_avg=True):
        super(Cellpose, self).__init__()
        self.device = device
        self.batch_size=8
        self.diam_mean = diam_mean
        if pretrained_model is None:
            if net_avg:
                pretrained_model = [os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_%d'%j)) for j in range(4)]
            else:
                pretrained_model = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_0'))
            if pretrained_size is None:
                pretrained_size = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    'models/size_cyto_A.npy'))
        self.pretrained_model = pretrained_model
        self.pretrained_size = pretrained_size
        self.cp = CellposeModel(device=self.device,
                                    pretrained_model=self.pretrained_model)
        if self.pretrained_size is not None:
            self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                                cp_model=self.cp, diam_mean=diam_mean)

    def eval(self, x, channels=None, rescale=1.0, do_3D=False, 
                net_avg=True, progress=None, tile=True):
        # make rescale into length of x
        if rescale is not None and (not isinstance(rescale, list) or len(rescale)==1):
            rescale = rescale * np.ones(len(x), np.float32)

        if self.pretrained_size is not None and rescale is None and not do_3D:
            diams = self.sz.eval(x, channels=channels, batch_size=self.batch_size, tile=tile)
            rescale = self.diam_mean / diams.copy()
            print('estimated cell diameters for all images')
        else:
            if rescale is None:
                rescale = np.ones(len(x), np.float32)
            diams = self.diam_mean / rescale.copy()
        masks, flows, styles = self.cp.eval(x, rescale=rescale, channels=channels, tile=tile,
                                            do_3D=do_3D, net_avg=net_avg, progress=progress)
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
                channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i]) for i in range(nimg)]
        diam = np.zeros(nimg, np.float32)
        if progress is not None:
            progress.setValue(10)
        if feat is None:
            for i in trange(nimg):
                img = x[i]
                feat = self.cp.eval([img], net_avg=False, tile=tile)[-1]
                if progress is not None:
                    progress.setValue(30)
                diam[i] = self.size_estimation(feat)
            if progress is not None:
                progress.setValue(50)
            masks = self.cp.eval(x, rescale=self.diam_mean/diam, net_avg=False, tile=tile)[0]
            diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
            if progress is not None:
                progress.setValue(100)
        else:
            for i in range(len(feat)):
                szest = self.net(feat[i])
                diam[i] = np.exp(szest.asnumpy()[:,0] + np.log(self.diam_mean))
        return diam

    def size_estimation(self, feat):
        szest = np.exp(self.params['A'] @ (feat - self.params['smean']).T +
                        np.log(self.diam_mean) + self.params['ymean'])
        szest = np.maximum(5., szest)
        return szest

class CellposeModel():
    def __init__(self, device, pretrained_model=None, batch_size=8, 
                    diam_mean=27., net_avg=True):
        super(CellposeModel, self).__init__()
        self.device = device
        self.pretrained_model = pretrained_model
        self.batch_size=batch_size
        self.diam_mean = diam_mean
        nbase = [32,64,128,256]
        self.net = resnet_style.CPnet(nbase, nout=3)
        if device==mx.gpu():
            self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.initialize(ctx = self.device)#, grad_req='null')
        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_parameters(pretrained_model)
            self.net.collect_params().setattr('grad_req', 'null')

        elif pretrained_model is None:
            if net_avg:
                pretrained_model = [os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_%d'%j)) for j in range(4)]
            else:
                pretrained_model = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'models/cyto_0'))
                self.net.load_parameters(pretrained_model)
                self.net.collect_params().grad_req = 'null'

            self.pretrained_model = pretrained_model

    def train(self, train_data, train_labels, test_data=None, test_labels=None, learning_rate=0.2, n_epochs=500,
                weight_decay=0.00001, batch_size=8, augmenter=None):
        d = datetime.now()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = 0.9

        nimg = len(train_data)
        diam_train = np.array([utils.diameters(train_labels[k])[0] for k in range(len(train_labels))])
        diam_train[diam_train<5] = 5.
        if test_labels is not None:
            diam_test = np.array([utils.diameters(test_labels[k])[0] for k in range(len(test_labels))])
            diam_test[diam_test<5] = 5.

        print('median diameter = %d'%self.diam_mean)
        print(self.learning_rate, self.momentum, self.batch_size, self.weight_decay)

        criterion  = gluon.loss.L2Loss()
        criterion2 = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'sgd',{'learning_rate': self.learning_rate,
                                'momentum': self.momentum, 'wd': self.weight_decay})

        eta = np.linspace(0, self.learning_rate, 10)
        t0 = tic()

        lavg, nsum = 0, 0

        file_path = file_path.rsplit('.', 1)[0]
        head, tail = os.path.split(file_path)
        file_path = os.path.join(head, 'models')
        file_path = os.path.join(file_path, tail)

        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg).astype('int32')
            if iepoch<len(eta):
                LR = eta[iepoch]
                trainer.set_learning_rate(LR)

            for ibatch in range(0,nimg,batch_size):
                rescale = diam_train[rperm[ibatch:ibatch+batch_size]] / self.diam_mean
                imgi, lbl, _ = transforms.random_rotate_and_resize(
                                            [train_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                            Y=[train_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                            rescale=rescale)
                X = nd.array(imgi, ctx=self.device)
                veci = 5. * nd.array(lbl[:,:], ctx=self.device)
                lbl = nd.array(lbl>.5, ctx=self.device)
                with mx.autograd.record():
                    y, style = self.net(X)
                    loss = criterion(y[:,:2] , veci) + criterion2(y[:,-1] , lbl)
                loss.backward()
                train_loss = nd.sum(loss).asscalar()
                lavg += train_loss
                nsum+=len(loss)
                if iepoch>0:
                    trainer.step(batch_size)

            if iepoch>nepochs-100 and iepoch%10==1:
                LR = LR/2
                trainer.set_learning_rate(LR)

            if iepoch%10==0 or iepoch<10:
                lavg = lavg / nsum
                lavgt = 0
                nsum = 0
                np.random.seed(42)
                rperm = np.random.permutation(len(vft)).astype('int32')
                for ibatch in range(0,len(vft),batch_size):
                    sc = Pt[rperm[ibatch:ibatch+batch_size]]/szmedian
                    imgi, veci, lbl = batch.withflow_twochan([vft[i] for i in
                        rperm[ibatch:ibatch+batch_size]], test = True, scales = sc)
                    X = nd.array(imgi, ctx=device)
                    veci = 5. * nd.array(veci[:,:], ctx=device)
                    lbl = nd.array(lbl>.5, ctx=device)

                    y,style = net(X)
                    loss = criterion(y[:,:2] , veci) + criterion2(y[:,-1] , lbl)
                    lavgt += nd.sum(loss).asscalar()
                    nsum+=len(loss)
                print('Epoch %d, Time %4.1fs, Loss %2.4f,Loss Test %2.4f, LR %2.4f'%(iepoch, toc(t0), lavg, lavgt/nsum, LR))
                lavg, nsum = 0, 0

            if save_flag:
                if iepoch==nepochs-1 or iepoch%100==1:
                    # save model at the end
                    if iepoch==1:
                        ksave = 0
                        while 1:
                            #file = '%s_%d_%d_%d_%d_ep%d_%d%d_ep%d_LR%d_%d'%(file_path,nbase[0],nbase[1],nbase[2],nbase[3], iepoch, d.month, d.day, args.nepochs, int(100*args.LR), ksave)
                            file = '{}_{}_{}{}_{}'.format(file_path, tuple(ls), d.month, d.day, ksave)
                            if len(glob(file))==0:
                                break
                            ksave = ksave+1
                        print(ksave)
                    file = '{}_{}_{}{}_{}'.format(file_path, tuple(ls), d.month, d.day, ksave)
                    net.save_parameters(file)

    def eval(self, x, rescale=None, tile=True, net_avg=True, channels=None, jit=True,
                do_3D=False, progress=None):
        """
            segment list of images x 
        """
        nimg = len(x)
        if channels is not None:
            if len(channels)==2:
                channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i]) for i in range(nimg)]
            if nimg>1 and do_3D:
                x = np.array(x)
                x = np.transpose(x, (1,0,2,3))
        styles = []
        flows = []
        masks = []
        if nimg>1 and do_3D:
            styles = []
            sstr = ['XY', 'XZ', 'YZ']
            if x.shape[-1] < 3:
                x = np.transpose(x, (3,0,1,2))
            pm = [(1,2,3,0), (2,1,3,0), (3,1,2,0)]
            ipm = [(0,1,2,3), (0,2,1,3), (0,2,3,1)]
            tic=time.time()
            for p in range(3):
                xsl = np.transpose(x.copy(), pm[p])
                flows.append(np.zeros(((3,xsl.shape[0],xsl.shape[1],xsl.shape[2])), np.float32))
                # per image
                iterator = trange(xsl.shape[0])
                print('running %s \n'%sstr[p])
                for i in iterator:
                    if isinstance(self.pretrained_model, str) or not net_avg:
                        y, style = self.run_net(xsl[i], tile=tile)
                    else:
                        y, style = self.run_many(xsl[i], tile=tile)
                    y = np.transpose(y[:,:,[1,0,2]], (2,0,1))
                    flows[p][:,i] = y
                flows[p] = np.transpose(flows[p], ipm[p])
            dX = flows[0][0] + flows[1][0]
            dY = flows[0][1] + flows[2][0]
            dZ = flows[1][1] + flows[2][1]
            cellprob = flows[0][-1] + flows[1][-1] + flows[2][-1]

            dP = np.concatenate((dZ[np.newaxis,...], dY[np.newaxis,...], dX[np.newaxis,...]), axis=0)
            print('flows computed %2.2fs'%(time.time()-tic))
            yout = dynamics.follow_flows(-1*dP * (cellprob>0.) / 5., do_3D=True)
            print('dynamics computed %2.2fs'%(time.time()-tic))
            masks = dynamics.get_masks(yout)
            print('masks computed %2.2fs'%(time.time()-tic))
            styles = [np.array([plot.dx_to_circ(dP[1:,i]) for i in range(dP.shape[1])])]
            styles.append(dP)
            styles.append(cellprob)
        else:
            if rescale is None:
                rescale = np.ones(nimg)
            if nimg > 1:
                iterator = trange(nimg)
            else:
                iterator = range(nimg)

            if isinstance(self.pretrained_model, list) and not net_avg:
                self.net.load_parameters(self.pretrained_model[0])

            for i in iterator:
                img = x[i].copy()
                if img.shape[0]<3:
                    img = np.transpose(img, (1,2,0))
                Ly,Lx = img.shape[:2]
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
                if jit:
                    p = dynamics.follow_flows(-1 * dP * (cellprob>0) / 5.)
                else:
                    p = utils.run_dynamics(-1 * dP[:,:,::-1] * (cellprob>0) / 5., niter=200)
                    p = p[:,:,::-1]
                if progress is not None:
                    progress.setValue(65)
                maski = dynamics.get_masks(p, flows=dP)
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
            #styles += style.asnumpy().sum(axis=0)
        #styles /= IMG.shape[0]
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
        #styles /= (styles**2).sum()**0.5
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
