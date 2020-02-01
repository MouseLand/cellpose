#from suite2p import nonrigid
import numpy as np
import cv2

def unaugment_tiles(y):
    for k in range(y.shape[0]):
        if k%4==1:
            y[k, :,:, :] = y[k, :,::-1, :]
            y[k,0,:,:] *= -1
        if k%4==2:
            y[k, :,:, :] = y[k, :,:, ::-1]
            y[k,1,:,:] *= -1
        if k%4==3:
            y[k, :,:, :] = y[k, :,::-1, ::-1]
            y[k,0,:,:] *= -1
            y[k,1,:,:] *= -1
    return y

def make_tiles(imgi, bsize=224, augment=True):
    nchan, Ly0, Lx0 = imgi.shape[-3:]
    # pad if image smaller than bsize
    if Ly0<bsize:
        imgi = np.concatenate((imgi, np.zeros((nchan,bsize-Ly0, Lx0))), axis=1)
        Ly0 = bsize
    if Lx0<bsize:
        imgi = np.concatenate((imgi, np.zeros((nchan,Ly0, bsize-Lx0))), axis=2)
    Ly, Lx = imgi.shape[-2:]


    # tile starts
    ystart = np.arange(0, Ly-bsize//2, bsize//2)
    xstart = np.arange(0, Lx-bsize//2, bsize//2)
    ystart = np.maximum(0, np.minimum(Ly-bsize, ystart))
    xstart = np.maximum(0, np.minimum(Lx-bsize, xstart))

    ysub = []
    xsub = []

    IMG = np.zeros((len(ystart), len(xstart), nchan,  bsize,bsize))
    k = 0
    for j in range(len(ystart)):
        for i in range(len(xstart)):
            ysub.append([ystart[j], ystart[j]+bsize])
            xsub.append([xstart[i], xstart[i]+bsize])

            IMG[j,i,:,:,:] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]

    IMG = np.reshape(IMG, (-1, nchan, bsize,bsize))

    # "augment" images
    for k in range(IMG.shape[0]):
        if k%4==1:
            IMG[k, :,:, :] = IMG[k, :,::-1, :]
        if k%4==2:
            IMG[k, :,:, :] = IMG[k, :,:, ::-1]
        if k%4==3:
            IMG[k, :,:, :] = IMG[k,:, ::-1, ::-1]

    return IMG, ysub, xsub, Ly, Lx


def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def reshape(data, channels=[0,0]):
    data = data.astype(np.float32)
    # use grayscale image
    if channels[0]==0:
        data = data.mean(axis=-1)
        data = np.expand_dims(data, axis=-1)
        data = normalize99(data)
    else:
        chanid = [channels[0]-1]
        if channels[1] > 0:
            chanid.append(channels[1]-1)
        data = data[:,:,chanid]
        for i in range(data.shape[-1]):
            data[...,i] = normalize99(data[...,i])
    if len(channels)>1 and data.shape[-1]==1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    elif len(channels)==1 and data.shape[-1]>1:
        data = data[...,:-1]
        data = normalize99(data)
    if data.ndim==4:
        data = np.transpose(data, (3,0,1,2))
    else:
        data = np.transpose(data, (2,0,1))
    return data


def pad_image_CS(img0, div=16, extra = 1):
    Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0,pads, mode='constant')
    return I,pads

def pad_image(img0, div=16, extra = 1):
    nc, Ly, Lx = img0.shape
    Lpad = int(div * np.ceil(Ly/div) - Ly)

    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(Lx/div) - Lx)
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    ysub = np.arange(xpad1, xpad1+Ly)
    xsub = np.arange(ypad1, ypad1+Lx)
    I = np.pad(img0,pads, mode='constant')
    return I, ysub, xsub

def elastic_transform(data, alpha=2, ngrid=16):
    nimg,nmaps,Ly,Lx = data.shape
    iy = np.arange(0,Ly,1,np.float32)
    ix = np.arange(0,Lx,1,np.float32)
    yb = np.linspace(0,Ly,ngrid+2)[1:-1]
    xb = np.linspace(0,Lx,ngrid+2)[1:-1]
    iy = np.interp(iy, yb, np.arange(0,yb.size,1,int)).astype(np.float32)
    ix = np.interp(ix, xb, np.arange(0,xb.size,1,int)).astype(np.float32)
    mshx,mshy = np.meshgrid(ix, iy)

    mshxA,mshyA = np.meshgrid(np.arange(0,Lx,1,np.float32), np.arange(0,Ly,1,np.float32))

    yshift = np.random.randn(nimg, ngrid, ngrid).astype(np.float32) * alpha
    xshift = np.random.randn(nimg, ngrid, ngrid).astype(np.float32) * alpha

    xshift = cv2.GaussianBlur(xshift,(3,3),cv2.BORDER_REFLECT)
    yshift = cv2.GaussianBlur(yshift,(3,3),cv2.BORDER_REFLECT)

    # interpolate from block centers to all points Ly x Lx
    yup = np.zeros((nimg,Ly,Lx), np.float32)
    xup = np.zeros((nimg,Ly,Lx), np.float32)
    nonrigid.block_interp(yshift, xshift, mshy, mshx, yup, xup)
    Y = np.zeros(data.shape, np.float32)
    #for i in range(nmaps):
        # use shifts and do bilinear interpolation
    #    nonrigid.shift_coordinates(data[:,i,:,:], yup, xup, mshyA, mshxA, Y[:,i,:,:])

    return Y

def random_rotate_and_resize(X, Y=None, scale_range=1., xy = (224,224), rescale=None):
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim>2:
        nchan = X[0].shape[-1]
        imgi  = np.zeros((nimg, xy[0], xy[1], nchan), np.float32)
    else:
        nchan = 1
        imgi  = np.zeros((nimg, xy[0], xy[1], 1), np.float32)
    lbl = []
    if Y is not None:
        if Y[0].ndim>2:
            nt = Y[0].shape[-1]
            lbl   = np.zeros((nimg, xy[0], xy[1], nt), np.int32)
        else:
            nt = 1
            lbl   = np.zeros((nimg, xy[0], xy[1], 1), np.int32)

    scale = np.zeros(nimg, np.float32)
    for k in range(nimg):
        Ly, Lx = X[k].shape[:2]

        # generate random augmentation parameters
        flip = np.random.rand()>.5
        theta = np.random.rand() * np.pi * 2
        scale[k] = (1-scale_range/2) + scale_range * np.random.rand()
        if rescale is not None:
            scale[k] *= 1. / rescale[k]
        dxy = np.maximum(0, np.array([Lx*scale[k]-xy[1],Ly*scale[k]-xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy

        # create affine transform
        cc = np.array([Lx/2, Ly/2])
        cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        pts2 = np.float32([cc1,
                cc1 + scale[k]*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale[k]*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)

        img = X[k].copy()
        if img.ndim<3:
            img = img[:,:,np.newaxis]
        if Y is not None:
            labels = Y[k].copy()
            if labels.ndim<3:
                labels = labels[:,:,np.newaxis]

        if flip:
            img = img[:, ::-1]
            if Y is not None:
                labels = labels[:,::-1]
                if nt > 1:
                    labels[...,1] = -labels[...,1]

        for n in range(nchan):
            imgi[k,:,:,n] = cv2.warpAffine(img[:,:,n],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)

        for n in range(nt):
            if n==0:
                lbl[k,:,:,n] = cv2.warpAffine(labels[:,:,n],M,(xy[0],xy[1]), flags=cv2.INTER_NEAREST)
            else:
                lbl[k,:,:,n] = cv2.warpAffine(labels[:,:,n],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
    if Y[0].ndim<3:
        lbl = lbl[...,0]
    #imgi = np.transpose(imgi, (0, 3, 1, 2))
    return imgi, lbl, scale
