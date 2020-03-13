#from suite2p import nonrigid
import numpy as np
import cv2

def unaugment_tiles(y):
    """ reverse test-time augmentations for averaging

    Parameters
    ----------
    y : float32
        array that's ntiles x chan x Ly x Lx where chan = (dY, dX, cell prob)
    
    Returns
    -------
    y : float32

    """

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
    """ make tiles of image to run at test-time

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx
    
    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize
    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles 
    xsub : list
        list of arrays with start and end of tiles in X of length ntiles 
    Ly : int
        size of tiles in Y
    Lx : int
        size of tiles in X

    """

    bsize = np.int32(bsize)
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

def X2zoom(img, X2=1):
    """ zoom in image

    Parameters
    ----------
    img : numpy array that's Ly x Lx 
    
    Returns
    -------
    img : numpy array that's Ly x Lx 

    """
    ny,nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2**X2)), int(ny * (2**X2))))
    return img

def image_resizer(img, resize=512, to_uint8=False):
    """ resize image

    Parameters
    ----------
    img : numpy array that's Ly x Lx 
    
    resize : int
        max size of image returned 

    to_uint8 : bool
        convert image to uint8

    Returns
    -------
    img : numpy array that's Ly x Lx, Ly,Lx<resize

    """
    ny,nx = img.shape[:2]
    if to_uint8:
        if img.max()<=255 and img.min()>=0 and img.max()>1:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
    if np.array(img.shape).max() > resize:
        if ny>nx:
            nx = int(nx/ny * resize)
            ny = resize
        else:
            ny = int(ny/nx * resize)
            nx = resize
        shape = (nx,ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img

def normalize99(img):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def reshape(data, channels=[0,0], invert=False):
    """ reshape data using channels and normalize intensities (w/ optional inversion)

    Parameters
    ----------
    data : numpy array that's (Z x ) Ly x Lx x nchan
    
    channels : int
        list [chan to seg, chan2 (opt)]
        (0=gray/none, 1=red, 2=green, 3=blue)

    invert : bool
        invert intensities

    Returns
    -------
    data : numpy array that's nchan x (Z x ) Ly x Lx

    """
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:,:,np.newaxis]
    elif data.shape[0]<8 and data.ndim==3:
        data = np.transpose(data, (1,2,0))

    # use grayscale image
    if data.shape[-1]==1:
        data = normalize99(data)
        if invert:
            data = -1*data + 1
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0]==0:
            data = data.mean(axis=-1)
            data = np.expand_dims(data, axis=-1)
            data = normalize99(data)
            if invert:
                data = -1*data + 1
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0]-1]
            if channels[1] > 0:
                chanid.append(channels[1]-1)
            data = data[:,:,chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[...,i]) > 0.0:
                    data[...,i] = normalize99(data[...,i])
                else:
                    if i==0:
                        print("WARNING: 'chan to seg' has value range of ZERO")
                    else:
                        print("WARNING: 'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
    if data.ndim==4:
        data = np.transpose(data, (3,0,1,2))
    else:
        data = np.transpose(data, (2,0,1))
    return data

def reshape_data(train_data, test_data=None, channels=None):
    """ inputs converted to correct shapes for training """

    # if training data is less than 2D
    nimg = len(train_data)
    if channels is not None:
        train_data = [reshape(train_data[n], channels=channels) for n in range(nimg)]
    if train_data[0].ndim < 3:
        train_data = [train_data[n][np.newaxis,:,:] for n in range(nimg)]
    elif train_data[0].shape[-1] < 8:
        print('NOTE: assuming train_data provided as Ly x Lx x nchannels, transposing axes to put channels first')
        train_data = [np.transpose(train_data[n], (2,0,1)) for n in range(nimg)]
    nchan = [train_data[n].shape[0] for n in range(nimg)]
    if nchan.count(nchan[0]) != len(nchan):
        return None, None, None
    nchan = nchan[0]

    # check for valid test data
    run_test = False
    if test_data is not None:
        nimg = len(test_data)
        if channels is not None:
            test_data = [reshape(test_data[n], channels=channels) for n in range(nimg)]
        if test_data[0].ndim==2:
            if nchan==1:
                run_test = True
                test_data = [test_data[n][np.newaxis,:,:] for n in range(nimg)]
        elif test_data[0].ndim==3:
            if test_data[0].shape[-1] < 8:
                print('NOTE: assuming test_data provided as Ly x Lx x nchannels, transposing axes to put channels first')
                test_data = [np.transpose(test_data[n], (2,0,1)) for n in range(nimg)]
            nchan_test = [test_data[n].shape[0] for n in range(nimg)]
            if nchan_test.count(nchan_test[0]) != len(nchan_test):
                run_test = False
            elif test_data[0].shape[0]==nchan:
                run_test = True

    #normalize_data
    nimg = len(train_data)
    for n in range(nimg):
        train_data[n] = train_data[n].astype(np.float32)
        for k in range(nchan):
            if np.ptp(train_data[n][k]) > 0.0:
                train_data[n][k] = normalize99(train_data[n][k])
    if run_test:
        nimg = len(test_data)
        for n in range(nimg):
            test_data[n] = test_data[n].astype(np.float32)
            for k in range(nchan):
                if np.ptp(test_data[n][k]) > 0.0:
                    test_data[n][k] = normalize99(test_data[n][k])
    
    return train_data, test_data, run_test



def pad_image_ND(img0, div=16, extra = 1):
    """ pad image for test-time (2 and 3D) """
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

    return I, pads

def pad_image(img0, div=16, extra = 1):
    """ pad image for test time (if tiling off) """
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

def random_rotate_and_resize(X, Y=None, scale_range=1., xy = (224,224), do_flip=True, rescale=None):
    """ augmentation by random rotation and resizing

        Parameters
        ----------
        data : numpy array that's (Z x ) Ly x Lx x nchan
        
        channels : int
            list [chan to seg, chan2 (opt)]
            (0=gray/none, 1=red, 2=green, 3=blue)

        invert : bool
            invert intensities

        Returns
        -------
        data : numpy array that's nchan x (Z x ) Ly x Lx

        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)

        if Y is 3 chan, it is [cell probability, Y flow, X flow]
    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim>2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi  = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)
    
    lbl = []
    if Y is not None:
        if Y[0].ndim>2:
            nt = Y[0].shape[0]     
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)

    scale = np.zeros(nimg, np.float32)
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        # generate random augmentation parameters
        flip = np.random.rand()>.5
        theta = np.random.rand() * np.pi * 2
        scale[n] = (1-scale_range/2) + scale_range * np.random.rand()
        if rescale is not None:
            scale[n] *= 1. / rescale[n]
        dxy = np.maximum(0, np.array([Lx*scale[n]-xy[1],Ly*scale[n]-xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy

        # create affine transform
        cc = np.array([Lx/2, Ly/2])
        cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        pts2 = np.float32([cc1,
                cc1 + scale[n]*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale[n]*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)

        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim<3:
                labels = labels[np.newaxis,:,:]

        if flip and do_flip:
            img = img[:, :, ::-1]
            if Y is not None:
                labels = labels[:, :, ::-1]
                if nt > 1:
                    labels[2] = -labels[2]

        for k in range(nchan):
            imgi[n,k] = cv2.warpAffine(img[k], M, (xy[0],xy[1]), flags=cv2.INTER_LINEAR)

        for k in range(nt):
            if k==0:
                lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[0],xy[1]), flags=cv2.INTER_NEAREST)
            else:
                lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        
        if nt>1:
            v1 = lbl[n,2].copy()
            v2 = lbl[n,1].copy()
            lbl[n,1] = (-v1 * np.sin(-theta) + v2*np.cos(-theta))
            lbl[n,2] = (v1 * np.cos(-theta) + v2*np.sin(-theta))

    if Y[0].ndim<3:
        lbl = lbl[0]
    #imgi = np.transpose(imgi, (0, 3, 1, 2))
    return imgi, lbl, scale