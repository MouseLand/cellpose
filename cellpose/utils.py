import cv2
from scipy.ndimage.filters import maximum_filter1d
from skimage import draw
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gpu, cpu
import time
from numba import njit, float32, int32
import os, pickle

def use_gpu(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
        return True
    except mx.MXNetError:
        return False


def get_cell_data(root_data, fname):
    iscell = np.load(os.path.join(root_data, 'iscell.npy'))
    ls = [1,2,3,4,5,0]

    file_path = os.path.join(root_data, fname)
    with open(file_path,"rb") as pickle_in:
        V = pickle.load(pickle_in)

    vf = []
    print(ls)
    for k in range(len(ls)):
        irange = (iscell==ls[k]).nonzero()[0]
        vf.extend([V[k] for k in irange])

    np.random.seed(101)
    r = np.random.rand(10000)

    vft = [vf[k] for k in (r[:len(vf)]<.1).nonzero()[0]]
    vf  = [vf[k] for k in (r[:len(vf)]>.1).nonzero()[0]]

    return vf, vft

def get_nuclei_data(root_data):
    iscell = np.load(os.path.join(root_data, 'isnuc.npy'))
    ls = [1,2,3,4]

    file_path = os.path.join(root_data, 'vf_nuclei.pickle')
    with open(file_path,"rb") as pickle_in:
        V = pickle.load(pickle_in)

    vf = []
    print(ls)
    for k in range(len(ls)):
        irange = (iscell==ls[k]).nonzero()[0]
        vf.extend([V[k] for k in irange])

    np.random.seed(101)
    r = np.random.rand(10000)

    vft = [vf[k] for k in (r[:len(vf)]<.1).nonzero()[0]]
    vf  = [vf[k] for k in (r[:len(vf)]>.1).nonzero()[0]]

    return vf, vft

def taper_mask(bsize=224, sig=7.5):
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask[:, np.newaxis]
    return mask


def diameters(masks):
    unique, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    return md, counts**0.5

def radius_distribution(masks, bins):
    unique, counts = np.unique(masks, return_counts=True)
    counts = counts[unique!=0]
    nb, _ = np.histogram((counts**0.5)*0.5, bins)
    nb = nb.astype(np.float32)
    if nb.sum() > 0:
        nb = nb / nb.sum()
    md = np.median(counts**0.5)*0.5
    if np.isnan(md):
        md = 0
    return nb, md, (counts**0.5)/2

def X2zoom(img, X2=1):
    ny,nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2**X2)), int(ny * (2**X2))))
    return img

def image_resizer(img, resize=512, to_uint8=False):
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
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def gabors(npix):
    ''' npix - size of gabor patch (should be ODD)'''
    y,x=np.meshgrid(np.arange(npix),np.arange(npix))
    sigma = 1
    f = 0.1
    theta = np.linspace(0, 2*np.pi, 33)[:-1]
    theta = theta[:,np.newaxis,np.newaxis]
    ycent,xcent = y.mean(), x.mean()
    yc = y - ycent
    xc = x - xcent
    ph = np.pi/2

    xc = xc[np.newaxis,:,:]
    yc = yc[np.newaxis,:,:]
    G = np.exp(-(xc**2 + yc**2) / (2*sigma**2)) * np.cos(ph + f * (yc*np.cos(theta) + xc*np.sin(theta)))

    return G

def format_data(X,Y):
    nimg = len(Y)
    vf = []

    t0 = time.time()

    Rs = np.zeros(nimg)
    for j in range(nimg):
        Ly, Lx = Y[j].shape
        xm, ym = np.meshgrid(np.arange(Lx),  np.arange(Ly))
        unqY = np.unique(Y[j])
        img = np.float32(X[j])
        #img = (img - img.mean())/np.std(img)

        labels = np.int32(Y[j])

        Ly, Lx = img.shape
        xm, ym = np.meshgrid(np.arange(Lx),  np.arange(Ly))
        unqY = np.unique(labels)

        ix = labels==0

        img = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1))

        V = np.zeros((4,Ly,Lx), 'float32')
        V[0] = img

        #V[1], V[2], maskE = compute_flow(Y[j])
        #V[3] = np.float32(labels>.5) + np.float32(maskE>.5)
        V[1], V[2] = new_flow(Y[j])
        V[3] = np.float32(labels>.5)
        vf.append(V)

        if j%20==1:
            print(j, time.time()-t0)
    return vf

def extendROI(ypix, xpix, Ly, Lx,niter=1):
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix-1, ypix+1), (xpix, xpix+1,xpix-1,xpix,xpix))
        yx = np.array(yx)
        yx = yx.reshape((2,-1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
        ypix,xpix = yu[:, ix]
    return ypix,xpix

def get_mask(y, rpad=20, nmax=20):
    xp = y[1,:,:].flatten().astype('int32')
    yp = y[0,:,:].flatten().astype('int32')
    _, Ly, Lx = y.shape
    xm, ym = np.meshgrid(np.arange(Lx),  np.arange(Ly))

    xedges = np.arange(-.5-rpad, xm.shape[1]+.5+rpad, 1)
    yedges = np.arange(-.5-rpad, xm.shape[0]+.5+rpad, 1)
    #xp = (xm-dx).flatten().astype('int32')
    #yp = (ym-dy).flatten().astype('int32')
    h,_,_ = np.histogram2d(xp, yp, bins=[xedges, yedges])

    hmax = maximum_filter1d(h, 5, axis=0)
    hmax = maximum_filter1d(hmax, 5, axis=1)

    yo, xo = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[yo, xo]
    isort = np.argsort(Nmax)[::-1]
    yo, xo = yo[isort], xo[isort]
    pix = []
    for t in range(len(yo)):
        pix.append([yo[t],xo[t]])

    for iter in range(5):
        for k in range(len(pix)):
            ye, xe = extendROI(pix[k][0], pix[k][1], h.shape[0], h.shape[1], 1)
            igood = h[ye, xe]>2
            ye, xe = ye[igood], xe[igood]
            pix[k][0] = ye
            pix[k][1] = xe

    ibad = np.ones(len(pix), 'bool')
    for k in range(len(pix)):
        #print(pix[k][0].size)
        if pix[k][0].size<nmax:
            ibad[k] = 0

    #pix = [pix[k] for k in ibad.nonzero()[0]]

    M = np.zeros(h.shape)
    for k in range(len(pix)):
        M[pix[k][0],    pix[k][1]] = 1+k

    M0 = M[rpad + xp, rpad + yp]
    M0 = np.reshape(M0, xm.shape)
    return M0, pix


def pad_image_CS0(img0, div=16):
    Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    xpad1 = Lpad//2
    xpad2 = Lpad - xpad1
    Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    ypad1 = Lpad//2
    ypad2 = Lpad - ypad1

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0,pads, mode='constant')
    return I, pads

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

def run_tile(net, imgi, bsize=224, device=mx.cpu()):
    nchan, Ly0, Lx0 = imgi.shape[-3:]
    if Ly0<bsize:
        imgi = np.concatenate((imgi, np.zeros((nchan,bsize-Ly0, Lx0))), axis=1)
        Ly0 = bsize
    if Lx0<bsize:
        imgi = np.concatenate((imgi, np.zeros((nchan,Ly0, bsize-Lx0))), axis=2)
    Ly, Lx = imgi.shape[-2:]


    ystart = np.arange(0, Ly-bsize//2, int(bsize//2))
    xstart = np.arange(0, Lx-bsize//2, int(bsize//2))

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

    if True:
        for k in range(IMG.shape[0]):
            if k%4==1:
                IMG[k, :,:, :] = IMG[k, :,::-1, :]
            if k%4==2:
                IMG[k, :,:, :] = IMG[k, :,:, ::-1]
            if k%4==3:
                IMG[k, :,:, :] = IMG[k,:, ::-1, ::-1]


    X = nd.array(IMG, ctx=device)
    nbatch = 8
    niter = int(np.ceil(IMG.shape[0]/nbatch))
    nout = 3
    y = np.zeros((IMG.shape[0], nout, bsize,bsize))

    for k in range(niter):
        irange = np.arange(nbatch*k, min(IMG.shape[0], nbatch*k+nbatch))
        y0, style = net(X[irange])
        y[irange] = y0[:,:,:,:].asnumpy()
        if k==0:
            styles = np.zeros(style.shape[1], np.float32)
        styles += style.asnumpy().sum(axis=0)
    styles /= IMG.shape[0]

    if True:
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


    Navg = np.zeros((Ly,Lx))
    ytiled = np.zeros((nout, Ly, Lx), 'float32')
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    sig = 10.
    mask = 1/(1 + np.exp((xm - (bsize/2-20.)) / sig))
    mask = mask * mask[:, np.newaxis]

    for j in range(len(ysub)):
        ytiled[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += mask
    ytiled /=Navg

    ytiled = ytiled[:,:Ly0, :Lx0]
    return ytiled, styles

def run_resize_tile(net, img, rsz, bsize=224, device=mx.cpu()):
    Ly = int(img.shape[0] * rsz)
    Lx = int(img.shape[1] * rsz)

    IMG = cv2.resize(img, (Lx, Ly))
    if IMG.ndim<3:
        IMG = np.expand_dims(IMG, axis=-1)
    imgi = np.transpose(IMG, (2,0,1))
    imgi, ysub, xsub = pad_image(imgi)

    y,style = run_tile(net, imgi, bsize, device=device)
    yup = np.transpose(y, (1,2,0))
    yup = yup[np.ix_(ysub, xsub, np.arange(yup.shape[-1]))]

    yup = cv2.resize(yup, (img.shape[1], img.shape[0]))
    return yup, style

def run_resize(net, img, rsz, device=mx.cpu()):
    Ly = int(img.shape[0] * rsz)
    Lx = int(img.shape[1] * rsz)


    IMG = cv2.resize(img, (Lx, Ly))
    if IMG.ndim<3:
        IMG = np.expand_dims(IMG, axis=-1)

    imgi, ysub, xsub = pad_image(np.transpose(IMG, (2,0,1)))

    imgi = np.expand_dims(imgi, 0)

    X = nd.array(imgi, ctx=device)
    y, style = net(X)
    y = y.asnumpy()
    style = style.asnumpy()

    yup = np.transpose(y[0], (1,2,0))
    yup = yup[np.ix_(ysub, xsub, np.arange(y.shape[1]))]

    yup = cv2.resize(yup, (img.shape[1], img.shape[0]))
    return yup, style

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

def process_cells(M0, npix=20):
    unq, ic = np.unique(M0, return_counts=True)
    for j in range(len(unq)):
        if ic[j]<npix:
            M0[M0==unq[j]] = 0
    return M0

def run_dynamics(y, niter = 200, eta=.1,p=0.):
    x0, y0 = np.meshgrid(np.arange(y.shape[-1]),  np.arange(y.shape[-2]))

    y = np.squeeze(y)
    xs, ys = x0.copy(), y0.copy()
    yout = np.zeros(y.shape)
    nc, Ly, Lx = y.shape

    dx = y[0,:,:]
    dy = y[1,:,:]

    ox = dx
    oy = dy
    for j in range(niter):
        xi = xs.astype('int')
        yi = ys.astype('int')
        xi = np.clip(xi, 0, Lx-1)
        yi = np.clip(yi, 0, Ly-1)

        ox = p * ox + dx[yi, xi]
        oy = p * oy + dy[yi, xi]
        xs = np.clip(xs - eta*ox, 0, Lx-1)
        ys = np.clip(ys - eta*oy, 0, Ly-1)
    yout[0] = ys
    yout[1] = xs
    return yout
