from scipy.ndimage.filters import maximum_filter1d
import numpy as np
import time
import mxnet as mx
import mxnet.ndarray as nd
from numba import njit, float32, int32, vectorize
from . import utils, metrics

@njit('(float32[:], float32[:,:], int32[:], int32[:],int32, int32)')
def neighbor_flow(p, alpha, y, x, Lx, niter):
    for t in range(niter):
        p[y*Lx + x] = (alpha[0] * p[y*Lx + x-1]  +
                       alpha[1] * p[y*Lx + x+1]  +
                       alpha[2] * p[(y-1)*Lx + x] +
                       alpha[3] * p[(y+1)*Lx + x] )
    return p

def neighbor_dynamics(dP, pC, niter=200):
    dP *= pC
    _,Ly,Lx = dP.shape
    p = np.meshgrid(np.arange(Ly+2), np.arange(Lx+2), indexing='ij')
    p = np.array(p).astype(np.float32)
    p = np.reshape(p, (2,-1))

    alpha = np.zeros((4, Ly, Lx), np.float32)
    for k in range(2):
        alpha[2*k] = np.maximum(0, -dP[k])
        alpha[2*k+1] = np.maximum(0, dP[k])
    #alpha += 1e-3
    #alpha /= (np.abs(alpha).sum(axis=0))
    y,x = np.nonzero(pC)
    alpha = alpha[:,y,x]
    y = y.astype(np.int32) + 1 # add 1 for padding
    x = x.astype(np.int32) + 1

    # only keep alpha positive for cell pixels
    # pad by +/- 1 pixel
    #pcell = np.zeros((Ly+2,Lx+2), np.bool)
    #pcell[1:-1,1:-1] = pC
    #alpha[0] *= pcell[y,x-1]
    #alpha[1] *= pcell[y,x+1]
    #alpha[2] *= pcell[y-1,x]
    #alpha[3] *= pcell[y+1,x]
    alpha /= (1e-7 + np.abs(alpha).sum(axis=0)) # normalize
    alpha = np.reshape(alpha, (4,-1))
    niter = np.int32(niter)
    for k in range(2):
        p[k] = neighbor_flow(p[k], alpha, y, x, np.int32(Lx+2), niter)
    p = np.reshape(p, (2,Ly+2,Lx+2))
    p = p[:,1:-1,1:-1]
    return p

def neighbor_dynamics2(alpha0, pC, niter=200):
    #alpha0 *= pC
    _,Ly,Lx = alpha0.shape
    p = np.meshgrid(np.arange(Ly+2), np.arange(Lx+2), indexing='ij')
    p = np.array(p).astype(np.float32)
    p = np.reshape(p, (2,-1))

    alpha = alpha0.copy()
    alpha += 1e-7
    alpha /= (np.abs(alpha).sum(axis=0))
    y,x = np.nonzero(pC)
    alpha = alpha[:,y,x]
    y = y.astype(np.int32) + 1 # add 1 for padding
    x = x.astype(np.int32) + 1

    # only keep alpha positive for cell pixels
    # pad by +/- 1 pixel
    pcell = np.zeros((Ly+2,Lx+2), np.bool)
    pcell[1:-1,1:-1] = pC

    dx = np.array([1, 0,  1,  1, -1,  0, -1, -1])
    dy = np.array([0, 1, -1,  1,  0, -1,  1, -1])

    for j in range(8):
        alpha[j] *= pcell[y+dy[j],x+dx[j]]

    alpha /= (1e-7 + np.abs(alpha).sum(axis=0)) # normalize
    alpha = np.reshape(alpha, (8,-1))
    niter = np.int32(niter)
    for k in range(2):
        p[k] = neighbor_flow(p[k], alpha, y, x, dy, dx, np.int32(Lx+2), niter)
    p = np.reshape(p, (2,Ly+2,Lx+2))
    p = p[:,1:-1,1:-1]
    return p

@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)')
def extend_centers(T,y,x,ymed,xmed,Lx, niter):
    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T

def masks_to_flows(masks):
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)
    # remove redundant labels
    Y = np.reshape(np.unique(masks, return_inverse=True)[1], (Ly, Lx))
    nmask = Y.max()
    # pad T0 and mask by 2
    T = np.zeros((Ly+2)*(Lx+2), np.float64)

    Lx += 2
    dia = utils.diameters(masks)[0]
    s2 = (.15 * dia)**2
    for k in range(nmask):
        y,x = np.nonzero(Y==(k+1))
        y = y.astype(np.int32) + 1
        x = x.astype(np.int32) + 1
        ymed = np.median(y)
        xmed = np.median(x)
        imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
        xmed = x[imin]
        ymed = y[imin]

        d2 = (x-xmed)**2 + (y-ymed)**2
        mu_c[y-1,x-1] = np.exp(-d2/s2)

        niter = 2*np.int32(np.ptp(x) + np.ptp(y))
        T = extend_centers(T, y, x, ymed, xmed, np.int32(Lx), niter)

        T[(y+1)*Lx + x+1] = np.log(1.+T[(y+1)*Lx + x+1])

        dy = T[(y+1)*Lx + x] - T[(y-1)*Lx + x]
        dx = T[y*Lx + x+1] - T[y*Lx + x-1]
        mu[:,y-1, x-1] = np.stack((dy,dx))
        T[y*Lx + x] = 0

    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)
    return mu, mu_c


@njit('(float32[:,:,:], float32[:,:,:], int32[:], int32)')
def steps(p, dP, shape, niter):
    for t in range(200):
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                for k in range(p.shape[2]):
                    p[i,j,k] = min(shape[i], max(0,
                                p[i,j,k] - 0.1*dP[i,int(p[0,j,k]),int(p[1,j,k])]))

@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)')
def steps3D(p, dP, inds, niter):
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            z = inds[j,0]
            y = inds[j,1]
            x = inds[j,2]
            p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
            p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] - dP[0,p0,p1,p2]))
            p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] - dP[1,p0,p1,p2]))
            p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] - dP[2,p0,p1,p2]))
    return p

@njit('(float32[:,:,:],float32[:,:,:], int32[:,:], int32)')
def steps2D(p, dP, inds, niter):
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            p[0,y,x] = min(shape[0]-1, max(0, p[0,y,x] - dP[0,p0,p1]))
            p[1,y,x] = min(shape[1]-1, max(0, p[1,y,x] - dP[1,p0,p1]))
    return p

@njit('(float32[:],float32[:],float32[:],float32[:],int32[:],int32[:],int32[:], int32)')
def steps2D2(dy, dx, py, px, y, x, shape, niter):
    Lx = shape[1]
    for t in range(niter):
        #pi = p.astype(np.int32)
        p0, p1 = py[y*Lx + x].astype(np.int32), px[y*Lx + x].astype(np.int32)
        py[y*Lx + x] = np.minimum(shape[0]-1, np.maximum(0, py[y*Lx + x] - dy[p0*Lx + p1]))
        px[y*Lx + x] = np.minimum(shape[1]-1, np.maximum(0, px[y*Lx + x] - dx[p0*Lx + p1]))
    return py, px

def follow_flows(dP, niter=200, do_3D=False):
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.int32(niter)
    if do_3D:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                np.arange(shape[2]), indexing='ij')
        p = np.array(p)
        inds = np.array(np.nonzero((dP[0]!=0))).astype(np.int32).T
        p = steps3D(p.astype(np.float32), dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero((dP[0]!=0))).astype(np.int32).T
        p = steps2D(p, dP, inds, niter)
        # no loop (slow...)
        #p = steps2D2(dP[0].flatten(), dP[1].flatten(),
         #            p[0].flatten(), p[1].flatten(), inds[0], inds[1],
          #           shape, niter)

    return p

def remove_bad_flow_masks(masks, flows, threshold=0.5):
    """ remove masks which have inconsistent flows """
    merrors = metrics.flow_error(masks, flows)
    badi = 1+(merrors>.5).nonzero()[0]
    #nbad =len(badi)
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, rpad=20, nmax=20, threshold=0.5, flows=None):
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    ibad = np.ones(len(pix), 'bool')
    for k in range(len(pix)):
        #print(pix[k][0].size)
        if pix[k][0].size<nmax:
            ibad[k] = 0

    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        #).sum())

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]
    _,M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    if threshold > 0 and flows is not None:
        M0 = remove_bad_flow_masks(M0, flows, threshold=threshold)
    _,M0 = np.unique(M0, return_inverse=True)
    #mlbl = np.random.permutation(M0.max()) + 1
    #mlbl = np.append(0, mlbl)
    #M0 = mlbl[M0]
    M0 = np.reshape(M0, shape0)
    return M0



def new_flow(Y, nuclei=None, device=mx.cpu()):
    w = nd.ones((1,1,3,3), ctx = device)
    bias = nd.zeros((32,), ctx = device)
    bias0 = nd.zeros((1,), ctx = device)
    #w[0,0,2,2] = 0
    #w[0,0,0,0] = 0
    #w[0,0,0,2] = 0
    #w[0,0,2,0] = 0

    w = w/nd.sum(w)
    Ly, Lx = Y.shape
    mu = np.zeros((2, Ly,Lx))
    edge = np.zeros((Ly,Lx))
    unq = np.unique(Y)
    nmask = len(unq)-1

    _, N = np.unique(Y, return_counts = True)
    R = np.median(N[1:]**.5)
    #print(R)

    for j in range(nmask):
        mask = (Y==unq[j+1])
        y,x = (Y==unq[j+1]).nonzero()

        y0 = np.min(y)
        x0 = np.min(x)

        if nuclei is not None:
            M = nuclei[y,x]
            M = M - M.min() + 1e-3
            M = M / M.sum()
            ymed = np.round(np.dot(M , (y-y0))).astype('int32')
            xmed = np.round(np.dot(M , (x-x0))).astype('int32')

        y = y-y0
        x = x-x0
        Ly, Lx = np.max(y)+1, np.max(x)+1

        T0 = nd.zeros((1,1,Ly+2,Lx+2), ctx=device)
        T0[0,0,y+1,x+1] = 1

        ff =  T0 * (nd.Convolution(T0, w, bias0, kernel = (3,3), pad=(1,1), num_filter = 1) <.95)
        ybound, xbound = np.nonzero(ff[0,0].asnumpy())
        ds = ((y[:, np.newaxis] + 1 - ybound)**2 + (x[:, np.newaxis] + 1 - xbound)**2)**.5
        dmin = np.min(ds, axis=1)
        edge[y+y0,x+x0] = dmin

        #imin = np.argmin( - np.min(ds, axis=1))

        if nuclei is None:
            if False:
                mask = nd.zeros((Ly+2,Lx+2), ctx=device)
                mask[y+1,x+1] = 1

                T0 = nd.zeros((1,1,Ly+2,Lx+2), ctx=device)
                T0[0,0,y+1,x+1] = 1

                for j in range(Ly+Lx):
                    T0 =  nd.Convolution(T0, w, bias0, kernel = (3,3), pad=(1,1), num_filter = 1)
                    T0 = T0 * mask
                    T0 = T0/T0.mean()
                xy = np.unravel_index(np.argmax((mask * T0[0, 0]).asnumpy().squeeze()), mask.shape)
                ymed, xmed = xy[0]-1, xy[1]-1
            elif False:
                ydist = y
                xdist = x
                if len(ydist)>200:
                    ix = np.random.permutation(len(y))[:200]
                    ydist = ydist[ix]
                    xdist = xdist[ix]
                ds = ((ydist - y[:, np.newaxis])**2 + (xdist - x[:, np.newaxis])**2)**.5
                imin = np.argmin(np.mean(ds, axis=1))
                ymed = y[imin]
                xmed = x[imin]
            else:
                ymed = int(np.median(y))
                xmed = int(np.median(x))
                imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
                xmed = x[imin]
                ymed = y[imin]

        mask = nd.zeros((1,1,Ly+2,Lx+2), ctx=device)
        T0 = nd.zeros((1,1,Ly+2,Lx+2), ctx = device)
        T0[0,0,ymed+1, xmed+1] += 1.
        mask[0,0,y+1,x+1] = 1

        T = T0.copy()
        T = nd.zeros((1,1,Ly+2,Lx+2), ctx=device) # T0.copy()
        for t in range(Ly+Lx):
            T = T * mask + T0
            T = nd.Convolution(T, w, bias0, kernel = (3,3), pad=(1,1), num_filter = 1)


        tnp = T[0,0].asnumpy()

        dx = tnp[1:-1, 2:] - tnp[1:-1, :-2]
        dy = tnp[2:, 1:-1] - tnp[:-2, 1:-1]
        D = np.stack((dx,dy))

        mu[:, y+y0,x+x0] = mu[:, y+y0,x+x0] + D[:, y,x]
    mu = mu / (1e-20 + np.sum(mu**2, axis=0)**.5)

    return mu[0], mu[1], edge
