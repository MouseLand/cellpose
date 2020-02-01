import numpy as np
from . import utils, dynamics
from numba import jit

@jit(nopython=True)
def label_overlap(x, y):
    """ fast function thanks to stardist """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap


def flow_error(maski, flows):
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1],
                       (maski.shape[0], maski.shape[1]))
    # flows
    a,_ = dynamics.masks_to_flows(maski)
    dY,dX = a
    #dY,dX,_=dynamics.new_flow(maski)
    iun = np.unique(maski)[1:]
    flow_error=np.zeros((len(iun),))
    for i,iu in enumerate(iun):
        ii = maski==iu
        flow_error[i] = ((dX[ii] - flows[1][ii]/5)**2 +
                         (dY[ii] - flows[0][ii]/5)**2).mean()
    return flow_error

def intersection_over_union(masks, labels):
    # IOU
    overlap = label_overlap(masks, labels.astype(np.int32))
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou = iou.max(axis=1)
    iou = iou[1:]
    return iou

def total_variation_loss(x):
    a = nd.square(x[:, :, :-1, :-1] - x[:, :, 1:, :-1])
    b = nd.square(x[:, :, :-1, :-1] - x[:, :, :-1, 1:])
    return nd.sum(nd.mean(nd.power(a + b, 1.25), axis=(2,3)))
