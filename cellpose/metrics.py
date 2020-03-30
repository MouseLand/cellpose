import numpy as np
from . import utils, dynamics
from numba import jit
from scipy.optimize import linear_sum_assignment

@jit(nopython=True)
def label_overlap(x, y):
    """ fast function thanks to stardist """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def intersection_over_union(masks_true, masks_pred):
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    return iou

def true_positive(iou, th):
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    for n in range(len(masks_true)):
        iou = intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
        for k,th in enumerate(threshold):
            tp[n,k] = true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
    return ap, tp, fp, fn

def flow_error(maski, dP_net):
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
    # flows predicted from estimated masks
    dP_masks,_ = dynamics.masks_to_flows(maski)
    iun = np.unique(maski)[1:]
    flow_error=np.zeros((len(iun),))
    for i,iu in enumerate(iun):
        ii = maski==iu
        if dP_masks.shape[0]==2:
            flow_error[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2).mean()
        else:
            flow_error[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2 * 0.5
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2
                            + (dP_masks[2][ii] - dP_net[2][ii]/5.)**2).mean()
    return flow_error, dP_masks

def total_variation_loss(x):
    a = nd.square(x[:, :, :-1, :-1] - x[:, :, 1:, :-1])
    b = nd.square(x[:, :, :-1, :-1] - x[:, :, :-1, 1:])
    return nd.sum(nd.mean(nd.power(a + b, 1.25), axis=(2,3)))
