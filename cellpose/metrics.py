import numpy as np
from . import utils, dynamics
from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve


def mask_ious(masks_true, masks_pred):
    """ return best-matched masks """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(masks_true.max(), 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds

def boundary_scores(masks_true, masks_pred, scales):
    """ boundary precision / recall / Fscore """
    diams = [utils.diameters(lbl)[0] for lbl in masks_true]
    precision = np.zeros((len(scales), len(masks_true)))
    recall = np.zeros((len(scales), len(masks_true)))
    fscore = np.zeros((len(scales), len(masks_true)))
    for j, scale in enumerate(scales):
        for n in range(len(masks_true)):
            diam = max(1, scale * diams[n])
            rs, ys, xs = utils.circleMask([int(np.ceil(diam)), int(np.ceil(diam))])
            filt = (rs <= diam).astype(np.float32)
            otrue = utils.masks_to_outlines(masks_true[n])
            otrue = convolve(otrue, filt)
            opred = utils.masks_to_outlines(masks_pred[n])
            opred = convolve(opred, filt)
            tp = np.logical_and(otrue==1, opred==1).sum()
            fp = np.logical_and(otrue==0, opred==1).sum()
            fn = np.logical_and(otrue==1, opred==0).sum()
            precision[j,n] = tp / (tp + fp)
            recall[j,n] = tp / (tp + fn)
        fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    return precision, recall, fscore


def aggregated_jaccard_index(masks_true, masks_pred):
    """ AJI = intersection of all matched masks / union of all masks 
    
    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    aji : aggregated jaccard index for each set of masks

    """

    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n]>0, masks_pred[n]>0).sum()
        overlap = overlap[inds[preds>0]+1, preds[preds>0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji 


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
        
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def flow_error(maski, dP_net):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------
    
    maski: ND-array (int) 
        masks produced from running dynamics on dP_net, 
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float) 
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
    # flows predicted from estimated masks
    dP_masks,_ = dynamics.masks_to_flows(maski)
    iun = np.unique(maski)[1:]
    flow_errors=np.zeros((len(iun),))
    for i,iu in enumerate(iun):
        ii = maski==iu
        if dP_masks.shape[0]==2:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2).mean()
        else:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2 * 0.5
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2
                            + (dP_masks[2][ii] - dP_net[2][ii]/5.)**2).mean()
    return flow_errors, dP_masks
