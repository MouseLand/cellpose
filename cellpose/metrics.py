"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import numpy as np
from . import utils
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix 


def mask_ious(masks_true, masks_pred):
    """Return best-matched masks."""
    iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind, pred_ind]
    preds = np.zeros(masks_true.max(), "int")
    preds[true_ind] = pred_ind + 1
    return iout, preds


def boundary_scores(masks_true, masks_pred, scales):
    """
    Calculate boundary precision, recall, and F-score.

    Args:
        masks_true (list): List of true masks.
        masks_pred (list): List of predicted masks.
        scales (list): List of scales.

    Returns:
        tuple: A tuple containing precision, recall, and F-score arrays.
    """
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
            tp = np.logical_and(otrue == 1, opred == 1).sum()
            fp = np.logical_and(otrue == 0, opred == 1).sum()
            fn = np.logical_and(otrue == 1, opred == 0).sum()
            precision[j, n] = tp / (tp + fp)
            recall[j, n] = tp / (tp + fn)
        fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    return precision, recall, fscore


def aggregated_jaccard_index(masks_true, masks_pred):
    """ 
    AJI = intersection of all matched masks / union of all masks 
    
    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)): 
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)): 
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        aji (float): aggregated jaccard index for each set of masks
    """
    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n] > 0, masks_pred[n] > 0).sum()
        overlap = overlap[inds[preds > 0] + 1, preds[preds > 0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ 
    Average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)): 
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)): 
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]): 
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]): 
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]): 
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]): 
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array([len(np.unique(mt)) - 1 for mt in masks_true])
    n_pred = np.array([len(np.unique(mp)) - 1 for mp in masks_pred])

    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn


def _intersection_over_union(masks_true, masks_pred):
    """Calculate the intersection over union of all mask pairs.

    Parameters:
        masks_true (np.ndarray, int): Ground truth masks, where 0=NO masks; 1,2... are mask labels.
        masks_pred (np.ndarray, int): Predicted masks, where 0=NO masks; 1,2... are mask labels.

    Returns:
        iou (np.ndarray, float): Matrix of IOU pairs of size [x.max()+1, y.max()+1].

    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 
    """
    if masks_true.size != masks_pred.size:
        raise ValueError("masks_true.size != masks_pred.size")
    overlap = csr_matrix((np.ones((masks_true.size,), "int"), 
                         (masks_true.flatten(), masks_pred.flatten())),
                         shape=(masks_true.max()+1, masks_pred.max()+1))
    overlap = overlap.toarray()
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """Calculate the true positive at threshold th.

    Args:
        iou (float, np.ndarray): Array of IOU pairs.
        th (float): Threshold on IOU for positive label.

    Returns:
        tp (float): Number of true positives at threshold.

    How it works:
        (1) Find minimum number of masks.
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...).
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs from these pairings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp
