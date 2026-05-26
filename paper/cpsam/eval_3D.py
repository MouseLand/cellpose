from segment_anything import sam_model_registry
from cellpose import io, metrics, models, utils, train, transforms, dynamics
import time
from tqdm import trange
from torch import nn 
import torch.nn.functional as F
import torch
import numpy as np
from pathlib import Path
from natsort import natsorted
import argparse
import os
import train_subsets

def eval_3D(root, ntrain, mtype, seed):
    print(root, ntrain, mtype)
    io.logger_setup()
    # mstr = "sam" if transformer else "cyto3"
    transformer = True if mtype=='cpsam' or 'cpdino' in mtype else False
    netstr = f"{mtype}_seed_{seed}_ntrain_{ntrain}"
    print(netstr)

    dd = "../" if "root" in str(root) or "ovules" in str(root) else ""
    if ntrain != -1:
        model = models.CellposeModel(gpu=True, pretrained_model=root / f"models/{netstr}")
    else:
        print(dd)
        if mtype=='cpsam':
            model = models.CellposeModel(gpu=True, pretrained_model=root / f"../{dd}models/cpsam8_2000_162519454")
        elif mtype=='cpdino':
            model = models.CellposeModel(gpu=True, pretrained_model=root / f"../{dd}models/cp2000_0.0002_0.4_0.1_84002256")
        elif mtype=='cpdino-vitb':
            model = models.CellposeModel(gpu=True, pretrained_model=root / f"../{dd}models/cp2000_0.0002_0.4_0.1_636227846")
        else:
            model = models.Cellpose(gpu=True, model_type="cyto3")

    test_files = [f for f in (root / "test_3D").glob("*.tif") if "_masks" not in f.name]

    masks_gt_all = []
    masks_pred_all = []
    for i, f in enumerate(test_files):
        img = io.imread(f)
        masks_gt = io.imread(str(f).replace(".tif", "_masks.tif"))    
        print(i, img.shape, masks_gt.shape)

        if transformer:
            masks, flows, styles = model.eval(np.stack((img, 0*img, 0*img), axis=-1), 
                                        channel_axis=-1, z_axis=0, niter=1000,
                                        flow3D_smooth=2, diameter=30,
                                        bsize=None, batch_size=64, 
                                        channels=None, do_3D=True, min_size=1000)
        else:
             masks, flows, styles = model.eval(img, diameter=model.net.diam_labels.item(), 
                                            flow3D_smooth=2, z_axis=0, bsize=224, niter=1000,
                                            batch_size=64, channels=[1,0], do_3D=True, min_size=1000)
        if seed == 0:
            io.imsave(root / f"test_3D/{f.stem}_ntrain_{ntrain}_{mtype}_masks.tif", masks)
        
        ap, tp, fp, fn = metrics.average_precision(masks_gt, masks)
        print(i, ap[0], tp[0], fp[0], fn[0])
        masks_gt_all.append(masks_gt)
        masks_pred_all.append(masks)

    threshold = np.arange(0.5, 1, 0.05)
    
    ap, tp, fp, fn = metrics.average_precision(masks_gt_all, masks_pred_all, threshold=threshold)
    print(ap[:, [0, 5, 8]].mean(axis=0))
    np.save(root / f"models/{netstr}_3D_AP_TP_FP_FN.npy", {
                "threshold": threshold, "ap": ap, "tp": tp, "fp": fp, "fn": fn,
                "test_files": test_files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--mtype", type=str, default='cyto3')
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    ntrain = args.ntrain
    seed = args.seed
    root = Path(args.root)
    eval_3D(root, ntrain=ntrain, mtype=args.mtype, seed=seed)
        
     
if __name__=="__main__":
    main()
