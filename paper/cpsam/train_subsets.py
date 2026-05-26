from cellpose import io, metrics, models, utils, train, transforms, dynamics
import time
from tqdm import trange
import torch
import numpy as np
from pathlib import Path
from natsort import natsorted
import argparse
import os

torch.backends.cuda.matmul.allow_tf32 = True


def train_subsets(root, seed=0, ntrain=1,
                  mtype='cpsam', save_results = True, 
                  save_test_masks = False, keep_model=False,
                  device=torch.device("cuda")):
    io.logger_setup()
    print(device)
    print(f"seed = {seed}, ntrain = {ntrain}")
    # mtype = "sam" if transformer else "cyto3"
    transformer = True if mtype=='cpsam' or 'cpdino' in mtype else False
    netstr = f"{mtype}_seed_{seed}_ntrain_{ntrain}"
    
    train_files = (root / "train").glob("*.tif")
    train_files = natsorted([tf for tf in train_files if "_masks" not in str(tf)])
    ntrain = len(train_files) if ntrain == 0 else ntrain

    np.random.seed(seed + 5)
    itrain = np.random.permutation(len(train_files))[:ntrain]
    train_files = [train_files[i] for i in itrain]

    test_files = (root / "test").glob("*.tif")  
    test_files = natsorted([tf for tf in test_files if "_masks" not in str(tf)])
    print(f"nimg_train = {len(train_files)}, nimg_test = {len(test_files)}")

    train_data, test_data = [], []
    k = 0 
    print("loading images")
    for k in range(2 if len(test_files) > 0 else 1):
        tf = train_files if k==0 else test_files
        for i in trange(len(tf)):
            img = io.imread(tf[i])
            img = np.tile(img[np.newaxis,:,:], (3,1,1))
            img[1:] = 0
            if k==0:
                train_data.append(img)
            else:
                test_data.append(img)
    print("loading labels and computing flows")
    train_masks = [io.imread(str(train_files[i])[:-4] + f'_masks.tif') for i in trange(len(train_files))]
    train_labels = dynamics.labels_to_flows(train_masks, device=device)
    if len(test_files) > 0:
        test_masks = [io.imread(str(test_files[i])[:-4] + f'_masks.tif') for i in trange(len(test_files))]
        test_labels = dynamics.labels_to_flows(test_masks, device=device)
    else:
        test_data = None
        test_labels = None
        
    dd = "../" if "root" in str(root) or "ovules" in str(root) else ""
    print(dd)
    if mtype=='cpsam':
        model = models.CellposeModel(gpu=True, pretrained_model=root / f"../{dd}models/cpsam8_2000_162519454")
    elif mtype=='cpdino':
        model = models.CellposeModel(gpu=True, pretrained_model=root / f"../{dd}models/cp2000_0.0002_0.4_0.1_84002256")
    elif mtype=='cpdino-vitb':
        model = models.CellposeModel(gpu=True, pretrained_model=root / f"../{dd}models/cp2000_0.0002_0.4_0.1_636227846")
    else:
        model = models.Cellpose(gpu=True, model_type="cyto3")
        
    # channels = None if transformer else [1,0]
    
    if ntrain >= 0:
        learning_rate = 1e-5 if transformer else 5e-3
        weight_decay = 0.1 if transformer else 1e-4
        SGD = False #not transformer
        # soft_start = not transformer
        batch_size = 1 if transformer else 8
        n_epochs = 100 if transformer else 300 #(100 if ntrain < 16 else 300)

        rescale = not transformer
        scale_range = 0.5 #None if transformer else 0.5
        max_nimg = 800 
        nimg_per_epoch_min = 8

        # train
        tic = time.time()
        out = train.train_seg(model.net, train_data=train_data, train_labels=train_labels, 
                              test_data=test_data, test_labels=test_labels,
                            learning_rate=learning_rate, weight_decay=weight_decay,
                            SGD=SGD, batch_size=batch_size, n_epochs=n_epochs,
                            nimg_per_epoch=min(max_nimg, max(nimg_per_epoch_min, len(train_data))), 
                            rescale=rescale, 
                            scale_range=scale_range, save_path=root,
                            nimg_test_per_epoch=len(test_data), model_name=netstr,
                              min_train_masks=0)
    
        toc = time.time() - tic
        print(f"training time: {toc:.2f}s")
        filename, train_losses, test_losses = out
        filename = str(filename)
    else:
        toc = 0
        filename, train_losses, test_losses = None, None, None
    # evaluate
    if ntrain >= 0:
        diameter = model.net.diam_labels.item() if not transformer else 30.
    else:
        diameter = 0

    masks_pred = model.eval(test_data, diameter=diameter, 
                            batch_size=64)[0]
    masks_gt = [tl.astype("uint16") for tl in test_masks]
    threshold = np.arange(0.5, 1.0, 0.05)
    ap, tp, fp, fn = metrics.average_precision(masks_gt, masks_pred, threshold=threshold)
    print(ap[:, [0, 5, 8]].mean(axis=0))
    if save_results:
        np.save(root / f"models/{netstr}_AP_TP_FP_FN.npy", {
                "threshold": threshold, "ap": ap, "tp": tp, "fp": fp, "fn": fn,
                "ntrain_masks": len(train_data), "test_files": test_files,
                "diam_labels": diameter, "model_path": filename,
                "train_losses": train_losses, "test_losses": test_losses,
                "test_masks_pred": masks_pred if save_test_masks else None,
                "train_time": toc,
            "nrois": np.array([td.max() for td in train_masks]).sum()})
        
    if ntrain>=0 and not keep_model:
        os.remove(filename)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mtype", type=str, default='cyto3')
    
    args = parser.parse_args()
    ntrain = args.ntrain
    seed = args.seed
    root = Path(args.root)
    train_subsets(root, ntrain=ntrain, seed=seed, mtype=args.mtype,
                  save_test_masks=True if seed==0 else False,
                  keep_model=True)


if __name__=="__main__":
    main()
