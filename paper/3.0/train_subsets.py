import time, os
import numpy as np
from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch, denoise
from cellpose.transforms import normalize_img
from pathlib import Path
import torch
from torch import nn
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Cellpose Command Line Parameters")
    parser.add_argument("--nsub", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_epochs", default=2000, type=int)

    args = parser.parse_args()
    n_epochs = args.n_epochs
    nsub = args.nsub
    seed = args.seed
    print(n_epochs, nsub, seed)

    root = Path("/groups/stringer/stringerlab/datasets_cellpose/images_cyto2/")
    #pretrained_model = str(root / "cyto3")
    batch_size = 8

    io.logger_setup()

    device = torch.device("cuda")
    ntrain = 796
    np.random.seed(seed)
    if nsub > 0:
        iperm = np.random.permutation(ntrain)[:nsub]
    else:
        iperm = np.arange(ntrain)
    
    # keep green channel
    train_data = []
    for i in iperm:
        img = io.imread(root / "train" / f"{i:03d}_img.tif")
        if img.ndim > 2:
            img = img[0]
        train_data.append(np.maximum(transforms.normalize99(img), 0)[np.newaxis,:,:])

    train_labels = [io.imread(root / "train" / f"{i:03d}_img_flows.tif") for i in iperm]

    test_data = []
    for i in range(68):
        img = io.imread(root / "test" / f"{i:03d}_img.tif")
        if img.ndim > 2:
            img = img[0]
        test_data.append(np.maximum(transforms.normalize99(img), 0)[np.newaxis,:,:])
    test_labels = [io.imread(root / "test" / f"{i:03d}_img_flows.tif") for i in range(91)]

    model = denoise.DenoiseModel(gpu=True, nchan=1, pretrained_model=None)

    (root / "models").mkdir(exist_ok=True)
    # poisson training
    model_path, train_losses, test_losses = denoise.train(model.net, train_data=train_data, train_labels=train_labels, 
                            test_data=test_data, test_labels=test_labels,
                            save_path=root / "models", blur=0., gblur=0.5,
                            iso=True, lam=[1.,1.5,0],
                            downsample=0., poisson=0.8, beta=0.7   , n_epochs=n_epochs, 
                            learning_rate=0.001, weight_decay=1e-5,
                            seg_model_type="cyto2", model_name=f"denoise_cyto2_{nsub}_{seed}_{n_epochs}")


if __name__ == "__main__":
    main()

def save_results(folder, sroot):
    nsubs = 2 ** np.arange(1, 10)
    nsubs = np.vstack((nsubs, 796))
    thresholds = np.arange(0.5, 1.05, 0.05)
    seg_model = models.CellposeModel(gpu=True, model_type="cyto2")
    noise_type = "poisson"
    ctype = "cyto2"

    folder_name = ctype
    diam_mean = 30
    root = Path(folder) / f"images_{folder_name}/"                                                                                        
    model_name = "cyto2"

    ### cellpose enhance
    dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                    allow_pickle=True).item()
    test_noisy = dat["test_noisy"][:68]
    masks_true = dat["masks_true"][:68]
    diam_test = dat["diam_test"][:68] if "diam_test" in dat else 30. * np.ones(
        len(test_noisy))

    aps = np.zeros((len(nsubs), 5, len(thresholds)))
    for k, nsub in enumerate(nsubs):
        if nsub != 796:
            continue
        ni = nsub if nsub < 796 else 0
        for seed in range(5):
            si = seed if nsub < 796 else f"{seed}_2000"
            dn_model = denoise.DenoiseModel(gpu=True, nchan=1, diam_mean=diam_mean,
                                    pretrained_model=str(sroot / f"denoise_cyto2_{ni}_{si}"))
            imgs2 = dn_model.eval([test_noisy[i][0] for i in range(len(test_noisy))],
                                diameter=diam_test, channel_axis=0)

            masks2, flows2, styles2 = seg_model.eval(imgs2, channels=[1, 0],
                                                    diameter=diam_test, channel_axis=-1,
                                                    normalize=True)
            
            ap, tp, fp, fn = metrics.average_precision(masks_true, masks2, threshold=thresholds)
            aps[k, seed] = ap.mean(axis=0)
            print(f"{nsub} AP@0.5 \t = {ap[:,0].mean(axis=0):.3f}")
            
    np.save("nsubs_aps.npy", aps)

    n_epochss = np.array([100, 200, 400, 800, 1600, 2000, 3200])
    nsub, seed = 0, 0
    thresholds = np.arange(0.5, 1.05, 0.05)
    aps = np.zeros((len(n_epochss), 5, len(thresholds)))
    for k, n_epochs in enumerate(n_epochss):
        for seed in range(5):
            dn_model = denoise.DenoiseModel(gpu=True, nchan=1, diam_mean=diam_mean,
                                    pretrained_model=str(sroot / f"denoise_cyto2_{nsub}_{seed}_{n_epochs}"))
            imgs2 = dn_model.eval([test_noisy[i][0] for i in range(len(test_noisy))],
                                diameter=diam_test, channel_axis=0)

            masks2, flows2, styles2 = seg_model.eval(imgs2, channels=[1, 0],
                                                    diameter=diam_test, channel_axis=-1,
                                                    normalize=True)
            
            ap, tp, fp, fn = metrics.average_precision(masks_true, masks2, threshold=thresholds)
            aps[k, seed] = ap.mean(axis=0)
            print(f"{n_epochs} AP@0.5 \t = {ap[:,0].mean(axis=0):.3f}")
            
    np.save("n_epochs_aps.npy", aps)