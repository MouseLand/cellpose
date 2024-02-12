"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt 
from pathlib import Path
import torch
from cellpose import models, metrics, io, transforms, denoise

# uses tensorflow
# pip install csbdeep
from csbdeep.data import RawData, create_patches
from csbdeep.data import no_background_patches, norm_percentiles, sample_percentiles
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

n_samples=20

def CIL_dataset(root):

    # keep green channel, first 89 images are cellimagelibrary
    train_data = []
    for i in range(89):
        img = io.imread(root / "train" / f"{i:03d}_img.tif")
        if img.ndim > 2:
            img = img[0]
        train_data.append(np.maximum(transforms.normalize99(img), 0)[np.newaxis,:,:])
        
    # first 11 are cellimagelibrary
    test_data = []
    for i in range(11):
        img = io.imread(root / "test" / f"{i:03d}_img.tif")
        if img.ndim > 2:
            img = img[0]
        test_data.append(np.maximum(transforms.normalize99(img), 0)[np.newaxis,:,:])
        
    im_train_all = []
    gt_train_all = []
    im_val_all = []
    gt_val_all = []

    for i in trange(len(train_data)):
        img = train_data[i].copy().astype("float32")
        #print(img.shape)
        Ly, Lx = img.shape[-2:]
        img0 = torch.from_numpy(img).unsqueeze(0)
        gt_train = np.stack((img[:,:Ly//2,:Lx//2], img[:,:Ly//2,Lx//2:2*(Lx//2)],
                                img[:,Ly//2:2*(Ly//2),:Lx//2]), axis=0)
        gt_val = img[:, Ly//2:, Lx//2:]
            
        for k in range(n_samples):
            imr = denoise.add_noise(img0, poisson=0.8, beta=0.7, 
                                        blur=0.0, downsample=0.0).numpy()[0]
                
            im_train = np.stack((imr[:,:Ly//2,:Lx//2], imr[:,:Ly//2,Lx//2:2*(Lx//2)],
                                imr[:,Ly//2:2*(Ly//2),:Lx//2]), axis=0)
            # divide image into 4 parts for training and validation
            im_train_all.extend(list(im_train.squeeze()))
            gt_train_all.extend(list(gt_train.squeeze()))
            
            im_val = imr[:, Ly//2:, Lx//2:]
            im_val_all.append(im_val.squeeze())
            gt_val_all.append(gt_val.squeeze())

    os.makedirs(Path(root / "noisy_test" / "care" / "GT"), exist_ok=True)
    os.makedirs(Path(root / "noisy_test" / "care" / "source"), exist_ok=True)
    [io.imsave(Path(root / "noisy_test" / "care" / "GT" / f"{i:03d}.tif"), im) for i, im in enumerate(gt_train_all)];
    [io.imsave(Path(root / "noisy_test" / "care" / "source" / f"{i:03d}.tif"), im) for i, im in enumerate(im_train_all)];
    n_train = len(im_train_all)
    n_val = len(im_val_all)
    print(n_train, n_val)
    [io.imsave(Path(root / "noisy_test" / "care" / "GT" / f"{i+n_train:03d}.tif"), im) for i, im in enumerate(gt_val_all)];
    [io.imsave(Path(root / "noisy_test" / "care" / "source" / f"{i+n_train:03d}.tif"), im) for i, im in enumerate(im_val_all)];

def train_test_specialist(root, lr=0.001, n_epochs=100, test=True):
    n_train = 3 * 89 * n_samples
    n_val = 1 * 89 * n_samples

    raw_data = RawData.from_folder (
        basepath    = Path(root / "noisy_test" / "care"),
        source_dirs = ["source"],
        target_dir  = "GT",
        axes        = "YX",
    )

    X, Y, XY_axes = create_patches (
        raw_data            = raw_data,
        patch_size          = (128, 128),
        patch_filter        = no_background_patches(0),
        n_patches_per_image = 2,
        normalization       = None,
        save_file           = Path(root / "noisy_test" / "care" / "training_data.npz"),
    )

    val_frac = n_val*2 / (n_train*2 + n_val*2)
    print(val_frac)
    (X,Y), (X_val,Y_val), axes = load_training_data(Path(root / "noisy_test" / "care" / "training_data.npz"), validation_split=val_frac, verbose=True)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, 
                        train_batch_size=8, train_steps_per_epoch=400,
                        train_learning_rate=lr, train_epochs=n_epochs)
    print(config)
    vars(config)

    model = CARE(config, f'CIL_lr{lr:0.5f}_ne{n_epochs}', basedir=Path(root / "noisy_test" / "care" / "models"))
    history = model.train(X,Y, validation_data=(X_val,Y_val))

    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
    plt.show()

    val_min = np.array(history.history["val_loss"]).min()
    print(val_min)

    if not test:
        return val_min
    else:
        dat = np.load(root / "noisy_test" / "test_poisson.npy", allow_pickle=True).item()
        test_noisy = dat["test_noisy"][:11]
        masks_true = dat["masks_true"][:11]
        diam_test = dat["diam_test"]

        restored = [model.predict(test_noisy[i][...,np.newaxis], axes).squeeze() for i in range(len(test_noisy))]

        seg_model = models.CellposeModel(gpu=True, model_type="cyto2_cp3")
        masks2 = seg_model.eval(restored, channels=[0,0], diameter=diam_test, normalize=True)[0]

        dat[f"test_care"] = restored
        dat[f"masks_care"] = masks2

        np.save(root / "noisy_test" / f"test_poisson_care_specialist.npy", dat)

        thresholds = np.arange(0.5,1.0,0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks2, threshold=thresholds)
        print(ap.mean(axis=0))

        return restored, masks2, ap