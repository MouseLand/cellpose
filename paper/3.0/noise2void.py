"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models, io, metrics
from cellpose.models import CellposeModel
from tqdm import trange
# uses tensorflow
# pip install n2v
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import os

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def train_per_image(img_noisy):
    img = img_noisy.copy()
    print(img.shape)
    Ly, Lx = img.shape[-2:]
    patch_shape = (min(64,
                       min(Ly // 2, Lx // 2) - 4), min(64,
                                                       min(Ly // 2, Lx // 2) - 4))
    if patch_shape[0] < 64:
        patch_shape = (60, 60)

    # We create our DataGenerator-object.
    # It will help us load data and extract patches for training and validation.
    datagen = N2V_DataGenerator()

    # divide image into 4 parts for training and validation
    im_train = np.stack(
        (img[:Ly // 2, :Lx // 2], img[:Ly // 2, Lx // 2:2 *
                                      (Lx // 2)], img[Ly // 2:2 * (Ly // 2), :Lx // 2]),
        axis=0)
    im_train = im_train[:, np.newaxis, :, :, np.newaxis]
    X = datagen.generate_patches_from_list(list(im_train), shape=patch_shape)

    im_val = img[Ly // 2:, Lx // 2:]
    im_val = im_val[np.newaxis, ..., np.newaxis]
    X_val = datagen.generate_patches_from_list([im_val], shape=patch_shape)

    print(len(X), len(X_val))
    # train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
    # is shown once per epoch.
    train_batch_size = min(128, len(X))
    config = N2VConfig(X, unet_kern_size=3,
                       train_steps_per_epoch=X.shape[0] // train_batch_size,
                       train_epochs=100, train_loss='mse', batch_norm=True,
                       train_batch_size=train_batch_size, n2v_perc_pix=0.198,
                       n2v_patch_shape=patch_shape, n2v_manipulator='uniform_withCP',
                       n2v_neighborhood_radius=5)

    # a name used to identify the model
    model_name = 'n2v_2D'
    # the base directory in which our model will live
    basedir = 'models'
    # We are now creating our network model.
    model = N2V(config, model_name, basedir=basedir)

    # We are ready to start training now.
    history = model.train(X, X_val, verbose=False)

    input_val = img_noisy.copy().squeeze()

    pred_val = model.predict(input_val, axes="YX")

    return pred_val


def train_per_image_synthetic(root, ctype="cyto2", plot=False, save=True):
    noise_type = "poisson"

    dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                  allow_pickle=True).item()
    test_noisy = dat["test_noisy"]
    test_labels = dat["masks_true"]
    diam_test = dat["diam_test"] if "diam_test" in dat else 30. * np.ones(
        len(test_noisy))

    imgs_n2v, masks_n2v = [], []

    seg_model = CellposeModel(gpu=True, model_type=f"{ctype}")
    for i in trange(len(test_noisy)):
        out = train_per_image(test_noisy[i].squeeze())

        masks = seg_model.eval(out, diameter=diam_test[i], channels=[1, 0],
                               channel_axis=0, normalize=True)[0]

        masks_n2v.append(masks)
        imgs_n2v.append(out)

        print(f">>> IMAGE {i}, n_masks = {masks.max()}")
        if plot:
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1)
            plt.imshow(test_noisy[i][0])
            plt.subplot(1, 4, 2)
            plt.imshow(out)
            plt.subplot(1, 4, 3)
            plt.imshow(masks)
            plt.subplot(1, 4, 4)
            plt.imshow(test_labels[i])
            plt.show()

    dat["masks_n2v"] = masks_n2v
    dat["test_n2v"] = imgs_n2v
    if save:
        np.save(root / "noisy_test" / f"test_{noise_type}_n2v.npy", dat)

    return imgs_n2v, masks_n2v


def train_test_specialist(root, n_epochs=100, lr=5e-4, test=True):
    n_train = 3 * 89 * 20
    n_val = 89 * 20

    dat = np.load(root / "noisy_test" / "test_poisson.npy", allow_pickle=True).item()
    test_noisy = dat["test_noisy"][:11]
    masks_true = dat["masks_true"][:11]
    diam_test = dat["diam_test"]

    im_train = [
        io.imread(Path(root / "noisy_test" / "care" / "source" /
                       f"{i:03d}.tif"))[np.newaxis, :, :, np.newaxis]
        for i in range(n_train)
    ]
    im_train.extend([tn[0][np.newaxis, :, :, np.newaxis] for tn in test_noisy])
    im_val = [
        io.imread(Path(root / "noisy_test" / "care" / "source" /
                       f"{i:03d}.tif"))[np.newaxis, :, :, np.newaxis]
        for i in range(n_train, n_train + n_val)
    ]

    patch_shape = (64, 64)
    datagen = N2V_DataGenerator()
    X = datagen.generate_patches_from_list(im_train, shape=patch_shape)
    X_val = datagen.generate_patches_from_list(im_val, shape=patch_shape)
    print(len(X), len(X_val))

    # train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
    # is shown once per epoch.
    train_batch_size = min(128, len(X))
    config = N2VConfig(
        X,
        unet_kern_size=3,
        train_steps_per_epoch=25,  # same as CARE
        validation_steps=5,
        train_epochs=n_epochs,
        train_learning_rate=lr,
        train_loss='mse',
        batch_norm=True,
        train_batch_size=train_batch_size,
        n2v_perc_pix=0.198,
        n2v_patch_shape=patch_shape,
        n2v_manipulator='uniform_withCP',
        n2v_neighborhood_radius=5)

    # a name used to identify the model
    model_name = 'n2v_2D_specialist'
    # the base directory in which our model will live
    basedir = 'models'
    # We are now creating our network model.
    model = N2V(config, model_name, basedir=basedir)

    # We are ready to start training now.
    history = model.train(X, X_val, verbose=True)

    val_min = np.array(history.history["val_loss"]).min()
    print(val_min)

    if not test:
        return val_min
    else:

        imgs_n2v, masks_n2v = [], []
        for i in range(len(test_noisy)):
            input_val = test_noisy[i].squeeze()
            pred_val = model.predict(input_val, axes="YX")
            seg_model = models.CellposeModel(gpu=True, model_type="cyto2")
            masks = seg_model.eval(pred_val, diameter=diam_test[i], channels=[1, 0],
                                   channel_axis=0, normalize=True)[0]
            imgs_n2v.append(pred_val)
            masks_n2v.append(masks)

        dat[f"test_n2v"] = imgs_n2v
        dat[f"masks_n2v"] = masks_n2v

        np.save(root / "noisy_test" / f"test_poisson_n2v_specialist.npy", dat)

        thresholds = np.arange(0.5, 1.0, 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_n2v,
                                                   threshold=thresholds)
        print(ap.mean(axis=0))

        return imgs_n2v, masks_n2v, ap
