import os
import numpy as np
from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch, denoise
from natsort import natsorted 
from pathlib import Path
from glob import glob

from cellpose.io import logger_setup

def prediction_test_hidden(root):
    """ root is path to Hidden folder """
    root = Path(root)

    logger_setup()
    # path to images
    fall = natsorted(glob((root / "images" / "*").as_posix()))
    img_files = [f for f in fall if "_masks" not in f and "_flows" not in f]

    # load images
    imgs = [io.imread(f) for f in img_files]
    nimg = len(imgs)

    # for 3 channel model, normalize images and convert to 3 channels if needed
    imgs_norm = []
    for img in imgs:
        if img.ndim==2:
            img = np.tile(img[:,:,np.newaxis], (1,1,3))
        img = transforms.normalize_img(img, axis=-1)
        imgs_norm.append(img.transpose(2,0,1))

    dat = {}
    for mtype in ["default", "transformer"]:
        if mtype=="default":
            model = models.Cellpose(gpu=True, nchan=3, model_type="neurips_cellpose_default")
            channels = None
            normalize = False
            diams = None # Cellpose will estimate diameter
        elif mtype=="transformer":
            model = models.CellposeModel(gpu=True, nchan=3, model_type="neurips_cellpose_transformer", backbone="transformer")
            channels = None 
            normalize = False
            diams = dat["diams_pred"] # (use diams from Cellpose default model for transformer)

        out = model.eval(imgs_norm, diameter=diams,
                        channels=channels, normalize=normalize, 
                        tile_overlap=0.6, augment=True)
        # predicted masks
        dat[mtype] = out[0]
        
        if mtype=="default":
            diams = out[-1]
            dat["diams_pred"] = diams
            dat[f"{mtype}_size_timing"] = model.sz.timing
            dat[f"{mtype}_mask_timing"] = model.cp.timing
        else:
            dat[f"{mtype}_mask_timing"] = model.timing
        
    np.savez_compressed("neurips_test_results.npz", dat)

def prediction_tuning(root, root2=None):
    """ root is path to Tuning folder, root2 is path to mediar results """
    root = Path(root)
    logger_setup()

    # path to images and masks
    fall = natsorted(glob((root / "images" / "*").as_posix()))
    # (exclude last image)
    img_files = [f for f in fall if "_masks" not in f and "_flows" not in f][:-1]
    mask_files = natsorted(glob((root / "labels" / "*").as_posix()))[:-1]

    # load images and masks
    imgs = [io.imread(f) for f in img_files]
    masks = [io.imread(f) for f in mask_files]
    nimg = len(imgs)

    # for 3 channel model, normalize images and convert to 3 channels if needed
    imgs_norm = []
    for img in imgs:
        if img.ndim==2:
            img = np.tile(img[:,:,np.newaxis], (1,1,3))
        img = transforms.normalize_img(img, axis=-1)
        imgs_norm.append(img.transpose(2,0,1))
    
    dat = {}

    ### RUN MODELS
    model_types = ["grayscale", "default", "transformer", "maetal",  "mediar"]
    for mtype in model_types[:-1]:
        print(mtype)
        if mtype=="grayscale"  or mtype=="maetal":
            if mtype=="grayscale":
                model = models.CellposeModel(gpu=True, model_type="neurips_grayscale_cyto2")
            else:
                ### need to download cellpose model from Ma et al
                # https://github.com/JunMa11/NeurIPS-CellSeg/tree/main/cellpose-omnipose-KIT-GE
                pretrained_model = "/home/carsen/Downloads/model.501776_epoch_499"
                if not os.path.exists(pretrained_model):
                    print("need to download cellpose model from Ma et al; https://github.com/JunMa11/NeurIPS-CellSeg/tree/main/cellpose-omnipose-KIT-GE")
                    print("skipping Ma et al model")
                    del model_types[-2]
                    break
                model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)
            channels = [0, 0]
            normalize = True
            diams = None # CellposeModel will use mean diameter from training set
        elif mtype=="default":
            model = models.Cellpose(gpu=True, nchan=3, model_type="neurips_cellpose_default")
            channels = None
            normalize = False
            diams = None # Cellpose will estimate diameter
        elif mtype=="transformer":
            model = models.CellposeModel(gpu=True, nchan=3, model_type="neurips_cellpose_transformer", backbone="transformer")
            channels = None 
            normalize = False
            diams = dat["diams_pred"] # (use diams from Cellpose default model for transformer)
        
        out = model.eval(imgs if mtype=="grayscale" else imgs_norm, diameter=diams,
                        channels=channels, normalize=normalize, 
                        tile_overlap=0.6, augment=True)
        if mtype=="default":
            diams = out[-1]
            dat["diams_pred"] = diams

        dat[mtype] = out[0]

    ### load Mediar results
    if root2 is not None:
        root2 = Path(root2)
        masks_pred_mediar = []
        for imgf in img_files:
            maskf = root2 / (os.path.splitext(os.path.split(imgf)[-1])[0] + "_label.tiff")
            m = io.imread(maskf)
            m = np.unique(m, return_inverse=True)[1].reshape(m.shape)
            masks_pred_mediar.append(m)

        dat["mediar"] = masks_pred_mediar
    else:
        print("no path to mediar files specified")
        print("skipping mediar")
        del model_types[-1]

    ### EVALUATION
    thresholds = np.arange(0.5, 1.05, 0.05)
    dat["thresholds"] = thresholds
    masks_true = [lbl.astype("uint32") for lbl in masks]
    for mtype in model_types:
        print(mtype)
        masks_pred = dat[mtype]
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=thresholds)
        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"{mtype}, F1 score @ 0.5 = {np.median(f1[:,0]):.3f}")
        
        dat[mtype+"_f1"] = f1
        dat[mtype+"_tp"] = tp
        dat[mtype+"_fp"] = fp
        dat[mtype+"_fn"] = fn

    np.savez_compressed("neurips_eval_results.npz", dat)

    return imgs_norm, masks, dat