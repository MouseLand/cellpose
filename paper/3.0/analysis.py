"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from cellpose import io, transforms, models, metrics, denoise, dynamics, utils
from natsort import natsorted
from pathlib import Path
import torch
from torch import nn
from tqdm import trange

# in same folder
try:
    import noise2self
    import noise2void
    import care
except Exception as e:
    print(e)
    print("noise2void / noise2self / care not installed, cannot run those comparisons")

io.logger_setup()

device = torch.device("cuda")

try:
    from cellpose.segformer import Transformer
except Exception as e:
    print(e)
    print("need to install segmentation_models_pytorch to run transformer")

model_names = {"poisson": "denoise", "blur": "deblur", "downsample": "upsample"}

def seg_eval_cp3(folder, noise_type="poisson"):
    """ need to download test_poisson.npy, test_blur.npy, test_downsample.npy
    (for cells and/or nuclei)
    """
    ctypes = ["cyto2", "nuclei"]
    for c, ctype in enumerate(ctypes):
        print(ctype)
        pretrained_models = [f"/home/carsen/.cellpose/models/{model_names[noise_type]}{istr}_{ctype}" 
                                 for istr in ["_rec", "_seg", "_per", ""]]
        pretrained_models.extend([f"/home/carsen/.cellpose/models/{model_names[noise_type]}_cyto3", 
                                  f"/home/carsen/.cellpose/models/oneclick_{ctype}", 
                                  f"/home/carsen/.cellpose/models/oneclick_cyto3"])

        seg_model = models.CellposeModel(gpu=True, model_type=ctype)

        folder_name = ctype
        root = Path(folder) / f"images_{folder_name}/"
        model_name = model_names[noise_type]
        nimg_test = 68 if ctype=="cyto2" else 111
        diam_mean = 30. if ctype == "cyto2" else 17.
        ### cellpose enhance
        dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                        allow_pickle=True).item()
        test_noisy = dat["test_noisy"][:nimg_test]
        masks_true = dat["masks_true"][:nimg_test]
        diam_test = dat["diam_test"][:nimg_test] if "diam_test" in dat else diam_mean * np.ones(
            len(test_noisy))

        thresholds = np.arange(0.5, 1.05, 0.05)
        istrs = ["rec", "seg", "per", "perseg", "noise_spec", "data_spec", "gen"]
        
        print(pretrained_models)
        aps = []
        for istr, pretrained_model in zip(istrs, pretrained_models):
            dn_model = denoise.DenoiseModel(gpu=True, nchan=1, 
                                            diam_mean = 30 if "cyto" in pretrained_model else 17,
                                            pretrained_model=pretrained_model)
            dn_model.pretrained_model = "test"
            imgs2 = dn_model.eval([test_noisy[i][0] for i in range(len(test_noisy))],
                                diameter=diam_test, channel_axis=0)

            masks2, flows, styles2 = seg_model.eval(imgs2, channels=[1, 0],
                                                    diameter=diam_test, channel_axis=-1,
                                                    normalize=True)

            ap, tp, fp, fn = metrics.average_precision(masks_true, masks2, threshold=thresholds)
            print(f"{noise_type} {istr} AP@0.5 \t = {ap[:,0].mean(axis=0):.3f}")

            dat[f"test_{istr}"] = imgs2
            dat[f"masks_{istr}"] = masks2
            dat[f"flows_{istr}"] = flows
            aps.append(ap)
        np.save(root / "noisy_test" / f"test_{noise_type}_cp3_all.npy", dat)

        if noise_type == "poisson":
            ### cellpose retrained
            dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                          allow_pickle=True).item()
            test_noisy = dat["test_noisy"]
            masks_true = dat["masks_true"]
            diam_test = dat["diam_test"] if "diam_test" in dat else 30. * np.ones(
                len(test_noisy))

            seg_model = models.CellposeModel(gpu=True, nchan=1,
                                             model_type=f"{ctype}_noisy")
            masks2 = seg_model.eval(test_noisy, channels=None, diameter=diam_test,
                                    normalize=False)[0]

            ap, tp, fp, fn = metrics.average_precision(masks_true, masks2)

            dat[f"masks_retrain"] = masks2

            np.save(root / "noisy_test" / f"test_{noise_type}_cp_retrain.npy", dat)


def blind_denoising(folder):
    ctypes = ["cyto2", "nuclei"]
    for ctype in ctypes:
        root = Path(folder) / f"images_{ctype}/"

        ### noise2void
        imgs_n2v, masks_n2v = noise2void.train_per_image_synthetic(
            root, ctype=ctype, plot=0, save=False)
        dat = np.load(root / "noisy_test" / f"test_poisson.npy",
                      allow_pickle=True).item()
        masks_true = dat["masks_true"]
        print(len(masks_n2v), len(masks_true))
        ap = metrics.average_precision(masks_true, masks_n2v)[0]
        print(ap.mean(axis=0))

        ### noise2self
        imgs_n2s, masks_n2s = noise2self.train_per_image_synthetic(
            root, ctype=ctype, plot=0, save=True)
        dat = np.load(root / "noisy_test" / f"test_poisson.npy",
                      allow_pickle=True).item()
        masks_true = dat["masks_true"]
        print(len(masks_n2s), len(masks_true))
        ap = metrics.average_precision(masks_true, masks_n2s)[0]
        print(ap.mean(axis=0))


def real_examples(folder):
    """ download test_data from CARE and put in folder """

    cols = ["r", "g", "b"]

    dsets = ["Projection_Flywing", "Denoising_Tribolium"]

    thresholds = np.arange(0.5, 1.05, 0.05)
    flow_threshold = 0.4

    dat = {}
    for dset in dsets:
        root = Path(folder) / f"{dset}/test_data/"
        print(root)
        if dset == "Projection_Flywing":
            model_type = "denoise_cyto2"
            diam_mean = 30.
            cp_model_type = "cyto2"
        else:
            model_type = "denoise_nuclei"
            diam_mean = 17.
            cp_model_type = "nuclei"

        if dset == "Projection_Flywing":
            clean = [
                transforms.normalize99(io.imread(tif))
                for tif in natsorted(root.glob("proj_C2*.tif"))
            ]
            diam = 20.
            cellprob_threshold = -2.
        else:
            imgs = [io.imread(tif) for tif in natsorted(root.glob("*.tif"))]
            dz = 8
            clean = [
                transforms.normalize99(img[-1][img.shape[1] // 2 -
                                               dz:img.shape[1] // 2 + dz].max(axis=0))
                for img in imgs
            ]
            diam = 12.
            cellprob_threshold = 0.

        print(diam_mean)
        model = denoise.DenoiseModel(gpu=True, nchan=1, diam_mean=diam_mean,
                                     model_type=model_type)
        seg_model = models.CellposeModel(gpu=True, model_type=cp_model_type,
                                         diam_mean=diam_mean)
        masks_clean = seg_model.eval(clean, diameter=diam,
                                     cellprob_threshold=cellprob_threshold,
                                     flow_threshold=flow_threshold)[0]
        dat["clean"] = clean
        dat["masks_clean"] = masks_clean
        dat["noisy"] = []
        dat["ap_noisy"] = []
        dat["masks_noisy"] = []
        dat["denoised"] = []
        dat["masks_denoised"] = []
        dat["ap_denoised"] = []

        plt.figure(figsize=(4, 4))
        for nl in range(3):
            if dset == "Projection_Flywing":
                nstr = f"proj_C{nl+(nl>1)}"
                print(nstr)
                noisy = [
                    transforms.normalize99(io.imread(tif))
                    for tif in natsorted(root.glob(f"{nstr}*.tif"))
                ]
            elif dset == "Denoising_Tribolium":
                noisy = [transforms.normalize99(img[nl][img.shape[1] // 2 - dz:img.shape[1] // 2 + dz].max(axis=0)) for img in imgs]  #.max(axis=0)
            else:
                noisy = [
                    io.imread(tif)
                    for tif in natsorted((root / f"condition_{nl+1}").glob("*.tif"))
                ]
                noisy = [
                    transforms.normalize99(img[img.shape[0] // 4:img.shape[0] // 4 +
                                               dz].max(axis=0)) for img in noisy
                ]

            denoised = model.eval(noisy, channels=None, diameter=diam)
            masks_denoised = seg_model.eval(
                denoised,
                diameter=diam,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
            )[0]
            masks_noisy = seg_model.eval(noisy, diameter=diam,
                                         cellprob_threshold=cellprob_threshold,
                                         flow_threshold=flow_threshold)[0]

            thresholds = np.arange(0.5, 1.05, 0.05)
            ap_noisy = metrics.average_precision(masks_clean, masks_noisy,
                                                 threshold=thresholds)[0]
            ap_denoised = metrics.average_precision(masks_clean, masks_denoised,
                                                    threshold=thresholds)[0]

            plt.plot(thresholds, ap_denoised.mean(axis=0), ls="-", color=cols[nl])
            plt.plot(thresholds, ap_noisy.mean(axis=0), ls="--", color=cols[nl])

            dat["noisy"].append(noisy)
            dat["masks_noisy"].append(masks_noisy)
            dat["ap_noisy"].append(ap_noisy)

            dat["denoised"].append(denoised)
            dat["masks_denoised"].append(masks_denoised)
            dat["ap_denoised"].append(ap_denoised)

        plt.show()
        np.save(root / "cp_masks.npy", dat)

        noisy = []
        for nl in range(3):
            noisy.extend(dat["noisy"][nl])
        print(len(noisy))

        imgs_n2s, masks_n2s = [], []
        for i in range(len(noisy)):
            pred_val = noise2self.train_per_image(noisy[i][np.newaxis,
                                                           ...].astype("float32"))
            masks = seg_model.eval(pred_val, diameter=diam,
                                   cellprob_threshold=cellprob_threshold,
                                   channels=[1, 0], channel_axis=0, normalize=True)[0]
            imgs_n2s.append(pred_val)
            masks_n2s.append(masks)

            print(f">>> IMAGE {i}, n_masks = {masks.max()}")

        ap_n2s = [
            metrics.average_precision(
                masks_clean,
                masks_n2s[nl * len(masks_clean):(nl + 1) * len(masks_clean)],
                threshold=thresholds)[0] for nl in range(3)
        ]
        dat2 = {}
        dat2["denoised_n2s"] = imgs_n2s
        dat2["masks_n2s"] = masks_n2s
        dat2["noisy"] = noisy
        dat2["masks_clean"] = masks_clean
        dat2["ap_n2s"] = ap_n2s
        np.save(root / "n2s_masks.npy", dat2)

        imgs_n2v, masks_n2v = [], []
        for i in range(len(noisy)):
            pred_val = noise2void.train_per_image(noisy[i])
            masks = seg_model.eval(pred_val, diameter=diam,
                                   cellprob_threshold=cellprob_threshold,
                                   channels=[1, 0], channel_axis=0, normalize=True)[0]
            imgs_n2v.append(pred_val)
            masks_n2v.append(masks)

            print(f">>> IMAGE {i}, n_masks = {masks.max()}")

        ap_n2v = [
            metrics.average_precision(
                masks_clean,
                masks_n2v[nl * len(masks_clean):(nl + 1) * len(masks_clean)],
                threshold=thresholds)[0] for nl in range(3)
        ]
        dat2 = {}
        dat2["denoised_n2v"] = imgs_n2v
        dat2["masks_n2v"] = masks_n2v
        dat2["noisy"] = noisy
        dat2["masks_clean"] = masks_clean
        dat2["ap_n2v"] = ap_n2v
        np.save(root / "n2v_masks.npy", dat2)

def real_examples_ribo(root):
    navgs = [1, 2, 4, 8, 16, 32, 64]
    noisy = [[], [], [], [], [], [], []]
    clean = []
    for i in [1, 3, 6, 4, 5]:
        imgs = io.imread(Path(root) / f"denoise_{i:05d}_00001.tif")[:300]
        imgs = [imgs[:, :512, :512], imgs[:, 512:, :512], imgs[:, :512, 512:], imgs[:, 512:, 512:]]
        clean.extend([img.mean(axis=0) for img in imgs])
        for n, navg in enumerate(navgs):
            iavg = np.linspace(0, len(imgs[0])-1, navg+2).astype(int)[1:-1]
            noisy[n].extend(np.array([img[iavg].mean(axis=0) for img in imgs]))
        print(len(clean), len(noisy[0]))

    thresholds = np.arange(0.5, 1.05, 0.05)
    diameter = 17
    normalize = True # {"tile_norm_blocksize": 80}
    seg_model = models.Cellpose(gpu=True, model_type="cyto2")
    model = denoise.DenoiseModel(gpu=True, model_type="denoise_cyto2")
    masks = seg_model.eval(clean, diameter=diameter, channels=[0,0], 
                        normalize=normalize)[0]
    ap_noisy = np.zeros((len(noisy), len(noisy[0]), len(thresholds)))
    ap_dn = np.zeros((len(noisy), len(noisy[0]), len(thresholds)))
    dat = {}
    dat["navgs"] = navgs
    dat["imgs_dn"] = []
    dat["masks_dn"] = []
    dat["masks_noisy"] = []
    dat["masks_clean"] = masks
    dat["noisy"] = noisy
    dat["clean"] = clean
    for n, imgs in enumerate(noisy):
        masks_noisy = seg_model.eval(imgs, diameter=diameter, channels=[0,0],
                                    normalize=normalize)[0]
        img_dn = model.eval(imgs, diameter=diameter, channels=[0,0],
                            normalize=normalize)
        ap, tp, fp, fn = metrics.average_precision(masks, masks_noisy, threshold=thresholds)
        ap_noisy[n] = ap
        masks_dn = seg_model.eval(img_dn, diameter=diameter, channels=[0,0],
                                normalize=normalize)[0]
        ap, tp, fp, fn = metrics.average_precision(masks, masks_dn, threshold=thresholds)
        ap_dn[n] = ap
        dat["imgs_dn"].append(img_dn)
        dat["masks_dn"].append(masks_dn)
        dat["masks_noisy"].append(masks_noisy)
        print(ap_noisy[n,:,0].mean(axis=0), ap_dn[n,:,0].mean(axis=0))
    dat["ap_noisy"] = ap_noisy
    dat["ap_dn"] = ap_dn
    np.save(Path(root) / "ribo_denoise.npy", dat)

    dat = {}
    dat["navgs"] = navgs
    dat["imgs_n2s"] = []
    dat["masks_n2s"] = []
    dat["masks_clean"] = masks
    dat["noisy"] = noisy
    dat["clean"] = clean
    dat["ap_n2s"] = np.zeros((len(noisy), len(noisy[0]), len(thresholds)))
    
    for n, imgs in enumerate(noisy):
        imgs_n2s = []
        for i in trange(len(imgs)):
            out = noise2self.train_per_image(imgs[i][np.newaxis,...].astype("float32"))
            imgs_n2s.append(out)
        imgs_n2s = np.array(imgs_n2s)
        masks_n2s = seg_model.eval(imgs_n2s, diameter=diameter, channels=[0,0])[0]
        ap, tp, fp, fn = metrics.average_precision(masks, masks_n2s, threshold=thresholds)
        dat["ap_n2s"][n] = ap
        dat["imgs_n2s"].append(imgs_n2s)
        dat["masks_n2s"].append(masks_n2s)
        print(n, ap.mean(axis=0)[[0, 5, 8]])

    np.save(Path(root) / "ribo_denoise_n2s.npy", dat)

    dat = {}
    dat["navgs"] = navgs
    dat["imgs_n2v"] = []
    dat["masks_n2v"] = []
    dat["masks_clean"] = masks
    dat["noisy"] = noisy
    dat["clean"] = clean
    dat["ap_n2v"] = np.zeros((len(noisy), len(noisy[0]), len(thresholds)))

    for n, imgs in enumerate(noisy):
        imgs_n2v = []
        for i in trange(len(imgs)):
            out = noise2void.train_per_image(imgs[i].astype("float32"))
            imgs_n2v.append(out)
        imgs_n2v = np.array(imgs_n2v)
        masks_n2v = seg_model.eval(imgs_n2v, diameter=diameter, channels=[0,0],
                                normalize=normalize)[0]
        ap, tp, fp, fn = metrics.average_precision(masks, masks_n2v, threshold=thresholds)
        #print(ap[:,0])
        dat["ap_n2v"][n] = ap
        dat["imgs_n2v"].append(imgs_n2v)
        dat["masks_n2v"].append(masks_n2v)
        print(n, ap.mean(axis=0)[[0, 5, 8]])

    np.save(Path(root) / "ribo_denoise_n2v.npy", dat)



def specialist_training(root):
    """ root is path to specialist images (first 89 images of cyto2 and first 11 test images) """

    # make patches for training CARE, noise2self and noise2void
    care.CIL_dataset(root)

    # train/test CARE
    care.train_test_specialist(root)

    # train/test noise2self
    imgs, masks_n2s, ap = noise2self.train_test_specialist(root, n_epochs=50, lr=5e-4,
                                                           test=True)

    # train/test noise2void
    noise2void.train_test_specialist(root, n_epochs=100, lr=4e-4, test=True)


def cyto3_comparisons(folder):
    """  diameters computed from generalist model cyto3
    will need segmentation_models_pytorch to run transformer """
    root = Path(folder)
    folders = [
        "cyto2", "nuclei", "tissuenet", "livecell", "yeast_BF", "yeast_PhC",
        "bact_phase", "bact_fluor", "deepbacs"
    ]

    net_types = ["generalist", "specialist", "transformer"]
    for net_type in net_types[-1:]:
        if net_type == "generalist":
            seg_model = models.Cellpose(gpu=True, model_type="cyto3")
        elif net_type == "transformer":
            pretrained_model = "/home/carsen/.cellpose/models/transformer_cp3"
            seg_model = models.CellposeModel(gpu=True, backbone="transformer",
                                             pretrained_model=pretrained_model)
        for f in folders:
            if net_type == "specialist":
                seg_model = models.CellposeModel(gpu=True, model_type=f"{f}_cp3")

            root = Path(folder) / f"images_{f}"
            channels = [1, 2] if f == "tissuenet" or f == "cyto2" else [1, 0]
            tifs = natsorted((root / "test").glob("*.tif"))
            tifs = [tif for tif in tifs]
            tifs = [
                tif for tif in tifs
                if "_masks" not in str(tif) and "_flows" not in str(tif)
            ]
            if net_type != "generalist":
                d = np.load(
                    Path(folder) / f"{f}_generalist_masks.npy",
                    allow_pickle=True).item()
                diams = d["diams"]
            else:
                diams = 0

            imgs = [io.imread(tif) for tif in tifs]
            flows = [io.imread(str(tif)[:-4] + "_flows.tif") for tif in tifs]
            masks = [flow[0].astype("uint16") for flow in flows]

            if f == "cyto2":
                imgs = imgs[:68]
                masks = masks[:68]

            dat = {}
            dat["imgs"] = imgs
            dat["masks"] = masks
            dat["files"] = tifs
            out = seg_model.eval(imgs, diameter=diams, channels=channels,
                                 tile_overlap=0.5, flow_threshold=0.4, augment=True,
                                 bsize=224, niter=2000 if f == "bact_phase" else None)
            if len(out) == 3:
                masks_pred, flows_pred, styles = out
            else:
                masks_pred, flows_pred, styles, diams = out
            ap, tp, fp, fn = metrics.average_precision(
                masks, masks_pred, threshold=np.arange(0.5, 1.05, 0.05))
            print(ap.mean(axis=0))
            dat["masks_pred"] = masks_pred
            dat["performance"] = [ap, tp, fp, fn]
            dat["diams"] = diams

            #np.save(f"/media/carsen/ssd4/datasets_cellpose/{f}_{net_type}_masks.npy", dat)


if __name__ == '__main__':
    # folder with folders images_cyto2 and images_nuclei for those datasets
    folder = "/media/carsen/ssd4/datasets_cellpose/"

    # https://publications.mpi-cbg.de/publications-sites/7207/
    folder2 = "/media/carsen/ssd4/denoising/"  # download CARE datasets here

    ### run cyto2/nuclei denoising analyses
    seg_eval_cp3(folder, noise_type="poisson")

    ### train cyto2/nuclei per image models with noise2void and noise2self
    blind_denoising(folder)

    ### real data: run cyto2/nuclei models + train cyto2/nuclei per image models with noise2void and noise2self
    real_examples(folder2)

    ### specialist denoising -- train CARE, noise2void + noise2self
    specialist_training(Path(folder) / "images_cyto2")

    ### run cyto2/nuclei deblurring analyses
    seg_eval_cp3(folder, noise_type="blur")

    ### run cyto2/nuclei upsampling analyses
    seg_eval_cp3(folder, noise_type="downsample")

    ### oneclick cyto2/nuclei performance
    seg_eval_oneclick(folder)

    ### run supergeneralist segmentation models analyses
    cyto3_comparisons(folder)
