import cv2 
import numpy as np
from cellpose import transforms, io, models, metrics, utils, denoise, vit
import torch
from pathlib import Path
from natsort import natsorted
import time
from tqdm import trange
import fastremap

dsets_all = ["deepbacs", "bact_phase", "bact_fluor", "tissuenet", "cyto2", 
        "monuseg", "livecell", "ovules", "root", "blastospim"]

dsets_others = ["flywing", "tribolium", "ribo"]

nstr = {"poisson": "denoise", "blur": "deblur", "downsample": "upsample", "aniso": "aniso"}

def load_dataset(dset, root=None):
    if root is None:
        root = Path(__file__).parent / "../"

    if dset in dsets_others:
        dat = np.load(root / f"{dset}/cp_masks.npy", allow_pickle=True).item()
        clean = dat["clean"]
        imgs = [dat["noisy"][i][j] for i in range(len(dat["noisy"])) for j in range(len(dat["noisy"][i]))]
        masks_true = [dat["masks_clean"][j] for i in range(len(dat["noisy"])) for j in range(len(dat["noisy"][i]))]
        return None, imgs, masks_true
    else:
        if dset == "monuseg":
            folder_name = "images_HandE/MoNuSeg/MoNuSegTestData"
        elif dset == "root" or dset == "ovules":
            folder_name = f"root_ovules_wolny/{dset}/"
        elif dset == "blastospim":
            folder_name = dset
        else:
            if dset[:5] == "cyto2":
                folder_name = f"images_cyto2"
            else:
                folder_name = f"images_{dset}"
        if dset == "monuseg":
            files = [f for f in Path(root / f"{folder_name}/").glob("*.tif")]
        elif dset == "ovules" or dset == "root" or dset == "blastospim":
            files = [f for f in Path(root / f"{folder_name}/train/").glob("*.tif")]
            files.extend([f for f in Path(root / f"{folder_name}/test/").glob("*.tif")])
        else:
            files = [f for f in Path(root / f"{folder_name}/test/").glob("*.tif")]
        
        files = natsorted([f for f in files if "_masks" not in str(f) and "_flows" not in str(f)])

        if dset == "ovules" or dset == "root" or dset == "blastospim" or dset == "monuseg":
            masks_true = [io.imread(str(f).replace(".tif", "_masks.tif")).astype("uint16") for f in files]
        else:
            masks_true = [io.imread(str(f).replace(".tif", "_flows.tif"))[0].astype("uint16") for f in files]
            
        imgs = [io.imread(f) for f in files]


        if dset[:5]=="cyto2":
            ind_im = np.array([68, 69, 71, 72, 73, 74, 75, 76, 84, 86, 89, 90])
            ind_im = np.hstack((np.arange(55), ind_im))
            imgs = [imgs[i] for i in ind_im]
            masks_true = [masks_true[i] for i in ind_im]

            diam_true = np.array([utils.diameters(m)[0] for m in masks_true])

            dtype = dset[6:]
            if len(dtype) > 0:
                if dtype[:2] == "sz":
                    sz = int(dtype[2:])
                    diameters = diam_true.copy() * (30. / sz)
                    imgs_rsz = [transforms.resize_image(imgs[i].transpose(1,2,0), rsz=30./diameters[i]).transpose(2,0,1) for i in range(len(imgs))]
                    masks_true_rsz = [transforms.resize_image(masks_true[i], rsz=30./diameters[i], no_channels=True, interpolation=cv2.INTER_NEAREST) for i in range(len(imgs))]
                    return files, imgs_rsz, masks_true_rsz
                elif len(dtype.split("_")) > 1 and dtype.split("_")[0] in nstr.keys():
                    noise_type = dtype.split("_")[0]
                    k = int(dtype.split("_")[1])
                    print(noise_type, k)
                    if noise_type=="poisson":
                        param = np.array([5, 2.5, 0.5])
                    elif noise_type=="blur":
                        param = np.array([2, 4, 8])# 48])
                    elif noise_type=="downsample":
                        param = np.array([2, 5, 10])
                    elif noise_type=="aniso":
                        param = np.array([2, 6, 12])
                    if noise_type=="poisson":
                        params = {"poisson": 1.0, "blur": 0.0, "downsample": 0.0, "pscale": param[k]}
                    elif noise_type=="blur":
                        params = {"poisson": 1.0, "pscale": 120., "blur": 1.0, "downsample": 0.0,
                                    "sigma0": param[k], "sigma1": param[k]}
                    elif noise_type=="downsample":
                        params = {"poisson": 0.0, "pscale": 0., "blur": 1.0, "downsample": 1.0, "ds": param[k],
                                    "sigma0": param[k]/2, "sigma1": param[k]/2}
                    else:
                        params = {"poisson": 0.0, "pscale": 0., "blur": 1.0, "downsample": 1.0, "ds": param[k],
                                    "sigma0": param[k]/2, "sigma1": param[k]/2*0, "iso": False}

                    imgs_noisy = []
                    for img in imgs:
                        if params["pscale"] > 0:
                            img = np.maximum(0, img)
                        denoise.deterministic()
                        img = denoise.add_noise(torch.from_numpy(img).unsqueeze(0),
                                        **params).cpu().numpy().squeeze()
                        imgs_noisy.append(img)
                    return files, imgs_noisy, masks_true            

        return files, imgs, masks_true


def convert_images_cellposesam(imgs0):
    imgs = []
    for img in imgs0:
        img = img.astype(np.float32)
        if img.ndim==2:
            img = img[...,None]
        elif img.ndim==3 and img.shape[0] <= 3:
            img = img.transpose(1, 2, 0)
        img = transforms.normalize_img(img, axis=-1)
        imgs.append(img)
    return imgs


def run_cellpose4(dsets=None, mtype="cpsam"):
    io.logger_setup()
    if dsets is None:
        dsets = dsets_all

    root = Path(__file__).parent
    if mtype=='cpsam':
        pretrained_model = root / "../models/cpsam8_2000_162519454"
    elif mtype=='cpdino':
        pretrained_model = root / "../models/cp2000_0.0002_0.4_0.1_84002256"
    elif mtype=='cpdino-vitb':
        pretrained_model = root / "../models/cp2000_0.0002_0.4_0.1_636227846"
    elif mtype=='cpdino_ps16':
        pretrained_model = root / "../models/cpsam8_2000_162519454"
    elif mtype=='cpsam_linear':
        pretrained_model = root / "../models/cpsam8_1000_266324281_linearprobe"
    elif mtype=='cpdino-vitb_linear':
        pretrained_model = root / "../models/cp1000_0.002_0.4_0.1_659878492_linearprobe"        
    elif mtype=='cpdino_linear':
        pretrained_model = root / "../models/cp1000_0.002_0.4_0.1_195219516_linearprobe"
    
    model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)

    if mtype=="cpdino_ps16":
        model.net = vit.CPDINO(ps=16, model_name="vitl").to(model.net.device)
        model.net.load_model("../models/cp2000_0.0002_0.4_0.1_977471113", device=model.net.device)
        model.backbone = "vitl"
    
    for dset in dsets:
        print(dset)
        files, imgs, masks_true = load_dataset(dset)
        
        imgs = convert_images_cellposesam(imgs)

        root = Path(__file__).parent
        # root = Path("/home/carsen/dm11_string/datasets_cellpose/benchmarks/")
        
        runtime = []
        masks_pred = []
        flows = []
        for i in trange(len(imgs)):
            img = imgs[i]

            flip = None
            if dset[:5]=="cyto2":
                cperms = {"RGB": [0, 1, 2], "BGR": [2, 1, 0], "GBR": [1, 2, 0], "random": "random"}
                if len(dset.split("_")) > 1 and dset.split("_")[1] in cperms:
                    irgb = cperms[dset.split("_")[1]]
                    if irgb == 'random':
                        irgb = np.random.permutation(3)
                    if img.ndim > 2:
                        if img.shape[2] < 3:
                            img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], 3-img.shape[2]), dtype=img.dtype)), axis=2)
                        img = img[:,:,irgb]
                        if i==0:
                            print(irgb)
                            print(img.max(axis=(0,1)))
                elif len(dset.split("_")) > 1 and (dset.split("_")[1] == "lr" or dset.split("_")[1] == "ud"):
                    flip = dset.split("_")[1]
                    if dset.split("_")[1] == "lr":
                        img = img[:, ::-1]
                    else:
                        img = img[::-1]

            tic = time.time()
            masks_pred0, flows0, styles = model.eval(img, augment=False, 
                                        niter=2000 if "bac" in dset else None,
                                        bsize=None, tile_overlap=0.1, batch_size=64,
                                        flow_threshold=0.4, cellprob_threshold=-0.5)
            if flip is not None:
                if flip == "lr":
                    masks_pred0 = masks_pred0[:, ::-1]
                else:
                    masks_pred0 = masks_pred0[::-1]

            toc = time.time() - tic
            runtime.append(toc)
            masks_pred.append(masks_pred0)
            flows.append(flows0)
        runtime = np.array(runtime)

        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

        flows = flows if "linear" in mtype else None
        np.save(f"results/{mtype}_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime, "flows": flows, "test_files": files})


def convert_images_cellsam(imgs0, reduce=False, enlarge=False, bsize=512):
    imgs = []
    pads = []
    for img in imgs0:
        if img.ndim == 2:
            img = np.stack((np.zeros_like(img), np.zeros_like(img), img), axis=0)
        elif img.ndim==3:
            if np.array(img.shape).argmin() == 2:
                img = img.transpose(2, 0, 1)
            if img.shape[0] < 3:
                img = np.concatenate((np.zeros((3-img.shape[0], *img.shape[1:]), dtype=img.dtype), 
                                    img[::-1]), axis=0)
            else:
                img = np.concatenate((np.zeros((2, *img.shape[1:]), "float32"), 
                                    img.astype("float32").mean(axis=0, keepdims=True)), axis=0)
        img = img.astype(np.float32)
        for k in range(3):
            if np.ptp(img[k]) > 1e-3:
               img[k] = (img[k] - img[k].min()) / (img[k].max() - img[k].min())
        Ly, Lx = img.shape[1:]
        Lyr, Lxr = Ly, Lx
        if reduce:
            # reduce to bsize x bsize if larger
            Lyr = bsize if Ly > bsize and Ly > Lx else Ly
            Lxr = bsize if Lx > bsize and Lx >= Ly else Lx
            Lxr = int(np.round(bsize * (Lx / Ly))) if Ly > Lx and Lyr==bsize else Lxr
            Lyr = int(np.round(bsize * (Ly / Lx))) if Lx >= Ly and Lxr==bsize else Lyr
        
        if enlarge and Ly < bsize and Lx < bsize: 
            # resize to bsize x bsize if smaller and not pad
            Lxr = int(np.round(bsize * (Lx / Ly))) if Ly > Lx else bsize 
            Lyr = int(np.round(bsize * (Ly / Lx))) if Lx >= Ly else bsize
        
        if Lyr != Ly or Lxr != Lx:
            img = cv2.resize(img.transpose(1, 2, 0), (Lxr, Lyr), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        
        # pad if smaller than bsize x bsize
        padyx = [[0, 0], [0, 0]]
        if Lyr < bsize:
            padyx[0] = [int(np.floor((bsize-Lyr)/2)), int(np.ceil((bsize-Lyr)/2))]
        if Lxr < bsize:
            padyx[1] = [int(np.floor((bsize-Lxr)/2)), int(np.ceil((bsize-Lxr)/2))]
        if padyx[0][0] > 0 or padyx[0][1] > 0 or padyx[1][0] > 0 or padyx[1][1] > 0:
            img = np.pad(img, ((0,0), padyx[0], padyx[1]), mode='constant')
        pads.append(padyx)
        imgs.append(img)
    return imgs, pads


def run_cellsam(dsets=None):
    """ torchvision==0.15.1 so inconsistent with microsam """
    from cellSAM import cellsam_pipeline
    dset_args = {
            "deepbacs": {"enlarge": True, "reduce": False},
            "cyto2": {"enlarge": False, "reduce": True},
            "tissuenet": {"enlarge": True, "reduce": False},
            "bact_phase": {"enlarge": False, "reduce": False},
            "bact_fluor": {"enlarge": False, "reduce": False},
            "monuseg": {"enlarge": False, "reduce": False},
            "ovules": {"enlarge": False, "reduce": True},
            "root": {"enlarge": False, "reduce": True},
            "blastospim": {"enlarge": False, "reduce": True},
    }
    if dsets is None:
        dsets = dset_args.keys()

    for dset in dsets:
        print(dset)
        files, imgs, masks_true = load_dataset(dset)
        bsize = 512
        if dset[:5] == "cyto2":
            dset_arg = dset_args[dset[:5]]
            if dset[6:8] == "sz":
                bsize = 1024
        elif dset in dsets_others:
            dset_arg = dset_args["cyto2"]
            if dset != "ribo":
                bsize = 1024
        else:
            dset_arg = dset_args[dset]

        imgsCS, padCS = convert_images_cellsam(imgs, bsize=bsize, **dset_arg)
        masks_pred = []
        runtime = []
        for i in trange(len(imgs)):
            padyx = padCS[i]
            tic = time.time()
            try:
                #masks, _, _ = segment_cellular_image(imgsCS[i], device='cuda', 
                #                            normalize=False if dset != "deepbacs" else True)
                masks = cellsam_pipeline(imgsCS[i], use_wsi=False)
            except:
                print(i)
                masks = np.zeros((bsize, bsize), dtype="uint16")
            if padyx[0][0] > 0 or padyx[1][0] > 0 or padyx[0][1] > 0 or padyx[1][1] > 0:
                masks = masks[padyx[0][0] : masks.shape[0] - padyx[0][1], 
                            padyx[1][0] : masks.shape[1] - padyx[1][1]]
            Ly, Lx = imgs[i].shape[-2:] if np.array(imgs[i].shape).argmin() == 0 else imgs[i].shape[:2]
            if Ly != masks.shape[0] or Lx != masks.shape[1]:
                masks = cv2.resize(masks, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
            masks = fastremap.renumber(masks)[0]
            masks_pred.append(masks)
            runtime.append(time.time() - tic)

        runtime = np.array(runtime)
        
        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

        np.save(f"results/cellsam_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime})

def convert_images_samcell(imgs0, resize=True):
    imgs = []
    pads = []
    for i in trange(len(imgs0)):
        if imgs0[i].ndim == 2:
            img = imgs0[i].copy()
        elif imgs0[i].ndim == 3 and imgs0[i].shape[0] < 4:
            img = imgs0[i].transpose(1,2,0).copy()
        else:
            img = imgs0[i].copy()
        
        if resize:
            #resize longest side to 512
            if img.shape[0] > img.shape[1]:
                img = cv2.resize(img, (int(img.shape[1] * (512 / img.shape[0])), 512))
            else:
                img = cv2.resize(img, (512, int(img.shape[0] * (512 / img.shape[1]))))
            #pad to 512x512
            pads.append([[0, 512 - img.shape[0]], [0, 512 - img.shape[1]]])
            img = cv2.copyMakeBorder(img, 0, 512 - img.shape[0], 0, 512 - img.shape[1], cv2.BORDER_CONSTANT, value=0)
        else:
            pads.append([[0, 0], [0, 0]])

        if img.ndim == 3:
            if np.ptp(img[:,:,1]) != 0:
                img = img.mean(axis=-1)
            else:
                img = img[:, :, 0]
        img = img.astype("float32")
        img -= img.min()
        img /= img.max()
        img *= 255 
        img = img.astype("uint8")

        imgs.append(img)
        
    return imgs, pads

def run_samcell(dsets=None):
    """ will need to clone repo and pip install -r requirements.txt """
    if dsets is None:
        dsets = ["cyto2", "livecell"]
    import sys
    sys.path.insert(0, os.path.expanduser("~/github/SAMCell/src/"))
    from model import FinetunedSAM
    from pipeline import SlidingWindowPipeline
    
    model = FinetunedSAM("facebook/sam-vit-base", finetune_vision=False, finetune_prompt=True, finetune_decoder=True)
    
    pipeline = SlidingWindowPipeline(model, 'cuda', crop_size=256)

    for dset in dsets:
        print(dset)
        model_name = "livecell" if dset=="livecell" else "cyto"
        trained_samcell_path = os.path.expanduser(f"~/github/SAMCell/samcell-{model_name}/pytorch_model.bin")
        model.load_weights(trained_samcell_path)
        files, imgs, masks_true = load_dataset(dset)

        imgsCS, padCS = convert_images_samcell(imgs, resize=False if dset=="livecell" else True)
        masks_pred = []
        runtime = []
        for i in trange(len(imgs)):
            padyx = padCS[i]
            tic = time.time()
            masks = pipeline.run(imgsCS[i])
            if padyx[0][0] > 0 or padyx[1][0] > 0 or padyx[0][1] > 0 or padyx[1][1] > 0:
                masks = masks[padyx[0][0] : masks.shape[0] - padyx[0][1], 
                            padyx[1][0] : masks.shape[1] - padyx[1][1]]
            Ly, Lx = imgs[i].shape[-2:] if np.array(imgs[i].shape).argmin() == 0 else imgs[i].shape[:2]
            if Ly != masks.shape[0] or Lx != masks.shape[1]:
                masks = cv2.resize(masks, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
            masks = fastremap.renumber(masks)[0]
            masks_pred.append(masks)
            runtime.append(time.time() - tic)

        runtime = np.array(runtime)
        
        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true[:len(masks_pred)], masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

        np.save(f"results/samcell_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime})


def run_cyto3(dsets=None, restore=False):
    """ will need to install cellpose==3.1.1.2 in new env """
    if dsets is None:
        dsets = dsets_all
        dsets.remove("monuseg")
    
    print(dsets, restore)
    rstr = "_restore" if restore else ""

    io.logger_setup()

    model = models.Cellpose(gpu=True, model_type="cyto3")
    
    for dset in dsets:
        print(dset)
        files, imgs, masks_true = load_dataset(dset)
        if "cyto2_" in dset:
            diam_true = np.array([utils.diameters(m)[0] for m in masks_true])
            if restore:
                noise_type = dset.split("_")[1]
                mstr = "cyto3" if noise_type!="aniso" else "cyto2"
                noise_type = dset.split("_")[1]
                model_type = f"{nstr[noise_type]}_{mstr}"
                dn_model = denoise.DenoiseModel(gpu=True, model_type=model_type, chan2=True)

        masks_pred = []
        runtime = []
        runtime_size = []

        diams = []
        for i in trange(len(imgs)):
            Ly, Lx = masks_true[i].shape
            img = imgs[i]
            tic = time.time()
            if "cyto2_" not in dset:
                diam = model.sz.eval(img, channels=[1,2], batch_size=64, augment=True)[0]
            else:
                if "sz" not in dset:
                    if restore:
                        noise_type = dset.split("_")[1]
                        if noise_type=="downsample" or noise_type=="aniso":
                            img = transforms.resize_image(img.transpose(1,2,0).copy(), rsz=30./diam_true[i]).transpose(2,0,1)
                        diam = diam_true[i] if noise_type!="downsample" and noise_type!="aniso" else 30.
                        img = dn_model.eval(img, diameter=diam, channels=[1,2], channel_axis=0)
                    else:
                        diam = diam_true[i]
                else:
                    diam = 30. # do not resize for cell size analysis
            runtime_size.append(time.time() - tic)
            masks_pred0, flows, styles = model.cp.eval(img, diameter=diam, channels=[1,2], 
                                            niter=2000 if "bac" in dset else None,
                                            bsize=224, tile_overlap=0.5, batch_size=64, augment=True,
                                            flow_threshold=0.4, cellprob_threshold=-0.5)
            toc = time.time() - tic
            
            if masks_pred0.shape[0] != Ly or masks_pred0.shape[1] != Lx:
                masks_pred0 = transforms.resize_image(masks_pred0, Ly=Ly, Lx=Lx, 
                                                      no_channels=True, 
                                                      interpolation=cv2.INTER_NEAREST)                

            diams.append(diam)
            runtime.append(toc)
            masks_pred.append(masks_pred0)

        runtime = np.array(runtime)
        runtime_size = np.array(runtime_size)
        
        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])
        
        np.save(f"results/cyto3{rstr}_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime})
        

def run_omnipose(dsets=None):
    from cellpose_omni.io import logger_setup
    from cellpose_omni import core, models
    import omnipose

    if dsets is None:
        dsets = ["bact_phase", "bact_fluor"]
    for dset in dsets:
        model = models.CellposeModel(gpu=True, model_type=f'{dset}_omni')
        chans = [0,0] if "bact" in dset else [2,1]
            
        files, imgs, masks_true = load_dataset(dset)

        logger,log_file=logger_setup()
        resample = False if dset=="cyto2" else True
        diam_threshold = 30 if dset=="cyto2" else 12

        masks_pred = []
        runtime = []
        if dset == "cyto2":
            diameters = [omnipose.core.diameters(mask) for mask in masks_true]
        else:
            diameters = [None for mask in masks_true]
        for i in trange(len(imgs)):
            tic = time.time()
            masks, flows, styles = model.eval(imgs[i],channels=chans,diameter=diameters[i],mask_threshold=-1,flow_threshold=0, diam_threshold=diam_threshold,
                                                            omni=True,cluster=True,resample=resample,tile=False,
                                                            verbose=False)
            masks = masks[0] if isinstance(masks, list) else masks
            toc = time.time() - tic 
            masks_pred.append(masks)
            runtime.append(toc)

        runtime = np.array(runtime)
        
        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

        np.save(f"results/omnipose_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime})


def convert_images_microsam(imgs0):
    imgs = []
    for img in imgs0:
        if img.ndim == 3:
            if np.array(img.shape).argmin() == 0:
                img = img.transpose(1, 2, 0)
            if np.ptp(img[:,:,1]) < 1e-3:
                img = img[:,:,0]
            elif img.shape[2] < 3:
                img = np.concatenate((img,
                                      np.zeros((img.shape[0], img.shape[1], 3-img.shape[2]), dtype=img.dtype)), axis=2)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        
        img = img.astype(np.float32)
        imgs.append(img)
    return imgs


def run_microsam(dsets=None):
    """ needs separate env from cellsam
    conda install -c conda-forge microsam
    """
    if dsets is None:
        dsets = ["livecell", "cyto2", "deepbacs", "tissuenet", 
                 "bact_phase", "bact_fluor", "blastospim"]

    from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

    model_type = "vit_l_lm"

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with your custom checkpoint.
        segmentation_mode='apg',
        is_tiled=False,  # Switch to 'True' in case you would like to perform tiling-window based prediction.
    )

    for dset in dsets:
        print(dset)
        files, imgs, masks_true = load_dataset(dset)
        imgs = convert_images_microsam(imgs)

        masks_pred = []
        runtime = []
        for i in trange(len(imgs)):
            tic = time.time()
            image = imgs[i]
            generate_kwargs = {
                "center_distance_threshold": 0.5,
                "boundary_distance_threshold": 0.5,
                "foreground_threshold": 0.5,
                "nms_threshold": 0.9,
            }
            try:
                prediction = automatic_instance_segmentation(
                    predictor=predictor,
                    segmenter=segmenter,
                    input_path=image,
                    ndim=2,
                    tile_shape=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a tile shape
                    halo=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a halo shape.
                    **generate_kwargs
                )
            except Exception as e:
                print(f"Error processing image {i} in dataset {dset}: {e}")
                prediction = np.zeros_like(masks_true[i], dtype="uint16")
            masks_pred.append(prediction)
            toc = time.time() - tic
            runtime.append(toc)

        runtime = np.array(runtime)
        
        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

        np.save(f"results/microsam_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime})

            
def run_pathosam(dsets=None):
    from patho_sam import automatic_segmentation
    files, imgs, masks_true = load_dataset("monuseg")
    masks_pred = []
    runtime = []
    for i in trange(len(imgs)):
        tic = time.time()
        masks_pred0 = automatic_segmentation.automatic_segmentation_wsi(imgs[i], 
                                                  model_type="vit_l_histopathology",
                                                  output_path=f"results/monuseg_{i}")

        toc = time.time() - tic
        runtime.append(toc)
        masks_pred.append(masks_pred0.astype("uint16"))

    runtime = np.array(runtime)
    
    threshold = np.arange(0.5, 1., 0.05)
    ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
    print(ap.mean(axis=0)[[0, 5, 8]])
    print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

    np.save(f"results/pathosam_monuseg.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                        "masks_true": masks_true, "masks_pred": masks_pred, 
                                        "runtime": runtime})


if __name__ == "__main__":
    # add algorithm arg 
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="cpsam", 
                        choices=["cpsam", "cpdino", "cpdino-vitb", "cpdino_ps16", "cpdino_linear", "cpdino-vitb_linear", "cpsam_linear",
                                 "cellsam", "samcell", "cyto3", "omnipose", "pathosam", "microsam"])
    parser.add_argument("--dsets", type=str, nargs="+", default=None)
    parser.add_argument("--restore", action="store_true", help="whether to perform image restoration for cyto3")

    args = parser.parse_args()
    alg = args.alg
    dsets = args.dsets
    restore = args.restore
    
    if (alg != "cellsam" and alg != "samcell" and alg != "omnipose" and alg != "cyto3"
        and alg != "pathosam" and alg != "microsam"):
        run_cellpose4(dsets=dsets, mtype=alg)
    elif alg == "cellsam":
        run_cellsam(dsets=dsets)
    elif alg == "samcell":
        run_samcell(dsets=dsets)
    elif alg == "omnipose":
        run_omnipose(dsets=dsets)
    elif alg == "pathosam":
        run_pathosam(dsets=dsets)
    elif alg == "microsam":
        run_microsam(dsets=dsets)
    elif alg == "cyto3":
        run_cyto3(dsets=dsets, restore=restore)
    
