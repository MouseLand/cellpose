import cv2 
import numpy as np
from cellpose import transforms, io, models, metrics
from pathlib import Path
from natsort import natsorted
import time
from tqdm import trange
import fastremap

dsets_all = ["deepbacs", "bact_phase", "bact_fluor", "tissuenet", "cyto2", 
        "monuseg", "livecell"]

def load_dataset(dset):
    if dset!= "monuseg":
        files = [f for f in Path(f"/media/carsen/ssd3/datasets_cellpose/images_{dset}/test/").glob("*.tif")]
        files = natsorted([f for f in files if "_masks" not in str(f) and "_flows" not in str(f)])
        masks_true = [io.imread(str(f).replace(".tif", "_flows.tif"))[0].astype("uint16") for f in files]
    else:
        files = [f for f in Path(f"/media/carsen/ssd3/datasets_cellpose/images_HandE/MoNuSeg/MoNuSegTestData/").glob("*.tif")]
        files = natsorted([f for f in files if "_masks" not in str(f)])
        masks_true = [io.imread(str(f).replace(".tif", "_masks.tif")).astype("uint16") for f in files]
    imgs = [io.imread(f) for f in files]

    if dset=="cyto2":
        ind_im = np.array([68, 69, 71, 72, 73, 74, 75, 76, 84, 86, 89, 90])
        ind_im = np.hstack((np.arange(55), ind_im))
        imgs = [imgs[i] for i in ind_im]
        masks_true = [masks_true[i] for i in ind_im]

    return files, imgs, masks_true

def convert_images_cellposesam(imgs0):
    imgs = []
    for img in imgs0:
        if img.ndim == 2:
            img = np.stack((np.zeros_like(img), np.zeros_like(img), img), axis=0)
        elif img.ndim == 3:
            if np.array(img.shape).argmin() == 2:
                img = img.transpose(2, 0, 1)
            if img.shape[0] < 3:
                img = np.concatenate((np.zeros((3-img.shape[0], *img.shape[1:]), dtype=img.dtype), 
                                    img), axis=0)
        img = img.astype(np.float32)
        img = transforms.normalize_img(img, axis=0)
        imgs.append(img)
    return imgs

def run_cellposesam(dsets=None):
    if dsets is None:
        dsets = dsets_all
    for dset in dsets:
        print(dset)
        files, imgs, masks_true = load_dataset(dset)
        
        imgs = convert_images_cellposesam(imgs)
        model = models.CellposeModel(gpu=True, nchan=3, 
                                    pretrained_model="models/cpsam8_0_2100_8_402175188")

        runtime = []
        masks_pred = []
        for i in trange(len(imgs)):
            img = imgs[i]
            tic = time.time()
            masks_pred0, flows, styles = model.eval(img, augment=False, 
                                        niter=2000 if "bac" in dset else None,
                                        bsize=256, tile_overlap=0.1, batch_size=64,
                                        flow_threshold=0.4, cellprob_threshold=0)
            toc = time.time() - tic
            runtime.append(toc)
            masks_pred.append(masks_pred0)
        runtime = np.array(runtime)

        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

        np.save(f"results/cellposesam_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime, "test_files": files})


def convert_images_cellsam(imgs0, reduce=False, enlarge=False):
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
            # reduce to 512 x 512 if larger
            Lyr = 512 if Ly > 512 and Ly > Lx else Ly
            Lxr = 512 if Lx > 512 and Lx >= Ly else Lx
            Lxr = int(np.round(512 * (Lx / Ly))) if Ly > Lx and Lyr==512 else Lxr
            Lyr = int(np.round(512 * (Ly / Lx))) if Lx >= Ly and Lxr==512 else Lyr
        
        if enlarge and Ly < 512 and Lx < 512: 
            # resize to 512 x 512 if smaller and not pad
            Lxr = int(np.round(512 * (Lx / Ly))) if Ly > Lx else 512 
            Lyr = int(np.round(512 * (Ly / Lx))) if Lx >= Ly else 512
        
        if Lyr != Ly or Lxr != Lx:
            img = cv2.resize(img.transpose(1, 2, 0), (Lxr, Lyr), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        
        # pad if smaller than 512 x 512
        padyx = [[0, 0], [0, 0]]
        if Lyr < 512:
            padyx[0] = [int(np.floor((512-Lyr)/2)), int(np.ceil((512-Lyr)/2))]
        if Lxr < 512:
            padyx[1] = [int(np.floor((512-Lxr)/2)), int(np.ceil((512-Lxr)/2))]
        if padyx[0][0] > 0 or padyx[0][1] > 0 or padyx[1][0] > 0 or padyx[1][1] > 0:
            img = np.pad(img, ((0,0), padyx[0], padyx[1]), mode='constant')
        pads.append(padyx)
        imgs.append(img)
    return imgs, pads

def run_cellsam(dsets=None):
    """ torchvision==0.15.1 so inconsistent with microsam """
    from cellSAM import segment_cellular_image
    dset_args = {
            "deepbacs": {"enlarge": True, "reduce": False},
            "cyto2": {"enlarge": False, "reduce": True},
            "tissuenet": {"enlarge": True, "reduce": False},
            "bact_phase": {"enlarge": False, "reduce": False},
            "bact_fluor": {"enlarge": False, "reduce": False,},
            "monuseg": {"enlarge": False, "reduce": False},
    }
    if dsets is None:
        dsets = dset_args.keys()

    for dset in dsets:
        print(dset)
        files, imgs, masks_true = load_dataset(dset)

        imgsCS, padCS = convert_images_cellsam(imgs, **dset_args[dset])
        masks_pred = []
        runtime = []
        for i in trange(len(imgs)):
            padyx = padCS[i]
            tic = time.time()
            try:
                masks, _, _ = segment_cellular_image(imgsCS[i], device='cuda', 
                                            normalize=False if dset != "deepbacs" else True)
            except:
                print(i)
                masks = np.zeros((512, 512), dtype="uint16")
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
    from model import FinetunedSAM
    from pipeline import SlidingWindowPipeline
    
    model = FinetunedSAM("facebook/sam-vit-base", finetune_vision=False, finetune_prompt=True, finetune_decoder=True)
    
    pipeline = SlidingWindowPipeline(model, 'cuda', crop_size=256)

    for dset in dsets:
        print(dset)
        model_name = "cyto" if dset=="cyto2" else "livecell"
        trained_samcell_path = f"/github/SAMCell/samcell-{model_name}/pytorch_model.bin"
        model.load_weights(trained_samcell_path)
        files, imgs, masks_true = load_dataset(dset)

        imgsCS, padCS = convert_images_samcell(imgs, resize=True if dset=="cyto2" else False)
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


def run_cyto3_segformer(dsets=None):
    """ will need to install cellpose==3.1.1.1 in new env """
    if dsets is None:
        dsets = dsets_all
        dsets.remove("monuseg")
    print(dsets)
    for dset in dsets:
        print(dset)
        files, imgs, masks_true = load_dataset(dset)
        
        masks_pred = []
        runtime = []
        runtime_size = []

        model = models.Cellpose(gpu=True, model_type="cyto3")
        diams = []
        for i in trange(len(imgs)):
            img = imgs[i]
            tic = time.time()
            diam = model.sz.eval(img, channels=[1,2], batch_size=64, augment=True)[0]
            runtime_size.append(time.time() - tic)
            masks_pred0, flows, styles = model.cp.eval(img, diameter=diam, channels=[1,2], 
                                            niter=2000 if "bac" in dset else None,
                                            bsize=224, tile_overlap=0.5, batch_size=64, augment=True,
                                            flow_threshold=0.4, cellprob_threshold=0)
            toc = time.time() - tic
            diams.append(diam)
            runtime.append(toc)
            masks_pred.append(masks_pred0)

        runtime = np.array(runtime)
        runtime_size = np.array(runtime_size)
        
        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])
        
        np.save(f"results/cyto3_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
                                            "masks_true": masks_true, "masks_pred": masks_pred, 
                                            "runtime": runtime})

        model = models.CellposeModel(gpu=True, backbone="transformer",
                                    model_type="transformer_cp3")
        masks_pred = []
        runtime = []
        for i in trange(len(imgs)):
            img = imgs[i]
            diam = diams[i]
            tic = time.time()
            masks_pred0, flows, styles = model.eval(img, diameter=diam, channels=None, 
                                            niter=1000 if "bac" in dset else None,
                                            bsize=224, tile_overlap=0.5, batch_size=64, augment=True,
                                            flow_threshold=0.4, cellprob_threshold=0)
            toc = time.time() - tic
            diams.append(diam)
            runtime.append(toc)
            masks_pred.append(masks_pred0)

        runtime = np.array(runtime)
        runtime += runtime_size
        
        threshold = np.arange(0.5, 1., 0.05)
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        print(ap.mean(axis=0)[[0, 5, 8]])
        print(((fp + fn) / (fn + tp)).mean(axis=0)[[0, 5, 8]])

        np.save(f"results/segformer_{dset}.npy", {"ap": ap, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold,
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
            if np.array(img.shape).argmin() == 2:
                img = img.transpose(2, 0, 1)
            if np.ptp(img[1]) != 0 :
                img = img.astype("float32").mean(axis=0)
            else:
                img = img[0]
            
        img = np.stack((img,)*3, axis=-1)
        img = img.astype(np.float32)
        imgs.append(img)
    return imgs

def run_microsam(dsets=None):
    """ needs separate env from cellsam
    conda install -c conda-forge microsam
    """
    if dsets is None:
        dsets = ["livecell", "cyto2", "deepbacs", "tissuenet"]

    from micro_sam import util
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
        AutomaticMaskGenerator,
        get_predictor_and_decoder,
        mask_data_to_segmentation
    )

    model_type = "vit_l_lm"

    # Step 1: Initialize the model attributes using the pretrained µsam model weights.
    #   - the 'predictor' object for generating predictions using the Segment Anything model.
    #   - the 'decoder' backbone (for AIS).
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint_path=None, #checkpoint_path="/home/carsen/Downloads/vit_l.pt",  # overwrite to pass our own finetuned model
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

            # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.
            image_embeddings = util.precompute_image_embeddings(
                predictor=predictor,  # the predictor object responsible for generating predictions
                input_=image,  # the input image
                ndim=2,  # number of input dimensions
                verbose=False,
            )

            # Step 3: Combining the decoder with the Segment Anything backbone for automatic instance segmentation.
            ais = InstanceSegmentationWithDecoder(predictor, decoder)

            # Step 4: Initializing the precomputed image embeddings to perform faster automatic instance segmentation.
            ais.initialize(
                image=image,  # the input image
                image_embeddings=image_embeddings,  # precomputed image embeddings
            )

            # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.
            prediction = ais.generate()
            if len(prediction) > 0:
                prediction = mask_data_to_segmentation(prediction, with_background=True)
                masks_pred.append(prediction)
            else:
                masks_pred.append(np.zeros(image.shape[:2], "uint16"))
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

            
def run_pathosam():
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
