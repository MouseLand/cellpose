"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from cellpose import io, transforms, models, metrics, denoise
from natsort import natsorted 
from pathlib import Path
import torch
from torch import nn

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
    import segmentation_models_pytorch as smp

    class Transformer(nn.Module):
        def __init__(self, pretrained_model=None, encoder="mit_b5", encoder_weights="imagenet", decoder="FPN"):
            super().__init__()
            net_fcn = smp.FPN if decoder=="FPN" else smp.MAnet
            self.net = net_fcn(encoder_name=encoder,        
                            encoder_weights=encoder_weights if pretrained_model is None else None,     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3, classes=3, activation=None)
            self.nout = 3
            self.mkldnn = False
            if pretrained_model is not None:
                state_dict = torch.load(pretrained_model)
                if list(state_dict.keys())[0][:7]=="module.":
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                        new_state_dict[name] = v
                    self.net.load_state_dict(new_state_dict)
                else:
                    self.load_state_dict(state_dict)
                
        def forward(self, X):
            X = torch.cat((X, torch.zeros((X.shape[0], 1, X.shape[2], X.shape[3]), device=X.device)), dim=1)
            y = self.net(X)
            return y, torch.zeros((X.shape[0], 256), device=X.device)
        
        @property
        def device(self):
            return next(self.parameters()).device

except Exception as e:
    print(e)
    print("need to install segmentation_models_pytorch to run transformer")

model_names = {"poisson": "denoise", "blur": "deblur", "downsample": "upsample"}

def seg_eval_cp3(folder, noise_type="poisson"):
    """ need to download test_poisson.npy, test_blur.npy, test_downsample.npy
    (for cells and/or nuclei)
    
    (was computed with old flows, but results similar with new flows) """
    ctypes = ["cyto2", "nuclei"]
    for ctype in ctypes:
        folder_name = ctype
        diam_mean = 30 if ctype=="cyto2" else 17
        root = Path(folder) / f"images_{folder_name}/"

        ### cellpose enhance
        dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy", allow_pickle=True).item()
        test_noisy = dat["test_noisy"]
        masks_true = dat["masks_true"]
        diam_test = dat["diam_test"] if "diam_test" in dat else 30. * np.ones(len(test_noisy))
        
        istr = ["rec", "seg", "per", "perseg"]
        for k in range(len(istr)):
            model_name = model_names[noise_type]
            if istr[k] != "perseg":
                model_name += "_" + istr[k]
            model = denoise.DenoiseModel(gpu=True, nchan=1, diam_mean=diam_mean,
                                         model_type=f"{model_name}_{ctype}")
            imgs2 = model.eval([test_noisy[i][0] for i in range(len(test_noisy))], diameter=diam_test, 
                                channel_axis=0)
            print(imgs2[0].shape)
            seg_model = models.CellposeModel(gpu=True, model_type=ctype)
            masks2, flows2, styles2 = seg_model.eval(imgs2, channels=[1,0], diameter=diam_test, 
                                                    channel_axis=0, normalize=True)
            flows = [flow[0] for flow in flows2]

            ap, tp, fp, fn = metrics.average_precision(masks_true, masks2)
            if ctype=="cyto2":
                print(f"{istr[k]} AP@0.5 \t = {ap[:68,0].mean(axis=0):.3f}")
            else:
                print(f"{istr[k]} AP@0.5 \t = {ap[:,0].mean(axis=0):.3f}")

            dat[f"test_{istr[k]}"] = imgs2
            dat[f"masks_{istr[k]}"] = masks2
            dat[f"flows_{istr[k]}"] = flows

        #np.save(root / "noisy_test" / f"test_{noise_type}_cp3.npy", dat)

        if noise_type=="poisson":
            ### cellpose retrained
            dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy", allow_pickle=True).item()
            test_noisy = dat["test_noisy"]
            masks_true = dat["masks_true"]
            diam_test = dat["diam_test"] if "diam_test" in dat else 30. * np.ones(len(test_noisy))

            seg_model = models.CellposeModel(gpu=True, nchan=1, model_type=f"{ctype}_noisy")
            masks2 = seg_model.eval(test_noisy, channels=None, diameter=diam_test, 
                                        normalize=False)[0]

            ap, tp, fp, fn = metrics.average_precision(masks_true, masks2)

            dat[f"masks_retrain"] = masks2

            #np.save(root / "noisy_test" / f"test_{noise_type}_cp_retrain.npy", dat)

def blind_denoising(folder):
    ctypes = ["cyto2", "nuclei"]
    for ctype in ctypes:
        root = Path(folder) / f"images_{ctype}/"
        
        ### noise2void
        imgs_n2v, masks_n2v = noise2void.train_per_image_synthetic(root, ctype=ctype, plot=0, save=False)
        dat = np.load(root / "noisy_test" / f"test_poisson.npy", allow_pickle=True).item()
        masks_true = dat["masks_true"]
        print(len(masks_n2v), len(masks_true))
        ap = metrics.average_precision(masks_true, masks_n2v)[0]
        print(ap.mean(axis=0))

        ### noise2self
        imgs_n2s, masks_n2s = noise2self.train_per_image_synthetic(root, ctype=ctype, plot=0, save=True)
        dat = np.load(root / "noisy_test" / f"test_poisson.npy", allow_pickle=True).item()
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
        if dset=="Projection_Flywing":
            model_type = "denoise_cyto2"
            diam_mean = 30.
            cp_model_type = "cyto2"
        else:
            model_type = "denoise_nuclei"
            diam_mean = 17.
            cp_model_type = "nuclei"
            
        if dset=="Projection_Flywing":
            clean = [transforms.normalize99(io.imread(tif)) for tif in natsorted(root.glob("proj_C2*.tif"))]
            diam = 20.
            cellprob_threshold = -2.
        else:
            imgs = [io.imread(tif) for tif in natsorted(root.glob("*.tif"))]
            dz = 8
            clean = [transforms.normalize99(img[-1][img.shape[1]//2 - dz : img.shape[1]//2 + dz].max(axis=0)) for img in imgs]
            diam = 12.
            cellprob_threshold = 0.
        
        print(diam_mean)
        model = denoise.DenoiseModel(gpu=True, nchan=1, diam_mean=diam_mean,
                                        model_type=model_type)
        seg_model = models.CellposeModel(gpu=True, model_type=cp_model_type, diam_mean=diam_mean)
        masks_clean = seg_model.eval(clean, diameter=diam, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)[0]
        dat["clean"] = clean 
        dat["masks_clean"] = masks_clean 
        dat["noisy"] = []
        dat["ap_noisy"] = []
        dat["masks_noisy"] = []
        dat["denoised"] = []
        dat["masks_denoised"] = []
        dat["ap_denoised"] = []

        plt.figure(figsize=(4,4))
        for nl in range(3):
            if dset=="Projection_Flywing":
                nstr = f"proj_C{nl+(nl>1)}"
                print(nstr)
                noisy = [transforms.normalize99(io.imread(tif)) for tif in natsorted(root.glob(f"{nstr}*.tif"))]
            elif dset=="Denoising_Tribolium":
                noisy = [transforms.normalize99(img[nl][img.shape[1]//2 - dz : img.shape[1]//2 + dz].max(axis=0)) for img in imgs] #.max(axis=0)
            else:
                noisy = [io.imread(tif) for tif in natsorted((root / f"condition_{nl+1}").glob("*.tif"))]
                noisy = [transforms.normalize99(img[img.shape[0]//4 : img.shape[0]//4 + dz].max(axis=0)) for img in noisy]
            
            denoised = model.eval(noisy, channels=None, diameter=diam)
            masks_denoised = seg_model.eval(denoised, diameter=diam, cellprob_threshold=cellprob_threshold, 
                                            flow_threshold=flow_threshold, 
                                            )[0]
            masks_noisy = seg_model.eval(noisy, diameter=diam, cellprob_threshold=cellprob_threshold, 
                    flow_threshold=flow_threshold)[0]        

            thresholds = np.arange(0.5, 1.05, 0.05)
            ap_noisy = metrics.average_precision(masks_clean, masks_noisy, threshold=thresholds)[0]
            ap_denoised = metrics.average_precision(masks_clean, masks_denoised, threshold=thresholds)[0]
        
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
            pred_val = noise2self.train_per_image(noisy[i][np.newaxis,...].astype("float32"))
            masks = seg_model.eval(pred_val, diameter=diam, cellprob_threshold=cellprob_threshold, channels=[1,0], 
                                channel_axis=0, normalize=True)[0]
            imgs_n2s.append(pred_val)
            masks_n2s.append(masks)

            print(f">>> IMAGE {i}, n_masks = {masks.max()}")
            
        ap_n2s = [metrics.average_precision(masks_clean, masks_n2s[nl*len(masks_clean):(nl+1)*len(masks_clean)], threshold=thresholds)[0] for nl in range(3)]
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
            masks = seg_model.eval(pred_val, diameter=diam, cellprob_threshold=cellprob_threshold, channels=[1,0], 
                                channel_axis=0, normalize=True)[0]
            imgs_n2v.append(pred_val)
            masks_n2v.append(masks)

            print(f">>> IMAGE {i}, n_masks = {masks.max()}")
            
        ap_n2v = [metrics.average_precision(masks_clean, masks_n2v[nl*len(masks_clean):(nl+1)*len(masks_clean)], threshold=thresholds)[0] for nl in range(3)]
        dat2 = {}
        dat2["denoised_n2v"] = imgs_n2v
        dat2["masks_n2v"] = masks_n2v
        dat2["noisy"] = noisy 
        dat2["masks_clean"] = masks_clean
        dat2["ap_n2v"] = ap_n2v
        np.save(root / "n2v_masks.npy", dat2)

def specialist_training(root):
    """ root is path to specialist images (first 89 images of cyto2 and first 11 test images) """
    
    # make patches for training CARE, noise2self and noise2void
    care.CIL_dataset(root)

    # train/test CARE
    care.train_test_specialist(root)

    # train/test noise2self
    imgs, masks_n2s, ap = noise2self.train_test_specialist(root, n_epochs=50, lr=5e-4, test=True)

    # train/test noise2void
    noise2void.train_test_specialist(root, n_epochs=100, lr=4e-4, test=True);


def seg_eval_oneclick(folder):
    noise_types = ["poisson", "blur", "downsample"]
    ctypes = ["cyto2", "nuclei"]
    for c, ctype in enumerate(ctypes):
        folder_name = ctype
        diam_mean = 30.
        root = Path(f"/media/carsen/ssd4/datasets_cellpose/images_{folder_name}/")
        print(ctype)
        for n, noise_type in enumerate(noise_types):
            print(noise_type)
            ### cellpose enhance
            dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy", allow_pickle=True).item()
            test_noisy = dat["test_noisy"]
            masks_true = dat["masks_true"]
            diam_test = dat["diam_test"] if "diam_test" in dat else 30. * np.ones(len(test_noisy))

            model = denoise.DenoiseModel(gpu=True, nchan=1, diam_mean=diam_mean,
                                         model_type=model_names[noise_type]+"_cyto3", 
                                         device=torch.device("cuda"))
            imgs2 = model.eval([test_noisy[i][0] for i in range(len(test_noisy))], diameter=diam_test, 
                                channel_axis=0)
            
            seg_model = models.CellposeModel(gpu=True, model_type=ctype, 
                                    device=torch.device("cuda"))
            masks2, flows2, styles2 = seg_model.eval(imgs2, channels=[1,0], diameter=diam_test, 
                                                    channel_axis=0, normalize=True)
            istr = "generalist"
            ap, tp, fp, fn = metrics.average_precision(masks_true, masks2)
            if ctype=="cyto2":
                print(f"{istr} AP@0.5 \t = {ap[:68,0].mean(axis=0):.3f}")
            else:
                print(f"{istr} AP@0.5 \t = {ap[:,0].mean(axis=0):.3f}")

            dat[f"test_{istr}"] = imgs2
            dat[f"masks_{istr}"] = masks2
            
            np.save(root / "noisy_test" / f"test_{noise_type}_generalist_cp3.npy", dat)

def cyto3_comparisons(folder):
    """  diameters computed from generalist model cyto3
    will need segmentation_models_pytorch to run transformer """
    root = Path(folder)
    folders = ["cyto2", "nuclei", "tissuenet", "livecell", "yeast_BF", "yeast_PhC", "bact_phase", "bact_fluor", "deepbacs"]
    
    net_types = ["generalist", "specialist", 
                "transformer"]
    for net_type in net_types:
        if net_type=="generalist":
            seg_model = models.Cellpose(gpu=True, model_type="cyto3")
        elif net_type=="transformer":
            seg_model = models.CellposeModel(gpu=True, pretrained_model=None)
            pretrained_model = "/home/carsen/.cellpose/models/transformer_cp3"
            seg_model.net = Transformer(pretrained_model=pretrained_model, decoder="MAnet").to(device)
        for f in folders:
            if net_type=="specialist":
                seg_model = models.CellposeModel(gpu=True, model_type=f"{f}_cp3")

            root = Path(folder) / f"images_{f}" 
            channels = [1,2] if f=="tissuenet" or f=="cyto2" else [1,0]
            tifs = (root / "test").glob("*.tif")
            tifs = [tif for tif in tifs]
            tifs = [tif for tif in tifs if "_masks" not in str(tif) and "_flows" not in str(tif)]
            if net_type!="generalist":
                d = np.load(f"/media/carsen/ssd4/datasets_cellpose/{f}_generalist_masks.npy", allow_pickle=True).item()
                diams = d["diams"]
            else:
                diams = 0 
            
            imgs = [io.imread(tif) for tif in tifs]
            flows = [io.imread(str(tif)[:-4]+"_flows.tif") for tif in tifs]
            masks = [flow[0].astype("uint16") for flow in flows]

            if f=="cyto2":
                imgs = imgs[:68]
                masks = masks[:68]

            dat = {}
            dat["imgs"] = imgs 
            dat["masks"] = masks
            dat["files"] = tifs
            out = seg_model.eval(imgs, diameter=diams, channels=channels, 
                                    tile_overlap=0.5, flow_threshold=0.4, 
                                    augment=True, bsize=224,
                                    niter=2000 if f=="bact_phase" else None)
            if len(out)==3:
                masks_pred, flows_pred, styles = out 
            else:
                masks_pred, flows_pred, styles, diams = out
            ap, tp, fp, fn = metrics.average_precision(masks, masks_pred, 
                                                    threshold=np.arange(0.5, 1.05, 0.05))
            print(ap.mean(axis=0))
            dat["masks_pred"] = masks_pred
            dat["performance"] = [ap, tp, fp, fn]
            dat["diams"] = diams
                    
            #p.save(f"/media/carsen/ssd4/datasets_cellpose/{f}_{net_type}_masks.npy", dat)

if __name__ == '__main__':
    # folder with folders images_cyto2 and images_nuclei for those datasets
    folder = "/media/carsen/ssd4/datasets_cellpose/"

    # https://publications.mpi-cbg.de/publications-sites/7207/
    folder2 = "/media/carsen/ssd4/denoising/" # download CARE datasets here

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
