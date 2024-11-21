"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch
from scipy.ndimage import gaussian_filter
import cv2
import gc

import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .resnet_torch import CPnet
from .core import assign_device, check_mkl, run_net, run_3D

_MODEL_URL = "https://www.cellpose.org/models"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath(".cellpose", "models")
MODEL_DIR = pathlib.Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = [
    "cyto3", "nuclei", "cyto2_cp3", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3",
    "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "cyto2", "cyto", "CPx",
    "transformer_cp3", "neurips_cellpose_default", "neurips_cellpose_transformer",
    "neurips_grayscale_cyto2",
    "CP", "CPx", "TN1", "TN2", "TN3", "LC1", "LC2", "LC3", "LC4"
]

MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath("gui_models.txt"))

normalize_default = {
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False
}


def model_path(model_type, model_index=0):
    torch_str = "torch"
    if model_type == "cyto" or model_type == "cyto2" or model_type == "nuclei":
        basename = "%s%s_%d" % (model_type, torch_str, model_index)
    else:
        basename = model_type
    return cache_model_path(basename)


def size_model_path(model_type):
    torch_str = "torch"
    if (model_type == "cyto" or model_type == "nuclei" or 
        model_type == "cyto2" or model_type == "cyto3"):
        if model_type == "cyto3":
            basename = "size_%s.npy" % model_type
        else:
            basename = "size_%s%s_0.npy" % (model_type, torch_str)
        return cache_model_path(basename)
    else:
        if os.path.exists(model_type) and os.path.exists(model_type + "_size.npy"):
            return model_type + "_size.npy"
        else:
            raise FileNotFoundError(f"size model not found ({model_type + '_size.npy'})")            
        

def cache_model_path(basename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_MODEL_URL}/{basename}"
    cached_file = os.fspath(MODEL_DIR.joinpath(basename))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        utils.download_url_to_file(url, cached_file, progress=True)
    return cached_file


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


class Cellpose():
    """Main model which combines SizeModel and CellposeModel.

    Args:
        gpu (bool, optional): Whether or not to use GPU, will check if GPU available. Defaults to False.
        model_type (str, optional): Model type. "cyto"=cytoplasm model; "nuclei"=nucleus model; 
            "cyto2"=cytoplasm model with additional user images; 
            "cyto3"=super-generalist model; Defaults to "cyto3".
        device (torch device, optional): Device used for model running / training. Overrides gpu input. Recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")). Defaults to None.

    Attributes:
        device (torch device): Device used for model running / training.
        gpu (bool): Flag indicating if GPU is used.
        diam_mean (float): Mean diameter for cytoplasm model.
        cp (CellposeModel): CellposeModel instance.
        pretrained_size (str): Pretrained size model path.
        sz (SizeModel): SizeModel instance.

    """

    def __init__(self, gpu=False, model_type="cyto3", nchan=2, device=None,
                 backbone="default"):
        super(Cellpose, self).__init__()

        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(use_torch=True, gpu=gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        self.backbone = backbone

        model_type = "cyto3" if model_type is None else model_type

        self.diam_mean = 30.  #default for any cyto model
        nuclear = "nuclei" in model_type
        if nuclear:
            self.diam_mean = 17.

        if model_type in ["cyto", "nuclei", "cyto2", "cyto3"] and nchan != 2:
            nchan = 2
            models_logger.warning(
                f"cannot set nchan to other value for {model_type} model")
        self.nchan = nchan

        self.cp = CellposeModel(device=self.device, gpu=self.gpu, model_type=model_type,
                                diam_mean=self.diam_mean, nchan=self.nchan,
                                backbone=self.backbone)
        self.cp.model_type = model_type

        # size model not used for bacterial model
        self.pretrained_size = size_model_path(model_type)
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=[0, 0], channel_axis=None, invert=False,
             normalize=True, diameter=30., do_3D=False, **kwargs):
        """Run cellpose size model and mask model and get masks.

        Args:
            x (list or array): List or array of images. Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array.
            batch_size (int, optional): Number of 224x224 patches to run simultaneously on the GPU. Can make smaller or bigger depending on GPU memory usage. Defaults to 8.
            channels (list, optional): List of channels, either of length 2 or of length number of images by 2. First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue). For instance, to segment grayscale images, input [0,0]. To segment images with cells in green and nuclei in blue, input [2,3]. To segment one grayscale image and one image with cells in green and nuclei in blue, input [[0,0], [2,3]]. Defaults to [0,0].
            channel_axis (int, optional): If None, channels dimension is attempted to be automatically determined. Defaults to None.
            invert (bool, optional): Invert image pixel intensity before running network (if True, image is also normalized). Defaults to False.
            normalize (bool, optional): If True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; can also pass dictionary of parameters (see CellposeModel for details). Defaults to True.
            diameter (float, optional): If set to None, then diameter is automatically estimated if size model is loaded. Defaults to 30..
            do_3D (bool, optional): Set to True to run 3D segmentation on 4D image input. Defaults to False.

        Returns:
            tuple containing
                - masks (list of 2D arrays or single 3D array): Labelled image, where 0=no masks; 1,2,...=mask labels.
                - flows (list of lists 2D arrays or list of 3D arrays): 
                    - flows[k][0] = XY flow in HSV 0-255
                    - flows[k][1] = XY flows at each pixel
                    - flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
                    - flows[k][3] = final pixel locations after Euler integration
                - styles (list of 1D arrays of length 256 or single 1D array): Style vector summarizing each image, also used to estimate size of objects in image.
                - diams (list of diameters or float): List of diameters or float (if do_3D=True).

        """

        tic0 = time.time()
        models_logger.info(f"channels set to {channels}")

        diam0 = diameter[0] if isinstance(diameter, (np.ndarray, list)) else diameter
        estimate_size = True if (diameter is None or diam0 == 0) else False

        if estimate_size and self.pretrained_size is not None and not do_3D and x[
                0].ndim < 4:
            tic = time.time()
            models_logger.info("~~~ ESTIMATING CELL DIAMETER(S) ~~~")
            diams, _ = self.sz.eval(x, channels=channels, channel_axis=channel_axis,
                                    batch_size=batch_size, normalize=normalize,
                                    invert=invert)
            diameter = None
            models_logger.info("estimated cell diameter(s) in %0.2f sec" %
                               (time.time() - tic))
            models_logger.info(">>> diameter(s) = ")
            if isinstance(diams, list) or isinstance(diams, np.ndarray):
                diam_string = "[" + "".join(["%0.2f, " % d for d in diams]) + "]"
            else:
                diam_string = "[ %0.2f ]" % diams
            models_logger.info(diam_string)
        elif estimate_size:
            if self.pretrained_size is None:
                reason = "no pretrained size model specified in model Cellpose"
            else:
                reason = "does not work on non-2D images"
            models_logger.warning(f"could not estimate diameter, {reason}")
            diams = self.diam_mean
        else:
            diams = diameter

        models_logger.info("~~~ FINDING MASKS ~~~")
        masks, flows, styles = self.cp.eval(x, channels=channels,
                                            channel_axis=channel_axis,
                                            batch_size=batch_size, normalize=normalize,
                                            invert=invert, diameter=diams, do_3D=do_3D,
                                            **kwargs)
        models_logger.info(">>>> TOTAL TIME %0.2f sec" % (time.time() - tic0))

        return masks, flows, styles, diams

def get_model_params(pretrained_model, model_type, pretrained_model_ortho, default_model="cyto3"):
    """ return pretrained_model path, diam_mean and if model is builtin """
    builtin = False
    use_default = False
    diam_mean = None
    model_strings = get_user_models()
    all_models = MODEL_NAMES.copy()
    all_models.extend(model_strings)
    
    # check if pretrained_model is builtin or custom user model saved in .cellpose/models
    # if yes, then set to model_type
    if (pretrained_model and not Path(pretrained_model).exists() and
            np.any([pretrained_model == s for s in all_models])):
        model_type = pretrained_model
        
    # check if model_type is builtin or custom user model saved in .cellpose/models
    if model_type is not None and np.any([model_type == s for s in all_models]):
        if np.any([model_type == s for s in MODEL_NAMES]):
            builtin = True
        models_logger.info(f">> {model_type} << model set to be used")
        if model_type == "nuclei":
            diam_mean = 17.
        pretrained_model = model_path(model_type)
    # if model_type is not None and does not exist, use default model
    elif model_type is not None:
        if Path(model_type).exists():
            pretrained_model = model_type
        else:
            models_logger.warning("model_type does not exist, using default model")
            use_default = True
    # if model_type is None...
    else:
        # if pretrained_model does not exist, use default model
        if pretrained_model and not Path(pretrained_model).exists():
            models_logger.warning(
                "pretrained_model path does not exist, using default model")
            use_default = True
        elif pretrained_model:
            if pretrained_model[-13:] == "nucleitorch_0":
                builtin = True
                diam_mean = 17.

    if pretrained_model_ortho:
        if pretrained_model_ortho in all_models:
            pretrained_model_ortho = model_path(pretrained_model_ortho)
        elif Path(pretrained_model_ortho).exists():
            pass
        else:
            pretrained_model_ortho = None 

    pretrained_model = model_path(default_model) if use_default else pretrained_model
    builtin = True if use_default else builtin
    return pretrained_model, diam_mean, builtin, pretrained_model_ortho
    

class CellposeModel():
    """
    Class representing a Cellpose model.

    Attributes:
        diam_mean (float): Mean "diameter" value for the model.
        builtin (bool): Whether the model is a built-in model or not.
        device (torch device): Device used for model running / training.
        mkldnn (None or bool): MKLDNN flag for the model.
        nchan (int): Number of channels used as input to the network.
        nclasses (int): Number of classes in the model.
        nbase (list): List of base values for the model.
        net (CPnet): Cellpose network.
        pretrained_model (str): Path to pretrained cellpose model.
        pretrained_model_ortho (str): Path or model_name for pretrained cellpose model for ortho views in 3D.
        backbone (str): Type of network ("default" is the standard res-unet, "transformer" for the segformer).

    Methods:
        __init__(self, gpu=False, pretrained_model=False, model_type=None, diam_mean=30., device=None, nchan=2):
            Initialize the CellposeModel.
        
        eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None, z_axis=None, normalize=True, invert=False, rescale=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None, stitch_threshold=0.0, min_size=15, niter=None, augment=False, tile_overlap=0.1, bsize=224, interp=True, compute_masks=True, progress=None):
            Segment list of images x, or 4D array - Z x nchan x Y x X.

    """

    def __init__(self, gpu=False, pretrained_model=False, model_type=None,
                 mkldnn=True, diam_mean=30., device=None, nchan=2, 
                 pretrained_model_ortho=None, backbone="default"):
        """
        Initialize the CellposeModel.

        Parameters:
            gpu (bool, optional): Whether or not to save model to GPU, will check if GPU available.
            pretrained_model (str or list of strings, optional): Full path to pretrained cellpose model(s), if None or False, no model loaded.
            model_type (str, optional): Any model that is available in the GUI, use name in GUI e.g. "livecell" (can be user-trained or model zoo).
            mkldnn (bool, optional): Use MKLDNN for CPU inference, faster but not always supported.
            diam_mean (float, optional): Mean "diameter", 30. is built-in value for "cyto" model; 17. is built-in value for "nuclei" model; if saved in custom model file (cellpose>=2.0) then it will be loaded automatically and overwrite this value.
            device (torch device, optional): Device used for model running / training (torch.device("cuda") or torch.device("cpu")), overrides gpu input, recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
            nchan (int, optional): Number of channels to use as input to network, default is 2 (cyto + nuclei) or (nuclei + zeros).
        """
        self.diam_mean = diam_mean

        ### set model path
        default_model = "cyto3" if backbone == "default" else "transformer_cp3"
        pretrained_model, diam_mean, builtin, pretrained_model_ortho = get_model_params(
                                                                pretrained_model, 
                                                                model_type, 
                                                                pretrained_model_ortho,
                                                                 default_model)
        self.diam_mean = diam_mean if diam_mean is not None else self.diam_mean
        
        ### assign model device
        self.mkldnn = None
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu
        if not self.gpu:
            self.mkldnn = check_mkl(True) if mkldnn else False

        ### create neural network
        self.nchan = nchan
        self.nclasses = 3
        nbase = [32, 64, 128, 256]
        self.nbase = [nchan, *nbase]
        self.pretrained_model = pretrained_model
        if backbone == "default":
            self.net = CPnet(self.nbase, self.nclasses, sz=3, mkldnn=self.mkldnn,
                             max_pool=True, diam_mean=self.diam_mean).to(self.device)
        else:
            from .segformer import Transformer
            self.net = Transformer(
                encoder_weights="imagenet" if not self.pretrained_model else None,
                diam_mean=self.diam_mean).to(self.device)

        ### load model weights
        if self.pretrained_model:
            models_logger.info(f">>>> loading model {pretrained_model}")
            self.net.load_model(self.pretrained_model, device=self.device)
            if not builtin:
                self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
            models_logger.info(
                f">>>> model diam_mean = {self.diam_mean: .3f} (ROIs rescaled to this size during training)"
            )
            if not builtin:
                models_logger.info(
                    f">>>> model diam_labels = {self.diam_labels: .3f} (mean diameter of training ROIs)"
                )
            if pretrained_model_ortho is not None:
                models_logger.info(f">>>> loading ortho model {pretrained_model_ortho}")
                self.net_ortho = CPnet(self.nbase, self.nclasses, sz=3, mkldnn=self.mkldnn,
                                        max_pool=True, diam_mean=self.diam_mean).to(self.device)
                self.net_ortho.load_model(pretrained_model_ortho, device=self.device)
            else:
                self.net_ortho = None
        else:
            models_logger.info(f">>>> no model weights loaded")
            self.diam_labels = self.diam_mean

        self.net_type = f"cellpose_{backbone}"

    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             dP_smooth=0, stitch_threshold=0.0, 
             min_size=15, max_size_fraction=0.4, niter=None, 
             augment=False, tile_overlap=0.1, bsize=224, 
             interp=True, compute_masks=True, progress=None):
        """ segment list of images x, or 4D array - Z x nchan x Y x X

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images
            batch_size (int, optional): number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 8.
            resample (bool, optional): run dynamics at original image size (will be slower but create more accurate boundaries). Defaults to True.
            channels (list, optional): list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].
                Defaults to None.
            channel_axis (int, optional): channel axis in element of list x, or of np.ndarray x. 
                if None, channels dimension is attempted to be automatically determined. Defaults to None.
            z_axis  (int, optional): z axis in element of list x, or of np.ndarray x. 
                if None, z dimension is attempted to be automatically determined. Defaults to None.
            normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; 
                can also pass dictionary of parameters (all keys are optional, default values shown): 
                    - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
                    - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
                    - "normalize"=True ; run normalization (if False, all following parameters ignored)
                    - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
                    - "tile_norm"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
                    - "norm3D"=False ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
                Defaults to True.
            invert (bool, optional): invert image pixel intensity before running network. Defaults to False.
            rescale (float, optional): resize factor for each image, if None, set to 1.0;
                (only used if diameter is None). Defaults to None.
            diameter (float, optional):  diameter for each image, 
                if diameter is None, set to diam_mean or diam_train if available. Defaults to None.
            flow_threshold (float, optional): flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
            cellprob_threshold (float, optional): all pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
            do_3D (bool, optional): set to True to run 3D segmentation on 3D/4D image input. Defaults to False.
            dP_smooth (int, optional): if do_3D and dP_smooth>0, smooth flows with gaussian filter of this stddev. Defaults to 0.
            anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
            stitch_threshold (float, optional): if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation. Defaults to 0.0.
            min_size (int, optional): all ROIs below this size, in pixels, will be discarded. Defaults to 15.
            max_size_fraction (float, optional): max_size_fraction (float, optional): Masks larger than max_size_fraction of
                total image size are removed. Default is 0.4.
            niter (int, optional): number of iterations for dynamics computation. if None, it is set proportional to the diameter. Defaults to None.
            augment (bool, optional): tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
            tile_overlap (float, optional): fraction of overlap of tiles when computing flows. Defaults to 0.1.
            bsize (int, optional): block size for tiles, recommended to keep at 224, like in training. Defaults to 224.
            interp (bool, optional): interpolate during 2D dynamics (not available in 3D) . Defaults to True.
            compute_masks (bool, optional): Whether or not to compute dynamics and return masks. This is set to False when retrieving the styles for the size model. Defaults to True.
            progress (QProgressBar, optional): pyqt progress bar. Defaults to None.

        Returns:
            A tuple containing:
                - masks (list, np.ndarray): labelled image(s), where 0=no masks; 1,2,...=mask labels
                - flows (list): list of lists: flows[k][0] = XY flow in HSV 0-255; flows[k][1] = XY(Z) flows at each pixel; flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics); flows[k][3] = final pixel locations after Euler integration 
                - styles (list, np.ndarray): style vector summarizing each image of size 256.
            
        """
        if isinstance(x, list) or x.squeeze().ndim == 5:
            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i], batch_size=batch_size,
                    channels=channels[i] if channels is not None and
                    ((len(channels) == len(x) and
                      (isinstance(channels[i], list) or
                       isinstance(channels[i], np.ndarray)) and len(channels[i]) == 2))
                    else channels, channel_axis=channel_axis, z_axis=z_axis,
                    normalize=normalize, invert=invert,
                    rescale=rescale[i] if isinstance(rescale, list) or
                    isinstance(rescale, np.ndarray) else rescale,
                    diameter=diameter[i] if isinstance(diameter, list) or
                    isinstance(diameter, np.ndarray) else diameter, do_3D=do_3D,
                    anisotropy=anisotropy, augment=augment, 
                    tile_overlap=tile_overlap, bsize=bsize, resample=resample,
                    interp=interp, flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, compute_masks=compute_masks,
                    min_size=min_size, max_size_fraction=max_size_fraction, 
                    stitch_threshold=stitch_threshold, dP_smooth=dP_smooth,
                    progress=progress, niter=niter)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        else:
            # reshape image
            x = transforms.convert_image(x, channels, channel_axis=channel_axis,
                                         z_axis=z_axis, do_3D=(do_3D or
                                                               stitch_threshold > 0),
                                         nchan=self.nchan)
            if x.ndim < 4:
                x = x[np.newaxis, ...]
            nimg = x.shape[0]
            
            if diameter is not None and diameter > 0:
                rescale = self.diam_mean / diameter
            elif rescale is None:
                rescale = self.diam_mean / self.diam_labels

            # normalize image
            normalize_params = normalize_default
            if isinstance(normalize, dict):
                normalize_params = {**normalize_params, **normalize}
            elif not isinstance(normalize, bool):
                raise ValueError("normalize parameter must be a bool or a dict")
            else:
                normalize_params["normalize"] = normalize
                normalize_params["invert"] = invert

            # pre-normalize if 3D stack for stitching or do_3D
            do_normalization = True if normalize_params["normalize"] else False
            x = np.asarray(x)
            if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
                normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
                x = transforms.normalize_img(x, **normalize_params)
                do_normalization = False # do not normalize again
            else:
                if normalize_params["norm3D"] and nimg > 1:
                    models_logger.warning(
                        "normalize_params['norm3D'] is True but do_3D is False and stitch_threshold=0, so setting to False"
                    )
                    normalize_params["norm3D"] = False
            if do_normalization:
                x = transforms.normalize_img(x, **normalize_params)

            dP, cellprob, styles = self._run_net(
                x, rescale=rescale, augment=augment, 
                batch_size=batch_size, tile_overlap=tile_overlap, bsize=bsize,
                resample=resample, do_3D=do_3D, anisotropy=anisotropy)

            if do_3D:    
                if dP_smooth > 0:
                    models_logger.info(f"smoothing flows with sigma={dP_smooth}")
                    dP = gaussian_filter(dP, (0, dP_smooth, dP_smooth, dP_smooth))
                torch.cuda.empty_cache()
                gc.collect()

            if compute_masks:
                niter0 = 200 if not resample else (1 / rescale * 200)
                niter = niter0 if niter is None or niter == 0 else niter
                masks = self._compute_masks(x.shape, dP, cellprob, flow_threshold=flow_threshold,
                               cellprob_threshold=cellprob_threshold, interp=interp, min_size=min_size,
                            max_size_fraction=max_size_fraction, niter=niter,
                            stitch_threshold=stitch_threshold, do_3D=do_3D)
            else:
                masks = np.zeros(0) #pass back zeros if not compute_masks
            
            masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()

            return masks, [plot.dx_to_circ(dP), dP, cellprob], styles

    def _run_net(self, x, rescale=1.0, resample=True, augment=False, 
                batch_size=8, tile_overlap=0.1,
                bsize=224, anisotropy=1.0, do_3D=False):
        """ run network on image x """
        tic = time.time()
        shape = x.shape
        nimg = shape[0]

        if do_3D:
            Lz, Ly, Lx = shape[:-1]
            if rescale != 1.0 or (anisotropy is not None and anisotropy != 1.0):
                models_logger.info(f"resizing 3D image with rescale={rescale:.2f} and anisotropy={anisotropy}")
                anisotropy = 1.0 if anisotropy is None else anisotropy
                if rescale != 1.0:
                    x = transforms.resize_image(x, Ly=int(Ly*rescale), 
                                                  Lx=int(Lx*rescale))
                x = transforms.resize_image(x.transpose(1,0,2,3),
                                        Ly=int(Lz*anisotropy*rescale), 
                                        Lx=int(Lx*rescale)).transpose(1,0,2,3)
            yf, styles = run_3D(self.net, x,
                                batch_size=batch_size, augment=augment,  
                                tile_overlap=tile_overlap, net_ortho=self.net_ortho)
            if resample:
                if rescale != 1.0 or Lz != yf.shape[0]:
                    models_logger.info("resizing 3D flows and cellprob to original image size")
                    if rescale != 1.0:
                        yf = transforms.resize_image(yf, Ly=Ly, Lx=Lx)
                    if Lz != yf.shape[0]:
                        yf = transforms.resize_image(yf.transpose(1,0,2,3),
                                            Ly=Lz, Lx=Lx).transpose(1,0,2,3)
            cellprob = yf[..., -1]
            dP = yf[..., :-1].transpose((3, 0, 1, 2))
        else:
            yf, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                                batch_size=batch_size,  
                                tile_overlap=tile_overlap, 
                                rsz=rescale if rescale!=1.0 else None)
            if resample:
                if rescale != 1.0:
                    yf = transforms.resize_image(yf, shape[1], shape[2])
            cellprob = yf[..., 2]
            dP = yf[..., :2].transpose((3, 0, 1, 2))
        
        styles = styles.squeeze()

        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info("network run in %2.2fs" % (net_time))

        return dP, cellprob, styles
    
    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       interp=True, min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
        """ compute masks from flows and cell probability """
        Lz, Ly, Lx = shape[:3]
        tic = time.time()
        if do_3D:
            masks = dynamics.resize_and_compute_masks(
                dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold, interp=interp, do_3D=do_3D,
                min_size=min_size, max_size_fraction=max_size_fraction, 
                resize=shape[:3] if (np.array(dP.shape[-3:])!=np.array(shape[:3])).sum() 
                        else None,
                device=self.device)
        else:
            nimg = shape[0]
            Ly0, Lx0 = cellprob[0].shape 
            resize = None if Ly0==Ly and Lx0==Lx else [Ly, Lx]
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                            mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                # turn off min_size for 3D stitching
                min_size0 = min_size if stitch_threshold == 0 or nimg == 1 else -1
                outputs = dynamics.resize_and_compute_masks(
                    dP[:, i], cellprob[i],
                    niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, interp=interp, resize=resize,
                    min_size=min_size0, max_size_fraction=max_size_fraction,
                    device=self.device)
                if i==0 and nimg > 1:
                    masks = np.zeros((nimg, shape[1], shape[2]), outputs.dtype)
                if nimg > 1:
                    masks[i] = outputs
                else:
                    masks = outputs

            if stitch_threshold > 0 and nimg > 1:
                models_logger.info(
                    f"stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks"
                )
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                masks = utils.fill_holes_and_remove_small_masks(
                    masks, min_size=min_size)
            elif nimg > 1:
                models_logger.warning(
                    "3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only"
                )

        flow_time = time.time() - tic
        if shape[0] > 1:
            models_logger.info("masks created in %2.2fs" % (flow_time))
        
        return masks

class SizeModel():
    """ 
    Linear regression model for determining the size of objects in image
    used to rescale before input to cp_model.
    Uses styles from cp_model.

    Attributes:
        pretrained_size (str): Path to pretrained size model.
        cp (UnetModel or CellposeModel): Model from which to get styles.
        device (torch device): Device used for model running / training 
            (torch.device("cuda") or torch.device("cpu")), overrides gpu input,
            recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
        diam_mean (float): Mean diameter of objects.
        
    Methods:
        eval(self, x, channels=None, channel_axis=None, normalize=True, invert=False,
             augment=False, batch_size=8, progress=None, interp=True):
            Use images x to produce style or use style input to predict size of objects in image.

    Raises:
        ValueError: If no pretrained cellpose model is specified, cannot compute size.
    """

    def __init__(self, cp_model, device=None, pretrained_size=None, **kwargs):
        super(SizeModel, self).__init__(**kwargs)
        """ 
        Initialize size model.

        Args:
            cp_model (UnetModel or CellposeModel): Model from which to get styles.
            device (torch device, optional): Device used for model running / training 
                (torch.device("cuda") or torch.device("cpu")), overrides gpu input,
                recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
            pretrained_size (str): Path to pretrained size model.
        """

        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.device = self.cp.device
        self.diam_mean = self.cp.diam_mean
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params["diam_mean"]
        if not hasattr(self.cp, "pretrained_model"):
            error_message = "no pretrained cellpose model specified, cannot compute size"
            models_logger.critical(error_message)
            raise ValueError(error_message)

    def eval(self, x, channels=None, channel_axis=None, normalize=True, invert=False,
             augment=False, batch_size=8, progress=None):
        """Use images x to produce style or use style input to predict size of objects in image.

        Object size estimation is done in two steps:
        1. Use a linear regression model to predict size from style in image.
        2. Resize image to predicted size and run CellposeModel to get output masks.
           Take the median object size of the predicted masks as the final predicted size.

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images
            channels (list, optional): list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].
                Defaults to None.
            channel_axis (int, optional): channel axis in element of list x, or of np.ndarray x. 
                if None, channels dimension is attempted to be automatically determined. Defaults to None.
            normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; 
                can also pass dictionary of parameters (all keys are optional, default values shown): 
                    - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
                    - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
                    - "normalize"=True ; run normalization (if False, all following parameters ignored)
                    - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
                    - "tile_norm"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
                    - "norm3D"=False ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
                Defaults to True.
            invert (bool, optional): Invert image pixel intensity before running network (if True, image is also normalized). Defaults to False.
            augment (bool, optional): tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
            batch_size (int, optional): number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 8.
            progress (QProgressBar, optional): pyqt progress bar. Defaults to None.


        Returns:
            A tuple containing:
                - diam (np.ndarray): Final estimated diameters from images x or styles style after running both steps.
                - diam_style (np.ndarray): Estimated diameters from style alone.
        """
        if isinstance(x, list):
            self.timing = []
            diams, diams_style = [], []
            nimg = len(x)
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                diam, diam_style = self.eval(
                    x[i], channels=channels[i] if
                    (channels is not None and len(channels) == len(x) and
                     (isinstance(channels[i], list) or
                      isinstance(channels[i], np.ndarray)) and
                     len(channels[i]) == 2) else channels, channel_axis=channel_axis,
                    normalize=normalize, invert=invert, augment=augment, 
                    batch_size=batch_size, progress=progress)
                diams.append(diam)
                diams_style.append(diam_style)
                self.timing.append(time.time() - tic)

            return diams, diams_style

        if x.squeeze().ndim > 3:
            models_logger.warning("image is not 2D cannot compute diameter")
            return self.diam_mean, self.diam_mean

        styles = self.cp.eval(x, channels=channels, channel_axis=channel_axis,
                              normalize=normalize, invert=invert, augment=augment,
                               batch_size=batch_size, resample=False,
                              compute_masks=False)[-1]

        diam_style = self._size_estimation(np.array(styles))
        diam_style = self.diam_mean if (diam_style == 0 or
                                        np.isnan(diam_style)) else diam_style

        masks = self.cp.eval(
            x, compute_masks=True, channels=channels, channel_axis=channel_axis,
            normalize=normalize, invert=invert, augment=augment, 
            batch_size=batch_size, resample=False,
            rescale=self.diam_mean / diam_style if self.diam_mean > 0 else 1,
            diameter=None, interp=False)[0]

        diam = utils.diameters(masks)[0]
        diam = self.diam_mean if (diam == 0 or np.isnan(diam)) else diam
        return diam, diam_style

    def _size_estimation(self, style):
        """ linear regression from style to size 
        
            sizes were estimated using "diameters" from square estimates not circles; 
            therefore a conversion factor is included (to be removed)
        
        """
        szest = np.exp(self.params["A"] @ (style - self.params["smean"]).T +
                       np.log(self.diam_mean) + self.params["ymean"])
        szest = np.maximum(5., szest)
        return szest
