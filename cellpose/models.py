"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import os, time
from pathlib import Path
import numpy as np
from tqdm import trange
import torch
from scipy.ndimage import gaussian_filter
import gc
import cv2

import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .vit_sam import Transformer
from .core import assign_device, run_net, run_3D

_CPSAM_MODEL_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home().joinpath(".cellpose", "models")
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = ["cpsam"]

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
    return cache_CPSAM_model_path()


def cache_CPSAM_model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = os.fspath(MODEL_DIR.joinpath('cpsam'))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(_CPSAM_MODEL_URL, cached_file))
        utils.download_url_to_file(_CPSAM_MODEL_URL, cached_file, progress=True)
    return cached_file


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


class CellposeModel():
    """
    Class representing a Cellpose model.

    Attributes:
        diam_mean (float): Mean "diameter" value for the model.
        builtin (bool): Whether the model is a built-in model or not.
        device (torch device): Device used for model running / training.
        nclasses (int): Number of classes in the model.
        nbase (list): List of base values for the model.
        net (CPnet): Cellpose network.
        pretrained_model (str): Path to pretrained cellpose model.
        pretrained_model_ortho (str): Path or model_name for pretrained cellpose model for ortho views in 3D.
        backbone (str): Type of network ("default" is the standard res-unet, "transformer" for the segformer).

    Methods:
        __init__(self, gpu=False, pretrained_model=False, model_type=None, diam_mean=30., device=None):
            Initialize the CellposeModel.
        
        eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None, z_axis=None, normalize=True, invert=False, rescale=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None, stitch_threshold=0.0, min_size=15, niter=None, augment=False, tile_overlap=0.1, bsize=224, interp=True, compute_masks=True, progress=None):
            Segment list of images x, or 4D array - Z x C x Y x X.

    """

    def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                 diam_mean=None, device=None, nchan=None):
        """
        Initialize the CellposeModel.

        Parameters:
            gpu (bool, optional): Whether or not to save model to GPU, will check if GPU available.
            pretrained_model (str or list of strings, optional): Full path to pretrained cellpose model(s), if None or False, no model loaded.
            model_type (str, optional): Any model that is available in the GUI, use name in GUI e.g. "livecell" (can be user-trained or model zoo).
            diam_mean (float, optional): Mean "diameter", 30. is built-in value for "cyto" model; 17. is built-in value for "nuclei" model; if saved in custom model file (cellpose>=2.0) then it will be loaded automatically and overwrite this value.
            device (torch device, optional): Device used for model running / training (torch.device("cuda") or torch.device("cpu")), overrides gpu input, recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
        """
        if diam_mean is not None:
            models_logger.warning(
                "diam_mean argument are not used in v4.0.1+. Ignoring this argument..."
            )
        if model_type is not None:
            models_logger.warning(
                "model_type argument is not used in v4.0.1+. Ignoring this argument..."
            )
        if nchan is not None:
            models_logger.warning("nchan argument is deprecated in v4.0.1+. Ignoring this argument")

        ### assign model device
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu

        if pretrained_model is None:
            raise ValueError("Must specify a pretrained model, training from scratch is not implemented")
        
        ### create neural network
        if pretrained_model and not os.path.exists(pretrained_model):
            # check if pretrained model is in the models directory
            model_strings = get_user_models()
            all_models = MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if pretrained_model in all_models:
                pretrained_model = os.path.join(MODEL_DIR, pretrained_model)
            else:
                pretrained_model = os.path.join(MODEL_DIR, "cpsam")
                models_logger.warning(
                    f"pretrained model {pretrained_model} not found, using default model"
                )

        self.pretrained_model = pretrained_model
        self.net = Transformer().to(self.device)

        if os.path.exists(self.pretrained_model):
            models_logger.info(f">>>> loading model {self.pretrained_model}")
            self.net.load_model(self.pretrained_model, device=self.device)
        else:
            if os.path.split(self.pretrained_model)[-1] != 'cpsam':
                raise FileNotFoundError('model file not recognized')
            cache_CPSAM_model_path()
            self.net.load_model(self.pretrained_model, device=self.device)
        
        
    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, 
             min_size=15, max_size_fraction=0.4, niter=None, 
             augment=False, tile_overlap=0.1, bsize=256, 
             compute_masks=True, progress=None):
        """ segment list of images x, or 4D array - Z x 3 x Y x X

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images. Images must have 3 channels.
            batch_size (int, optional): number of 256x256 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 64.
            resample (bool, optional): run dynamics at original image size (will be slower but create more accurate boundaries). 
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
                    - "tile_norm_blocksize"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
                    - "norm3D"=True ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
                Defaults to True.
            invert (bool, optional): invert image pixel intensity before running network. Defaults to False.
            rescale (float, optional): resize factor for each image, if None, set to 1.0;
                (only used if diameter is None). Defaults to None.
            diameter (float or list of float, optional): diameters are used to rescale the image to 30 pix cell diameter.
            flow_threshold (float, optional): flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
            cellprob_threshold (float, optional): all pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
            do_3D (bool, optional): set to True to run 3D segmentation on 3D/4D image input. Defaults to False.
            flow3D_smooth (int, optional): if do_3D and flow3D_smooth>0, smooth flows with gaussian filter of this stddev. Defaults to 0.
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
            compute_masks (bool, optional): Whether or not to compute dynamics and return masks. Returns empty array if False. Defaults to True.
            progress (QProgressBar, optional): pyqt progress bar. Defaults to None.

        Returns:
            A tuple containing (masks, flows, styles, diams): 
            masks (list of 2D arrays or single 3D array): Labelled image, where 0=no masks; 1,2,...=mask labels;
            flows (list of lists 2D arrays or list of 3D arrays): 
                flows[k][0] = XY flow in HSV 0-255; 
                flows[k][1] = XY flows at each pixel; 
                flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics); 
                flows[k][3] = final pixel locations after Euler integration; 
            styles (list of 1D arrays of length 256 or single 1D array): Style vector summarizing each image, also used to estimate size of objects in image.
            
        """

        if rescale is not None:
            models_logger.warning("rescaling deprecated in v4.0.1+") 
        if channels is not None:
            models_logger.warning("channels deprecated in v4.0.1+. If data contain more than 3 channels, only the first 3 channels will be used")

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
                    x[i], 
                    batch_size=batch_size,
                    channel_axis=channel_axis, 
                    z_axis=z_axis,
                    normalize=normalize, 
                    invert=invert,
                    diameter=diameter[i] if isinstance(diameter, list) or
                        isinstance(diameter, np.ndarray) else diameter, 
                    do_3D=do_3D,
                    anisotropy=anisotropy, 
                    augment=augment, 
                    tile_overlap=tile_overlap, 
                    bsize=bsize, 
                    resample=resample,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, 
                    compute_masks=compute_masks,
                    min_size=min_size, 
                    max_size_fraction=max_size_fraction, 
                    stitch_threshold=stitch_threshold, 
                    flow3D_smooth=flow3D_smooth,
                    progress=progress, 
                    niter=niter)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        ############# actual eval code ############
        # reshape image
        x = transforms.convert_image(x, channel_axis=channel_axis,
                                        z_axis=z_axis, 
                                        do_3D=(do_3D or stitch_threshold > 0))
        
        # Add batch dimension if not present
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        nimg = x.shape[0]
        
        image_scaling = None
        Ly_0 = x.shape[1]
        Lx_0 = x.shape[2]
        Lz_0 = None
        if do_3D or stitch_threshold > 0:
            Lz_0 = x.shape[0]
        if diameter is not None:
            image_scaling = 30. / diameter
            x = transforms.resize_image(x,
                                        Ly=int(x.shape[1] * image_scaling),
                                        Lx=int(x.shape[2] * image_scaling))


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
        if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
            normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
            x = transforms.normalize_img(x, **normalize_params)
            do_normalization = False # do not normalize again
        else:
            if normalize_params["norm3D"] and nimg > 1 and do_normalization:
                models_logger.warning(
                    "normalize_params['norm3D'] is True but do_3D is False and stitch_threshold=0, so setting to False"
                )
                normalize_params["norm3D"] = False
        if do_normalization:
            x = transforms.normalize_img(x, **normalize_params)

        # ajust the anisotropy when diameter is specified and images are resized:
        if isinstance(anisotropy, (float, int)) and image_scaling:
            anisotropy = image_scaling * anisotropy

        dP, cellprob, styles = self._run_net(
            x, 
            augment=augment, 
            batch_size=batch_size, 
            tile_overlap=tile_overlap, 
            bsize=bsize,
            do_3D=do_3D, 
            anisotropy=anisotropy)

        if do_3D:    
            if flow3D_smooth > 0:
                models_logger.info(f"smoothing flows with sigma={flow3D_smooth}")
                dP = gaussian_filter(dP, (0, flow3D_smooth, flow3D_smooth, flow3D_smooth))
            torch.cuda.empty_cache()
            gc.collect()

        if resample:
            # upsample flows before computing them: 
            dP = self._resize_gradients(dP, to_y_size=Ly_0, to_x_size=Lx_0, to_z_size=Lz_0)
            cellprob = self._resize_cellprob(cellprob, to_x_size=Lx_0, to_y_size=Ly_0, to_z_size=Lz_0)


        if compute_masks:
            niter0 = 200
            niter = niter0 if niter is None or niter == 0 else niter
            masks = self._compute_masks(x.shape, dP, cellprob, flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold, min_size=min_size,
                        max_size_fraction=max_size_fraction, niter=niter,
                        stitch_threshold=stitch_threshold, do_3D=do_3D)
        else:
            masks = np.zeros(0) #pass back zeros if not compute_masks
        
        masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()

        # undo resizing:
        if image_scaling is not None or anisotropy is not None:

            dP = self._resize_gradients(dP, to_y_size=Ly_0, to_x_size=Lx_0, to_z_size=Lz_0) # works for 2 or 3D: 
            cellprob = self._resize_cellprob(cellprob, to_x_size=Lx_0, to_y_size=Ly_0, to_z_size=Lz_0)

            if do_3D:
                if compute_masks:
                    # Rescale xy then xz:
                    masks = transforms.resize_image(masks, Ly=Ly_0, Lx=Lx_0, no_channels=True, interpolation=cv2.INTER_NEAREST)
                    masks = masks.transpose(1, 0, 2)
                    masks = transforms.resize_image(masks, Ly=Lz_0, Lx=Lx_0, no_channels=True, interpolation=cv2.INTER_NEAREST)
                    masks = masks.transpose(1, 0, 2)

            else:
                # 2D or 3D stitching case:
                if compute_masks:
                    masks = transforms.resize_image(masks, Ly=Ly_0, Lx=Lx_0, no_channels=True, interpolation=cv2.INTER_NEAREST)

        return masks, [plot.dx_to_circ(dP), dP, cellprob], styles
    

    def _resize_cellprob(self, prob: np.ndarray, to_y_size: int, to_x_size: int, to_z_size: int = None) -> np.ndarray:
        """
        Resize cellprob array to specified dimensions for either 2D or 3D.

        Parameters:
            prob (numpy.ndarray): The cellprobs to resize, either in 2D or 3D. Returns the same ndim as provided.
            to_y_size (int): The target size along the Y-axis.
            to_x_size (int): The target size along the X-axis.
            to_z_size (int, optional): The target size along the Z-axis. Required
                for 3D cellprobs.

        Returns:
            numpy.ndarray: The resized cellprobs array with the same number of dimensions
            as the input.

        Raises:
            ValueError: If the input cellprobs array does not have 3 or 4 dimensions.
        """
        prob_shape = prob.shape
        prob = prob.squeeze()
        squeeze_happened = prob.shape != prob_shape
        prob_shape = np.array(prob_shape)

        if prob.ndim == 2:
            # 2D case:
            prob = transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
            if squeeze_happened:
                prob = np.expand_dims(prob, int(np.argwhere(prob_shape == 1))) # add back empty axis for compatibility
        elif prob.ndim == 3:
            # 3D case: 
            prob = transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
            prob = transforms.resize_image(prob, Ly=to_z_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
        else:
            raise ValueError(f'gradients have incorrect dimension after squeezing. Should be 2 or 3, prob shape: {prob.shape}')
        
        return prob


    def _resize_gradients(self, grads: np.ndarray, to_y_size: int, to_x_size: int, to_z_size: int = None) -> np.ndarray:
        """
        Resize gradient arrays to specified dimensions for either 2D or 3D gradients.

        Parameters:
            grads (np.ndarray): The gradients to resize, either in 2D or 3D. Returns the same ndim as provided.
            to_y_size (int): The target size along the Y-axis.
            to_x_size (int): The target size along the X-axis.
            to_z_size (int, optional): The target size along the Z-axis. Required
                for 3D gradients.

        Returns:
            numpy.ndarray: The resized gradient array with the same number of dimensions
            as the input.

        Raises:
            ValueError: If the input gradient array does not have 3 or 4 dimensions.
        """
        grads_shape = grads.shape
        grads = grads.squeeze()
        squeeze_happened = grads.shape != grads_shape
        grads_shape = np.array(grads_shape)

        if grads.ndim == 3:
            # 2D case, with XY flows in 2 channels:
            grads = np.moveaxis(grads, 0, -1) # Put gradients last
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = np.moveaxis(grads, -1, 0) # Put gradients first

            if squeeze_happened:
                grads = np.expand_dims(grads, int(np.argwhere(grads_shape == 1))) # add back empty axis for compatibility
        elif grads.ndim == 4:
            # dP has gradients that can be treated as channels:
            grads = grads.transpose(1, 2, 3, 0) # move gradients last:
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(1, 0, 2, 3) # switch axes to resize again
            grads = transforms.resize_image(grads, Ly=to_z_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(3, 1, 0, 2) # undo transposition
        else:
            raise ValueError(f'gradients have incorrect dimension after squeezing. Should be 3 or 4, grads shape: {grads.shape}')
        
        return grads


    def _run_net(self, x, 
                augment=False, 
                batch_size=8, tile_overlap=0.1,
                bsize=224, anisotropy=1.0, do_3D=False):
        """ run network on image x """
        tic = time.time()
        shape = x.shape
        nimg = shape[0]


        if do_3D:
            Lz, Ly, Lx = shape[:-1]
            if anisotropy is not None and anisotropy != 1.0:
                models_logger.info(f"resizing 3D image with anisotropy={anisotropy}")
                x = transforms.resize_image(x.transpose(1,0,2,3),
                                        Ly=int(Lz*anisotropy), 
                                        Lx=int(Lx)).transpose(1,0,2,3)
            yf, styles = run_3D(self.net, x,
                                batch_size=batch_size, augment=augment,  
                                tile_overlap=tile_overlap, 
                                bsize=bsize
                                )
            cellprob = yf[..., -1]
            dP = yf[..., :-1].transpose((3, 0, 1, 2))
        else:
            yf, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                                batch_size=batch_size,  
                                tile_overlap=tile_overlap, 
                                )
            cellprob = yf[..., -1]
            dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
            if yf.shape[-1] > 3:
                styles = yf[..., :-3]
        
        styles = styles.squeeze()

        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info("network run in %2.2fs" % (net_time))

        return dP, cellprob, styles
    
    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
        """ compute masks from flows and cell probability """
        changed_device_from = None
        if self.device.type == "mps" and do_3D:
            models_logger.warning("MPS does not support 3D post-processing, switching to CPU")
            self.device = torch.device("cpu")
            changed_device_from = "mps"
        Lz, Ly, Lx = shape[:3]
        tic = time.time()
        if do_3D:
            masks = dynamics.resize_and_compute_masks(
                dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold, do_3D=do_3D,
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
                    flow_threshold=flow_threshold, resize=resize,
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
        
        if changed_device_from is not None:
            models_logger.info("switching back to device %s" % self.device)
            self.device = torch.device(changed_device_from)
        return masks
