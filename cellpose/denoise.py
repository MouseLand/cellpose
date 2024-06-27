"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, time, datetime
import numpy as np
from scipy.stats import mode
import cv2
import torch
from torch import nn
from torch.nn.functional import conv2d, interpolate
from tqdm import trange
from pathlib import Path

import logging

denoise_logger = logging.getLogger(__name__)

from cellpose import transforms, resnet_torch, utils, io
from cellpose.core import run_net
from cellpose.resnet_torch import CPnet
from cellpose.models import CellposeModel, model_path, normalize_default, assign_device, check_mkl

MODEL_NAMES = []
for ctype in ["cyto3", "cyto2", "nuclei"]:
    for ntype in ["denoise", "deblur", "upsample"]:
        MODEL_NAMES.append(f"{ntype}_{ctype}")
        if ctype != "cyto3":
            for ltype in ["per", "seg", "rec"]:
                MODEL_NAMES.append(f"{ntype}_{ltype}_{ctype}")

criterion = nn.MSELoss(reduction="mean")
criterion2 = nn.BCEWithLogitsLoss(reduction="mean")


def deterministic(seed=0):
    """ set random seeds to create test data """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loss_fn_rec(lbl, y):
    """ loss function between true labels lbl and prediction y """
    loss = 80. * criterion(y, lbl)
    return loss


def loss_fn_seg(lbl, y):
    """ loss function between true labels lbl and prediction y """
    veci = 5. * lbl[:, 1:]
    lbl = (lbl[:, 0] > .5).float()
    loss = criterion(y[:, :2], veci)
    loss /= 2.
    loss2 = criterion2(y[:, 2], lbl)
    loss = loss + loss2
    return loss


def get_sigma(Tdown):
    """ Calculates the correlation matrices across channels for the perceptual loss.

    Args:
        Tdown (list): List of tensors output by each downsampling block of network.

    Returns:
        list: List of correlations for each input tensor.
    """
    Tnorm = [x - x.mean((-2, -1), keepdim=True) for x in Tdown]
    Tnorm = [x / x.std((-2, -1), keepdim=True) for x in Tnorm]
    Sigma = [
        torch.einsum("bnxy, bmxy -> bnm", x, x) / (x.shape[-2] * x.shape[-1])
        for x in Tnorm
    ]
    return Sigma


def imstats(X, net1):
    """
    Calculates the image correlation matrices for the perceptual loss.

    Args:
        X (torch.Tensor): Input image tensor.
        net1: Cellpose net.

    Returns:
        list: A list of tensors of correlation matrices.
    """
    _, _, Tdown = net1(X)
    Sigma = get_sigma(Tdown)
    Sigma = [x.detach() for x in Sigma]
    return Sigma


def loss_fn_per(img, net1, yl):
    """
    Calculates the perceptual loss function for image restoration.

    Args:
        img (torch.Tensor): Input image tensor (noisy/blurry/downsampled).
        net1 (torch.nn.Module): Perceptual loss net (Cellpose segmentation net).
        yl (torch.Tensor): Clean image tensor.

    Returns:
        torch.Tensor: Mean perceptual loss.
    """
    Sigma = imstats(img, net1)
    sd = [x.std((1, 2)) + 1e-6 for x in Sigma]
    Sigma_test = get_sigma(yl)
    losses = torch.zeros(len(Sigma[0]), device=img.device)
    for k in range(len(Sigma)):
        losses = losses + (((Sigma_test[k] - Sigma[k])**2).mean((1, 2)) / sd[k]**2)
    return losses.mean()


def test_loss(net0, X, net1=None, img=None, lbl=None, lam=[1., 1.5, 0.]):
    """
    Calculates the test loss for image restoration tasks.

    Args:
        net0 (torch.nn.Module): The image restoration network.
        X (torch.Tensor): The input image tensor.
        net1 (torch.nn.Module, optional): The segmentation network for segmentation or perceptual loss. Defaults to None.
        img (torch.Tensor, optional): Clean image tensor for perceptual or reconstruction loss. Defaults to None.
        lbl (torch.Tensor, optional): The ground truth flows/cellprob tensor for segmentation loss. Defaults to None.
        lam (list, optional): The weights for different loss components (perceptual, segmentation, reconstruction). Defaults to [1., 1.5, 0.].

    Returns:
        tuple: A tuple containing the total loss and the perceptual loss.
    """
    net0.eval()
    if net1 is not None:
        net1.eval()
    loss, loss_per = torch.zeros(1, device=X.device), torch.zeros(1, device=X.device)

    with torch.no_grad():
        img_dn = net0(X)[0]
        if lam[2] > 0.:
            loss += lam[2] * loss_fn_rec(img, img_dn)
        if lam[1] > 0. or lam[0] > 0.:
            y, _, ydown = net1(img_dn)
        if lam[1] > 0.:
            loss += lam[1] * loss_fn_seg(lbl, y)
        if lam[0] > 0.:
            loss_per = loss_fn_per(img, net1, ydown)
            loss += lam[0] * loss_per
    return loss, loss_per


def train_loss(net0, X, net1=None, img=None, lbl=None, lam=[1., 1.5, 0.]):
    """
    Calculates the train loss for image restoration tasks.

    Args:
        net0 (torch.nn.Module): The image restoration network.
        X (torch.Tensor): The input image tensor.
        net1 (torch.nn.Module, optional): The segmentation network for segmentation or perceptual loss. Defaults to None.
        img (torch.Tensor, optional): Clean image tensor for perceptual or reconstruction loss. Defaults to None.
        lbl (torch.Tensor, optional): The ground truth flows/cellprob tensor for segmentation loss. Defaults to None.
        lam (list, optional): The weights for different loss components (perceptual, segmentation, reconstruction). Defaults to [1., 1.5, 0.].

    Returns:
        tuple: A tuple containing the total loss and the perceptual loss.
    """
    net0.train()
    if net1 is not None:
        net1.eval()
    loss, loss_per = torch.zeros(1, device=X.device), torch.zeros(1, device=X.device)

    img_dn = net0(X)[0]
    if lam[2] > 0.:
        loss += lam[2] * loss_fn_rec(img, img_dn)
    if lam[1] > 0. or lam[0] > 0.:
        y, _, ydown = net1(img_dn)
    if lam[1] > 0.:
        loss += lam[1] * loss_fn_seg(lbl, y)
    if lam[0] > 0.:
        loss_per = loss_fn_per(img, net1, ydown)
        loss += lam[0] * loss_per
    return loss, loss_per


def img_norm(imgi):
    """
    Normalizes the input image by subtracting the 1st percentile and dividing by the difference between the 99th and 1st percentiles.

    Args:
        imgi (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    shape = imgi.shape
    imgi = imgi.reshape(imgi.shape[0], imgi.shape[1], -1)
    perc = torch.quantile(imgi, torch.tensor([0.01, 0.99], device=imgi.device), dim=-1,
                          keepdim=True)
    for k in range(imgi.shape[1]):
        hask = (perc[1, :, k, 0] - perc[0, :, k, 0]) > 1e-3
        imgi[hask, k] -= perc[0, hask, k]
        imgi[hask, k] /= (perc[1, hask, k] - perc[0, hask, k])
    imgi = imgi.reshape(shape)
    return imgi


def add_noise(lbl, alpha=4, beta=0.7, poisson=0.7, blur=0.7, gblur=1.0, downsample=0.7,
              ds_max=7, diams=None, pscale=None, iso=True, sigma0=None, sigma1=None,
              ds=None):
    """Adds noise to the input image.

    Args:
        lbl (torch.Tensor): The input image tensor of shape (nimg, nchan, Ly, Lx).
        alpha (float, optional): The shape parameter of the gamma distribution used for generating poisson noise. Defaults to 4.
        beta (float, optional): The rate parameter of the gamma distribution used for generating poisson noise. Defaults to 0.7.
        poisson (float, optional): The probability of adding poisson noise to the image. Defaults to 0.7.
        blur (float, optional): The probability of adding gaussian blur to the image. Defaults to 0.7.
        gblur (float, optional): The scale factor for the gaussian blur. Defaults to 1.0.
        downsample (float, optional): The probability of downsampling the image. Defaults to 0.7.
        ds_max (int, optional): The maximum downsampling factor. Defaults to 7.
        diams (torch.Tensor, optional): The diameter of the objects in the image. Defaults to None.
        pscale (torch.Tensor, optional): The scale factor for the poisson noise, instead of sampling. Defaults to None.
        iso (bool, optional): Whether to use isotropic gaussian blur. Defaults to True.
        sigma0 (torch.Tensor, optional): The standard deviation of the gaussian filter for the Y axis, instead of sampling. Defaults to None.
        sigma1 (torch.Tensor, optional): The standard deviation of the gaussian filter for the X axis, instead of sampling. Defaults to None.
        ds (torch.Tensor, optional): The downsampling factor for each image, instead of sampling. Defaults to None.

    Returns:
        torch.Tensor: The noisy image tensor of the same shape as the input image.
    """
    device = lbl.device
    imgi = torch.zeros_like(lbl)

    diams = diams if diams is not None else 30. * torch.ones(len(lbl), device=device)
    #ds0 = 1 if ds is None else ds.item()
    ds = ds * torch.ones(
        (len(lbl),), device=device, dtype=torch.long) if ds is not None else ds

    # add gaussian blur
    iblur = np.random.rand(len(lbl)) < blur
    if iblur.sum() > 0:
        if sigma0 is None:
            # was 10
            xrand = np.random.exponential(1, size=iblur.sum())
            xrand = np.clip(xrand * 0.5, 0.1, 1.0)
            xrand *= gblur
            sigma0 = diams[iblur] / 30. * 5. * torch.from_numpy(xrand).float().to(
                device)
            #(1 + torch.rand(iblur.sum(), device=device))
            if not iso:
                sr = diams[iblur] / 30. * 2 * (1 +
                                               torch.rand(iblur.sum(), device=device))
                sigma1 = (torch.rand(iblur.sum(), device=device) > 0.66) * sr
            else:
                sigma1 = sigma0.clone(
                )  #+ torch.randint(0, 3, size=(len(sigma0.clone()),), device=device)
        else:
            sigma0 = sigma0 * torch.ones((iblur.sum(),), device=device)
            sigma1 = sigma1 * torch.ones((iblur.sum(),), device=device)

        # create gaussian filter
        xr = max(8, sigma0.max().long() * 2)
        gfilt0 = torch.exp(-torch.arange(-xr + 1, xr, device=device)**2 /
                           (2 * sigma0.unsqueeze(-1)**2))
        gfilt0 /= gfilt0.sum(axis=-1, keepdims=True)
        gfilt1 = torch.zeros_like(gfilt0)
        gfilt1[sigma1 == sigma0] = gfilt0[sigma1 == sigma0]
        gfilt1[sigma1 != sigma0] = torch.exp(
            -torch.arange(-xr + 1, xr, device=device)**2 /
            (2 * sigma1[sigma1 != sigma0].unsqueeze(-1)**2))
        gfilt1[sigma1 == 0] = 0.
        gfilt1[sigma1 == 0, xr] = 1.
        gfilt1 /= gfilt1.sum(axis=-1, keepdims=True)
        gfilt = torch.einsum("ck,cl->ckl", gfilt0, gfilt1)
        gfilt /= gfilt.sum(axis=(1, 2), keepdims=True)

        imgi[iblur] = conv2d(lbl[iblur].transpose(1, 0), gfilt.unsqueeze(1),
                             padding=gfilt.shape[-1] // 2,
                             groups=gfilt.shape[0]).transpose(1, 0)

    imgi[~iblur] = lbl[~iblur]

    # downsample
    ii = []
    idownsample = np.random.rand(len(lbl)) < downsample
    if (ds is None and idownsample.sum() > 0.) or not iso:
        ds = torch.ones(len(lbl), dtype=torch.long, device=device)
        ds[idownsample] = torch.randint(2, ds_max + 1, size=(idownsample.sum(),),
                                        device=device)
        ii = torch.nonzero(ds > 1)
    elif ds is not None and (ds > 1).sum():
        ii = torch.nonzero(ds > 1)
    for k in ii:
        i0 = imgi[k:k + 1, :, ::ds[k], ::ds[k]] if iso else imgi[k:k + 1, :, ::ds[k]]
        imgi[k] = interpolate(i0, size=lbl[k].shape[-2:], mode="bilinear")

    # add poisson noise
    ipoisson = np.random.rand(len(lbl)) < poisson
    if ipoisson.sum() > 0:
        if pscale is None:
            pscale = torch.zeros(len(lbl))
            m = torch.distributions.gamma.Gamma(alpha, beta)
            pscale = torch.clamp(m.rsample(sample_shape=(ipoisson.sum(),)), 1.)
            #pscale = torch.clamp(20 * (torch.rand(size=(len(lbl),), device=lbl.device)), 1.5)
            pscale = pscale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        else:
            pscale = pscale * torch.ones((ipoisson.sum(), 1, 1, 1), device=device)
        imgi[ipoisson] = torch.poisson(pscale * imgi[ipoisson])
    imgi[~ipoisson] = imgi[~ipoisson]

    # renormalize
    imgi = img_norm(imgi)

    return imgi


def random_rotate_and_resize_noise(data, labels=None, diams=None, poisson=0.7, blur=0.7,
                                   downsample=0.0, beta=0.7, gblur=1.0, diam_mean=30,
                                   ds_max=7, iso=True, rotate=True,
                                   device=torch.device("cuda"), xy=(224, 224),
                                   nchan_noise=1, keep_raw=True):
    """
    Applies random rotation, resizing, and noise to the input data.

    Args:
        data (numpy.ndarray): The input data.
        labels (numpy.ndarray, optional): The flow and cellprob labels associated with the data. Defaults to None.
        diams (float, optional): The diameter of the objects. Defaults to None.
        poisson (float, optional): The Poisson noise probability. Defaults to 0.7.
        blur (float, optional): The blur probability. Defaults to 0.7.
        downsample (float, optional): The downsample probability. Defaults to 0.0.
        beta (float, optional): The beta value for the poisson noise distribution. Defaults to 0.7.
        gblur (float, optional): The Gaussian blur level. Defaults to 1.0.
        diam_mean (float, optional): The mean diameter. Defaults to 30.
        ds_max (int, optional): The maximum downsample value. Defaults to 7.
        iso (bool, optional): Whether to apply isotropic augmentation. Defaults to True.
        rotate (bool, optional): Whether to apply rotation augmentation. Defaults to True.
        device (torch.device, optional): The device to use. Defaults to torch.device("cuda").
        xy (tuple, optional): The size of the output image. Defaults to (224, 224).
        nchan_noise (int, optional): The number of channels to add noise to. Defaults to 1.
        keep_raw (bool, optional): Whether to keep the raw image. Defaults to True.

    Returns:
        torch.Tensor: The augmented image and augmented noisy/blurry/downsampled version of image.
        torch.Tensor: The augmented labels.
        float: The scale factor applied to the image.
    """

    diams = 30 if diams is None else diams
    random_diam = diam_mean * (2**(2 * np.random.rand(len(data)) - 1))
    random_rsc = diams / random_diam  #/ random_diam
    #rsc /= random_scale
    xy0 = (340, 340)
    nchan = data[0].shape[0]
    data_new = np.zeros((len(data), (1 + keep_raw) * nchan, xy0[0], xy0[1]), "float32")
    labels_new = np.zeros((len(data), 3, xy0[0], xy0[1]), "float32")
    for i in range(
            len(data)):  #, (sc, img, lbl) in enumerate(zip(random_rsc, data, labels)):
        sc = random_rsc[i]
        img = data[i]
        lbl = labels[i] if labels is not None else None
        # create affine transform to resize
        Ly, Lx = img.shape[-2:]
        dxy = np.maximum(0, np.array([Lx / sc - xy0[1], Ly / sc - xy0[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy
        cc = np.array([Lx / 2, Ly / 2])
        cc1 = cc - np.array([Lx - xy0[1], Ly - xy0[0]]) / 2 + dxy
        pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
        pts2 = np.float32(
            [cc1, cc1 + np.array([1, 0]) / sc, cc1 + np.array([0, 1]) / sc])
        M = cv2.getAffineTransform(pts1, pts2)

        # apply to image
        for c in range(nchan):
            img_rsz = cv2.warpAffine(img[c], M, xy0, flags=cv2.INTER_LINEAR)
            #img_noise = add_noise(torch.from_numpy(img_rsz).to(device).unsqueeze(0)).cpu().numpy().squeeze(0)
            data_new[i, c] = img_rsz
            if keep_raw:
                data_new[i, c + nchan] = img_rsz

        if lbl is not None:
            # apply to labels
            labels_new[i, 0] = cv2.warpAffine(lbl[0], M, xy0, flags=cv2.INTER_NEAREST)
            labels_new[i, 1] = cv2.warpAffine(lbl[1], M, xy0, flags=cv2.INTER_LINEAR)
            labels_new[i, 2] = cv2.warpAffine(lbl[2], M, xy0, flags=cv2.INTER_LINEAR)

    rsc = random_diam / diam_mean

    # add noise before augmentations
    img = torch.from_numpy(data_new).to(device)
    img = torch.clamp(img, 0.)
    # just add noise to cyto if nchan_noise=1
    img[:, :nchan_noise] = add_noise(
        img[:, :nchan_noise], poisson=poisson, blur=blur, ds_max=ds_max, iso=iso,
        downsample=downsample, beta=beta, gblur=gblur,
        diams=torch.from_numpy(random_diam).to(device).float())
    # img -= img.mean(dim=(-2,-1), keepdim=True)
    # img /= img.std(dim=(-2,-1), keepdim=True) + 1e-3
    img = img.cpu().numpy()

    # augmentations
    img, lbl, scale = transforms.random_rotate_and_resize(
        img,
        Y=labels_new,
        xy=xy,
        rotate=False if not iso else rotate,
        #(iso and downsample==0),
        rescale=rsc,
        scale_range=0.5)
    img = torch.from_numpy(img).to(device)
    lbl = torch.from_numpy(lbl).to(device)

    return img, lbl, scale


def one_chan_cellpose(device, model_type="cyto2", pretrained_model=None):
    """
    Creates a Cellpose network with a single input channel.

    Args:
        device (str): The device to run the network on.
        model_type (str, optional): The type of Cellpose model to use. Defaults to "cyto2".
        pretrained_model (str, optional): The path to a pretrained model file. Defaults to None.

    Returns:
        torch.nn.Module: The Cellpose network with a single input channel.
    """
    if pretrained_model is not None and not os.path.exists(pretrained_model):
        model_type = pretrained_model
        pretrained_model = None
    nbase = [32, 64, 128, 256]
    nchan = 1
    net1 = resnet_torch.CPnet([nchan, *nbase], nout=3, sz=3).to(device)
    filename = model_path(model_type,
                          0) if pretrained_model is None else pretrained_model
    weights = torch.load(filename)
    zp = 0
    print(filename)
    for name in net1.state_dict():
        if ("res_down_0.conv.conv_0" not in name and
                #"output" not in name and
                "res_down_0.proj" not in name and name != "diam_mean" and
                name != "diam_labels"):
            net1.state_dict()[name].copy_(weights[name])
        elif "res_down_0" in name:
            if len(weights[name].shape) > 0:
                new_weight = torch.zeros_like(net1.state_dict()[name])
                if weights[name].shape[0] == 2:
                    new_weight[:] = weights[name][0]
                elif len(weights[name].shape) > 1 and weights[name].shape[1] == 2:
                    new_weight[:, zp] = weights[name][:, 0]
                else:
                    new_weight = weights[name]
            else:
                new_weight = weights[name]
            net1.state_dict()[name].copy_(new_weight)
    return net1


class CellposeDenoiseModel():
    """ model to run Cellpose and Image restoration """

    def __init__(self, gpu=False, pretrained_model=False, model_type=None,
                 restore_type="denoise_cyto3", chan2_restore=False, device=None):

        self.dn = DenoiseModel(gpu=gpu, model_type=restore_type, chan2=chan2_restore,
                               device=device)
        self.cp = CellposeModel(gpu=gpu, model_type=model_type,
                                pretrained_model=pretrained_model, device=device)

    def eval(self, x, batch_size=8, channels=None, channel_axis=None, z_axis=None,
             normalize=True, rescale=None, diameter=None, tile=True, tile_overlap=0.1,
             augment=False, resample=True, invert=False, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None, stitch_threshold=0.0,
             min_size=15, niter=None, interp=True):
        """
        Restore array or list of images using the image restoration model, and then segment.

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images
            batch_size (int, optional): number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 8.
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
            rescale (float, optional): resize factor for each image, if None, set to 1.0;
                (only used if diameter is None). Defaults to None.
            diameter (float, optional):  diameter for each image, 
                if diameter is None, set to diam_mean or diam_train if available. Defaults to None.
            tile (bool, optional): tiles image to ensure GPU/CPU memory usage limited (recommended). Defaults to True.
            tile_overlap (float, optional): fraction of overlap of tiles when computing flows. Defaults to 0.1.
            augment (bool, optional): augment tiles by flipping and averaging for segmentation. Defaults to False.
            resample (bool, optional): run dynamics at original image size (will be slower but create more accurate boundaries). Defaults to True.
            invert (bool, optional): invert image pixel intensity before running network. Defaults to False.
            flow_threshold (float, optional): flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
            cellprob_threshold (float, optional): all pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
            do_3D (bool, optional): set to True to run 3D segmentation on 3D/4D image input. Defaults to False.
            anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
            stitch_threshold (float, optional): if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation. Defaults to 0.0.
            min_size (int, optional): all ROIs below this size, in pixels, will be discarded. Defaults to 15.
            niter (int, optional): number of iterations for dynamics computation. if None, it is set proportional to the diameter. Defaults to None.
            interp (bool, optional): interpolate during 2D dynamics (not available in 3D) . Defaults to True.
            
        Returns:
            masks (list, np.ndarray): labelled image(s), where 0=no masks; 1,2,...=mask labels
            flows (list): list of lists: flows[k][0] = XY flow in HSV 0-255; flows[k][1] = XY(Z) flows at each pixel; flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics); flows[k][3] = final pixel locations after Euler integration 
            styles (list, np.ndarray): style vector summarizing each image of size 256.
            imgs (list of 2D/3D arrays): Restored images
        """
        if isinstance(normalize, dict):
            normalize_params = {**normalize_default, **normalize}
        elif not isinstance(normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params = normalize_default
            normalize_params["normalize"] = normalize
        normalize_params["invert"] = invert

        img_restore = self.dn.eval(x, batch_size=batch_size, channels=channels,
                                   channel_axis=channel_axis, z_axis=z_axis,
                                   normalize=normalize_params, rescale=rescale,
                                   diameter=diameter, tile=tile,
                                   tile_overlap=tile_overlap)

        # turn off special normalization for segmentation
        normalize_params = normalize_default

        # change channels for segmentation (denoise model outputs up to 2 channels)
        channels_new = [0, 0] if channels[0] == 0 else [1, 2]
        # change diameter if self.ratio > 1 (upsampled to self.dn.diam_mean)
        diameter = self.dn.diam_mean if self.dn.ratio > 1 else diameter
        masks, flows, styles = self.cp.eval(
            img_restore, batch_size=batch_size, channels=channels_new, channel_axis=-1,
            normalize=normalize_params, rescale=rescale, diameter=diameter, tile=tile,
            tile_overlap=tile_overlap, augment=augment, resample=resample,
            invert=invert, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, do_3D=do_3D, anisotropy=anisotropy,
            stitch_threshold=stitch_threshold, min_size=min_size, niter=niter,
            interp=interp)

        return masks, flows, styles, img_restore


class DenoiseModel():
    """
    DenoiseModel class for denoising images using Cellpose denoising model.

    Args:
        gpu (bool, optional): Whether to use GPU for computation. Defaults to False.
        pretrained_model (bool or str or Path, optional): Pretrained model to use for denoising.
            Can be a string or path. Defaults to False.
        nchan (int, optional): Number of channels in the input images, all Cellpose 3 models were trained with nchan=1. Defaults to 1.
        model_type (str, optional): Type of pretrained model to use ("denoise_cyto3", "deblur_cyto3", "upsample_cyto3", ...). Defaults to None.
        chan2 (bool, optional): Whether to use a separate model for the second channel. Defaults to False.
        diam_mean (float, optional): Mean diameter of the objects in the images. Defaults to 30.0.
        device (torch.device, optional): Device to use for computation. Defaults to None.

    Attributes:
        nchan (int): Number of channels in the input images.
        diam_mean (float): Mean diameter of the objects in the images.
        net (CPnet): Cellpose network for denoising.
        pretrained_model (bool or str or Path): Pretrained model path to use for denoising.
        net_chan2 (CPnet or None): Cellpose network for the second channel, if applicable.
        net_type (str): Type of the denoising network.

    Methods:
        eval(x, batch_size=8, channels=None, channel_axis=None, z_axis=None,
                normalize=True, rescale=None, diameter=None, tile=True, tile_overlap=0.1)
            Denoise array or list of images using the denoising model.

        _eval(net, x, normalize=True, rescale=None, diameter=None, tile=True,
                tile_overlap=0.1)
            Run denoising model on a single channel.
    """

    def __init__(self, gpu=False, pretrained_model=False, nchan=1, model_type=None,
                 chan2=False, diam_mean=30., device=None):
        self.nchan = nchan
        if pretrained_model and (not isinstance(pretrained_model, str) and
                                 not isinstance(pretrained_model, Path)):
            raise ValueError("pretrained_model must be a string or path")

        self.diam_mean = diam_mean
        builtin = True
        if model_type is not None or (pretrained_model and
                                      not os.path.exists(pretrained_model)):
            pretrained_model_string = model_type if model_type is not None else "denoise_cyto3"
            if ~np.any([pretrained_model_string == s for s in MODEL_NAMES]):
                pretrained_model_string = "denoise_cyto3"
            pretrained_model = model_path(pretrained_model_string)
            if (pretrained_model and not os.path.exists(pretrained_model)):
                denoise_logger.warning("pretrained model has incorrect path")
            denoise_logger.info(f">> {pretrained_model_string} << model set to be used")
            self.diam_mean = 17. if "nuclei" in pretrained_model_string else 30.
        else:
            if pretrained_model:
                builtin = False
                pretrained_model_string = pretrained_model
                denoise_logger.info(f">>>> loading model {pretrained_model_string}")

        # assign network device
        self.mkldnn = None
        if device is None:
            sdevice, gpu = assign_device(use_torch=True, gpu=gpu)
        self.device = device if device is not None else sdevice
        if device is not None:
            device_gpu = self.device.type == "cuda"
        self.gpu = gpu if device is None else device_gpu
        if not self.gpu:
            self.mkldnn = check_mkl(True)

        # create network
        self.nchan = nchan
        self.nclasses = 1
        nbase = [32, 64, 128, 256]
        self.nchan = nchan
        self.nbase = [nchan, *nbase]

        self.net = CPnet(self.nbase, self.nclasses, sz=3, mkldnn=self.mkldnn,
                         max_pool=True, diam_mean=diam_mean).to(self.device)

        self.pretrained_model = pretrained_model
        self.net_chan2 = None
        if self.pretrained_model:
            self.net.load_model(self.pretrained_model, device=self.device)
            denoise_logger.info(
                f">>>> model diam_mean = {self.diam_mean: .3f} (ROIs rescaled to this size during training)"
            )
            if chan2 and builtin:
                chan2_path = model_path(
                    os.path.split(self.pretrained_model)[-1].split("_")[0] + "_nuclei")
                print(f"loading model for chan2: {os.path.split(str(chan2_path))[-1]}")
                self.net_chan2 = CPnet(self.nbase, self.nclasses, sz=3,
                                       mkldnn=self.mkldnn, max_pool=True,
                                       diam_mean=17.).to(self.device)
                self.net_chan2.load_model(chan2_path, device=self.device)
        self.net_type = "cellpose_denoise"

    def eval(self, x, batch_size=8, channels=None, channel_axis=None, z_axis=None,
             normalize=True, rescale=None, diameter=None, tile=True, tile_overlap=0.1):
        """
        Restore array or list of images using the image restoration model.

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images
            batch_size (int, optional): number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 8.
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
            rescale (float, optional): resize factor for each image, if None, set to 1.0;
                (only used if diameter is None). Defaults to None.
            diameter (float, optional):  diameter for each image, 
                if diameter is None, set to diam_mean or diam_train if available. Defaults to None.
            tile (bool, optional): tiles image to ensure GPU/CPU memory usage limited (recommended). Defaults to True.
            tile_overlap (float, optional): fraction of overlap of tiles when computing flows. Defaults to 0.1.
            
        Returns:
            imgs (list of 2D/3D arrays): Restored images

        """
        if isinstance(x, list) or x.squeeze().ndim == 5:
            tqdm_out = utils.TqdmToLogger(denoise_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            imgs = []
            for i in iterator:
                imgi = self.eval(
                    x[i], batch_size=batch_size,
                    channels=channels[i] if channels is not None and
                    ((len(channels) == len(x) and
                      (isinstance(channels[i], list) or
                       isinstance(channels[i], np.ndarray)) and len(channels[i]) == 2))
                    else channels, channel_axis=channel_axis, z_axis=z_axis,
                    normalize=normalize,
                    rescale=rescale[i] if isinstance(rescale, list) or
                    isinstance(rescale, np.ndarray) else rescale,
                    diameter=diameter[i] if isinstance(diameter, list) or
                    isinstance(diameter, np.ndarray) else diameter, tile=tile,
                    tile_overlap=tile_overlap)
                imgs.append(imgi)
            if isinstance(x, np.ndarray):
                imgs = np.array(imgs)
            return imgs

        else:
            # reshape image
            x = transforms.convert_image(x, channels, channel_axis=channel_axis,
                                         z_axis=z_axis)
            if x.ndim < 4:
                squeeze = True
                x = x[np.newaxis, ...]
            else:
                squeeze = False

            # may need to interpolate image before running upsampling
            self.ratio = 1.
            if "upsample" in self.pretrained_model:
                Ly, Lx = x.shape[-3:-1]
                if diameter is not None and 3 <= diameter < self.diam_mean:
                    self.ratio = self.diam_mean / diameter
                    denoise_logger.info(
                        f"upsampling image to {self.diam_mean} pixel diameter ({self.ratio:0.2f} times)"
                    )
                    Lyr, Lxr = int(Ly * self.ratio), int(Lx * self.ratio)
                    x = transforms.resize_image(x, Ly=Lyr, Lx=Lxr)
                else:
                    denoise_logger.warning(
                        f"not interpolating image before upsampling because diameter is set >= {self.diam_mean}"
                    )
                    #raise ValueError(f"diameter is set to {diameter}, needs to be >=3 and < {self.dn.diam_mean}")

            self.batch_size = batch_size

            if diameter is not None and diameter > 0:
                rescale = self.diam_mean / diameter
            elif rescale is None:
                rescale = 1.0

            if np.ptp(x[..., -1]) < 1e-3 or channels[-1] == 0:
                x = x[..., :1]

            for c in range(x.shape[-1]):
                rescale0 = rescale * 30. / 17. if c == 1 else rescale
                if c == 0 or self.net_chan2 is None:
                    x[...,
                      c] = self._eval(self.net, x[..., c:c + 1], batch_size=batch_size,
                                      normalize=normalize, rescale=rescale0, tile=tile,
                                      tile_overlap=tile_overlap)
                else:
                    x[...,
                      c] = self._eval(self.net_chan2, x[...,
                                                        c:c + 1], batch_size=batch_size,
                                      normalize=normalize, rescale=rescale0, tile=tile,
                                      tile_overlap=tile_overlap)
            x = x[0] if squeeze else x
        return x

    def _eval(self, net, x, batch_size=8, normalize=True, rescale=None, tile=True,
              tile_overlap=0.1):
        """
        Run image restoration model on a single channel.

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images
            batch_size (int, optional): number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 8.
            normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; 
                can also pass dictionary of parameters (all keys are optional, default values shown): 
                    - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
                    - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
                    - "normalize"=True ; run normalization (if False, all following parameters ignored)
                    - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
                    - "tile_norm"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
                    - "norm3D"=False ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
                Defaults to True.
            rescale (float, optional): resize factor for each image, if None, set to 1.0;
                (only used if diameter is None). Defaults to None.
            tile (bool, optional): tiles image to ensure GPU/CPU memory usage limited (recommended). Defaults to True.
            tile_overlap (float, optional): fraction of overlap of tiles when computing flows. Defaults to 0.1.
            
        Returns:
            imgs (list of 2D/3D arrays): Restored images

        """
        if isinstance(normalize, dict):
            normalize_params = {**normalize_default, **normalize}
        elif not isinstance(normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params = normalize_default
            normalize_params["normalize"] = normalize

        tic = time.time()
        shape = x.shape
        nimg = shape[0]

        do_normalization = True if normalize_params["normalize"] else False

        tqdm_out = utils.TqdmToLogger(denoise_logger, level=logging.INFO)
        iterator = trange(nimg, file=tqdm_out,
                          mininterval=30) if nimg > 1 else range(nimg)
        imgs = np.zeros((*x.shape[:-1], 1), np.float32)
        for i in iterator:
            img = np.asarray(x[i])
            if do_normalization:
                img = transforms.normalize_img(img, **normalize_params)
            if rescale != 1.0:
                img = transforms.resize_image(img, rsz=[rescale, rescale])
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]
            yf, style = run_net(net, img, batch_size=batch_size, augment=False,
                                tile=tile, tile_overlap=tile_overlap)
            img = transforms.resize_image(yf, Ly=x.shape[-3], Lx=x.shape[-2])

            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            imgs[i] = img
            del yf, style
        net_time = time.time() - tic
        if nimg > 1:
            denoise_logger.info("imgs denoised in %2.2fs" % (net_time))

        return imgs.squeeze()


def train(net, train_data=None, train_labels=None, train_files=None, test_data=None,
          test_labels=None, test_files=None, train_probs=None, test_probs=None,
          lam=[1., 1.5, 0.], scale_range=0.5, seg_model_type="cyto2", save_path=None,
          save_every=100, save_each=False, poisson=0.7, beta=0.7, blur=0.7, gblur=1.0,
          iso=True, downsample=0., learning_rate=0.005, n_epochs=500, momentum=0.9,
          weight_decay=0.00001, batch_size=8, nimg_per_epoch=None,
          nimg_test_per_epoch=None):

    # net properties
    device = net.device
    nchan = net.nchan
    diam_mean = net.diam_mean.item()

    args = np.array([poisson, beta, blur, gblur, downsample])
    if args.ndim == 1:
        args = args[:, np.newaxis]
    poisson, beta, blur, gblur, downsample = args
    nnoise = len(poisson)

    d = datetime.datetime.now()
    if save_path is not None:
        filename = ""
        lstrs = ["per", "seg", "rec"]
        for k, (l, s) in enumerate(zip(lam, lstrs)):
            filename += f"{s}_{l:.2f}_"
        if poisson.sum() > 0:
            filename += "poisson_"
        if blur.sum() > 0:
            if iso:
                filename += "blur_"
            else:
                filename += "bluraniso_"
        if downsample.sum() > 0:
            filename += "downsample_"
        filename += d.strftime("%Y_%m_%d_%H_%M_%S.%f")
        filename = os.path.join(save_path, filename)
        print(filename)
    for i in range(len(poisson)):
        denoise_logger.info(
            f"poisson: {poisson[i]: 0.2f}, beta: {beta[i]: 0.2f}, blur: {blur[i]: 0.2f}, gblur: {gblur[i]: 0.2f}, downsample: {downsample[i]: 0.2f}"
        )
    net1 = one_chan_cellpose(device=device, pretrained_model=seg_model_type)

    learning_rate_const = learning_rate
    LR = np.linspace(0, learning_rate_const, 10)
    LR = np.append(LR, learning_rate_const * np.ones(n_epochs - 100))
    for i in range(10):
        LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    learning_rate = LR

    batch_size = 8
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate[0],
                                  weight_decay=weight_decay)
    if train_data is not None:
        nimg = len(train_data)
        diam_train = np.array(
            [utils.diameters(train_labels[k])[0] for k in trange(len(train_labels))])
        diam_train[diam_train < 5] = 5.
        if test_data is not None:
            diam_test = np.array(
                [utils.diameters(test_labels[k])[0] for k in trange(len(test_labels))])
            diam_test[diam_test < 5] = 5.
            nimg_test = len(test_data)
    else:
        nimg = len(train_files)
        denoise_logger.info(">>> using files instead of loading dataset")
        train_labels_files = [str(tf)[:-4] + f"_flows.tif" for tf in train_files]
        denoise_logger.info(">>> computing diameters")
        diam_train = np.array([
            utils.diameters(io.imread(train_labels_files[k])[0])[0]
            for k in trange(len(train_labels_files))
        ])
        diam_train[diam_train < 5] = 5.
        if test_files is not None:
            nimg_test = len(test_files)
            test_labels_files = [str(tf)[:-4] + f"_flows.tif" for tf in test_files]
            diam_test = np.array([
                utils.diameters(io.imread(test_labels_files[k])[0])[0]
                for k in trange(len(test_labels_files))
            ])
            diam_test[diam_test < 5] = 5.
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs

    tic = time.time()

    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    if test_files is not None or test_data is not None:
        nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    nbatch = 0
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                 p=train_probs)
        torch.manual_seed(iepoch)
        np.random.seed(iepoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate[iepoch]
        lavg, lavg_per, nsum = 0, 0, 0
        for ibatch in range(0, nimg_per_epoch, batch_size):
            inds = rperm[ibatch:ibatch + batch_size]
            if train_data is None:
                imgs = [np.maximum(0, io.imread(train_files[i])[:nchan]) for i in inds]
                lbls = [io.imread(train_labels_files[i])[1:] for i in inds]
            else:
                imgs = [train_data[i][:nchan] for i in inds]
                lbls = [train_labels[i][1:] for i in inds]
            inoise = nbatch % nnoise
            img, lbl, scale = random_rotate_and_resize_noise(
                imgs, lbls, diam_train[inds].copy(), poisson=poisson[inoise],
                beta=beta[inoise], gblur=gblur[inoise], blur=blur[inoise], iso=iso,
                downsample=downsample[inoise], diam_mean=diam_mean, device=device)
            #print(torch.isnan(img).sum())
            if torch.isnan(img).sum():
                import pdb
                pdb.set_trace()
            optimizer.zero_grad()
            loss, loss_per = train_loss(net, img[:, :nchan], net1=net1,
                                        img=img[:, nchan:], lbl=lbl, lam=lam)

            loss.backward()
            optimizer.step()
            lavg += loss.item() * img.shape[0]
            lavg_per += loss_per.item() * img.shape[0]
            nsum += len(img)
            nbatch += 1

        if iepoch % 10 == 0 or iepoch < 10:
            lavg = lavg / nsum
            lavg_per = lavg_per / nsum
            if test_data is not None or test_files is not None:
                lavgt, nsum = 0., 0
                np.random.seed(42)
                rperm = np.random.choice(np.arange(0, nimg_test),
                                         size=(nimg_test_per_epoch,), p=test_probs)
                inoise = iepoch % nnoise
                torch.manual_seed(inoise)
                for ibatch in range(0, nimg_test_per_epoch, batch_size):
                    inds = rperm[ibatch:ibatch + batch_size]
                    if test_data is None:
                        imgs = [
                            np.maximum(0,
                                       io.imread(test_files[i])[:nchan]) for i in inds
                        ]
                        lbls = [io.imread(test_labels_files[i])[1:] for i in inds]
                    else:
                        imgs = [test_data[i][:nchan] for i in inds]
                        lbls = [test_labels[i][1:] for i in inds]
                    img, lbl, scale = random_rotate_and_resize_noise(
                        imgs, lbls, diam_test[inds].copy(), poisson=poisson[inoise],
                        beta=beta[inoise], blur=blur[inoise], gblur=gblur[inoise],
                        iso=iso, downsample=downsample[inoise], diam_mean=diam_mean,
                        device=device)
                    loss, loss_per = test_loss(net, img[:, :nchan], net1=net1,
                                               img=img[:, nchan:], lbl=lbl, lam=lam)

                    lavgt += loss.item() * img.shape[0]
                    nsum += len(img)
                denoise_logger.info(
                    "Epoch %d, Time %4.1fs, Loss %0.3f, loss_per %0.3f, Loss Test %0.3f, LR %2.4f"
                    % (iepoch, time.time() - tic, lavg, lavg_per, lavgt / nsum,
                       learning_rate[iepoch]))
            else:
                denoise_logger.info(
                    "Epoch %d, Time %4.1fs, Loss %0.3f, loss_per %0.3f, LR %2.4f" %
                    (iepoch, time.time() - tic, lavg, lavg_per, learning_rate[iepoch]))
        elif iepoch < 50:
            lavg = lavg / nsum
            lavg_per = lavg_per / nsum
            denoise_logger.info(
                "Epoch %d, Time %4.1fs, Loss %0.3f, loss_per %0.3f, LR %2.4f" %
                (iepoch, time.time() - tic, lavg, lavg_per, learning_rate[iepoch]))

        if save_path is not None:
            if iepoch == n_epochs - 1 or iepoch % save_every == 1:
                if save_each:  #separate files as model progresses
                    filename0 = filename + "_epoch_" + str(iepoch)
                else:
                    filename0 = filename
                denoise_logger.info(f"saving network parameters to {filename0}")
                net.save_model(filename0)
        else:
            filename = save_path

    return filename


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="cellpose parameters")

    input_img_args = parser.add_argument_group("input image arguments")
    input_img_args.add_argument("--dir", default=[], type=str,
                                help="folder containing data to run or train on.")
    input_img_args.add_argument("--img_filter", default=[], type=str,
                                help="end string for images to run on")

    model_args = parser.add_argument_group("model arguments")
    model_args.add_argument("--pretrained_model", default=[], type=str,
                            help="pretrained denoising model")

    training_args = parser.add_argument_group("training arguments")
    training_args.add_argument("--test_dir", default=[], type=str,
                               help="folder containing test data (optional)")
    training_args.add_argument("--file_list", default=[], type=str,
                               help="npy file containing list of train and test files")
    training_args.add_argument("--seg_model_type", default="cyto2", type=str,
                               help="model to use for seg training loss")
    training_args.add_argument(
        "--noise_type", default=[], type=str,
        help="noise type to use (if input, then other noise params are ignored)")
    training_args.add_argument("--poisson", default=0.8, type=float,
                               help="fraction of images to add poisson noise to")
    training_args.add_argument("--beta", default=0.7, type=float,
                               help="scale of poisson noise")
    training_args.add_argument("--blur", default=0., type=float,
                               help="fraction of images to blur")
    training_args.add_argument("--gblur", default=1.0, type=float,
                               help="scale of gaussian blurring stddev")
    training_args.add_argument("--downsample", default=0., type=float,
                               help="fraction of images to downsample")
    training_args.add_argument("--lam_per", default=1.0, type=float,
                               help="weighting of perceptual loss")
    training_args.add_argument("--lam_seg", default=1.5, type=float,
                               help="weighting of segmentation loss")
    training_args.add_argument("--lam_rec", default=0., type=float,
                               help="weighting of reconstruction loss")
    training_args.add_argument(
        "--diam_mean", default=30., type=float, help=
        "mean diameter to resize cells to during training -- if starting from pretrained models it cannot be changed from 30.0"
    )
    training_args.add_argument("--learning_rate", default=0.001, type=float,
                               help="learning rate. Default: %(default)s")
    training_args.add_argument("--n_epochs", default=2000, type=int,
                               help="number of epochs. Default: %(default)s")
    training_args.add_argument(
        "--nimg_per_epoch", default=0, type=int,
        help="number of images per epoch. Default is length of training images")
    training_args.add_argument(
        "--nimg_test_per_epoch", default=0, type=int,
        help="number of test images per epoch. Default is length of testing images")

    io.logger_setup()

    args = parser.parse_args()

    if len(args.noise_type) > 0:
        noise_type = args.noise_type
        if noise_type == "poisson":
            poisson = 0.8
            blur = 0.
            downsample = 0.
            beta = 0.7
            gblur = 1.0
        elif noise_type == "blur":
            poisson = 0.8
            blur = 0.8
            downsample = 0.
            beta = 0.1
            gblur = 1.0
        elif noise_type == "downsample":
            poisson = 0.8
            blur = 0.8
            downsample = 0.8
            beta = 0.01
            gblur = 0.5
        elif noise_type == "all":
            poisson = [0.8, 0.8, 0.8]
            blur = [0., 0.8, 0.8]
            downsample = [0., 0., 0.8]
            beta = [0.7, 0.1, 0.01]
            gblur = [0., 1.0, 0.5]
        else:
            raise ValueError(f"{noise_type} noise_type is not supported")
    else:
        poisson, beta = args.poisson, args.beta
        blur, gblur = args.blur, args.gblur
        downsample = args.downsample

    pretrained_model = None if len(
        args.pretrained_model) == 0 else args.pretrained_model
    model = DenoiseModel(gpu=True, nchan=1, diam_mean=args.diam_mean,
                         pretrained_model=pretrained_model)

    train_data, labels, train_files, train_probs = None, None, None, None
    test_data, test_labels, test_files, test_probs = None, None, None, None
    if len(args.file_list) == 0:
        output = io.load_train_test_data(args.dir, args.test_dir, "_img", "_masks", 0,
                                         0)
        images, labels, image_names, test_images, test_labels, image_names_test = output
        train_data = []
        for i in range(len(images)):
            img = images[i].astype("float32")
            if img.ndim > 2:
                img = img[0]
            train_data.append(
                np.maximum(transforms.normalize99(img), 0)[np.newaxis, :, :])
        if len(args.test_dir) > 0:
            test_data = []
            for i in range(len(test_images)):
                img = test_images[i].astype("float32")
                if img.ndim > 2:
                    img = img[0]
                test_data.append(
                    np.maximum(transforms.normalize99(img), 0)[np.newaxis, :, :])
        save_path = os.path.join(args.dir, "../models/")
    else:
        root = args.dir
        denoise_logger.info(
            ">>> using file_list (assumes images are normalized and have flows!)")
        dat = np.load(args.file_list, allow_pickle=True).item()
        train_files = dat["train_files"]
        test_files = dat["test_files"]
        train_probs = dat["train_probs"] if "train_probs" in dat else None
        test_probs = dat["test_probs"] if "test_probs" in dat else None
        if str(train_files[0])[:len(str(root))] != str(root):
            for i in range(len(train_files)):
                new_path = root / Path(*train_files[i].parts[-3:])
                if i == 0:
                    print(f"changing path from {train_files[i]} to {new_path}")
                train_files[i] = new_path

            for i in range(len(test_files)):
                new_path = root / Path(*test_files[i].parts[-3:])
                test_files[i] = new_path
        save_path = os.path.join(args.dir, "models/")

    os.makedirs(save_path, exist_ok=True)

    nimg_per_epoch = None if args.nimg_per_epoch == 0 else args.nimg_per_epoch
    nimg_test_per_epoch = None if args.nimg_test_per_epoch == 0 else args.nimg_test_per_epoch

    model_path = train(
        model.net, train_data=train_data, train_labels=labels, train_files=train_files,
        test_data=test_data, test_labels=test_labels, test_files=test_files,
        train_probs=train_probs, test_probs=test_probs, poisson=poisson, beta=beta,
        blur=blur, gblur=gblur, downsample=downsample, iso=True, n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        lam=[args.lam_per, args.lam_seg, args.lam_rec
            ], seg_model_type=args.seg_model_type, nimg_per_epoch=nimg_per_epoch,
        nimg_test_per_epoch=nimg_test_per_epoch, save_path=save_path)


def seg_train_noisy(model, train_data, train_labels, test_data=None, test_labels=None,
                    poisson=0.8, blur=0.0, downsample=0.0, save_path=None,
                    save_every=100, save_each=False, learning_rate=0.2, n_epochs=500,
                    momentum=0.9, weight_decay=0.00001, SGD=True, batch_size=8,
                    nimg_per_epoch=None, diameter=None, rescale=True, z_masking=False,
                    model_name=None):
    """ train function uses loss function model.loss_fn in models.py
    
    (data should already be normalized)

    """

    d = datetime.datetime.now()

    model.n_epochs = n_epochs
    if isinstance(learning_rate, (list, np.ndarray)):
        if isinstance(learning_rate, np.ndarray) and learning_rate.ndim > 1:
            raise ValueError("learning_rate.ndim must equal 1")
        elif len(learning_rate) != n_epochs:
            raise ValueError(
                "if learning_rate given as list or np.ndarray it must have length n_epochs"
            )
        model.learning_rate = learning_rate
        model.learning_rate_const = mode(learning_rate)[0][0]
    else:
        model.learning_rate_const = learning_rate
        # set learning rate schedule
        if SGD:
            LR = np.linspace(0, model.learning_rate_const, 10)
            if model.n_epochs > 250:
                LR = np.append(
                    LR, model.learning_rate_const * np.ones(model.n_epochs - 100))
                for i in range(10):
                    LR = np.append(LR, LR[-1] / 2 * np.ones(10))
            else:
                LR = np.append(
                    LR,
                    model.learning_rate_const * np.ones(max(0, model.n_epochs - 10)))
        else:
            LR = model.learning_rate_const * np.ones(model.n_epochs)
        model.learning_rate = LR

    model.batch_size = batch_size
    model._set_optimizer(model.learning_rate[0], momentum, weight_decay, SGD)
    model._set_criterion()

    nimg = len(train_data)

    # compute average cell diameter
    if diameter is None:
        diam_train = np.array(
            [utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
        diam_train_mean = diam_train[diam_train > 0].mean()
        model.diam_labels = diam_train_mean
        if rescale:
            diam_train[diam_train < 5] = 5.
            if test_data is not None:
                diam_test = np.array([
                    utils.diameters(test_labels[k][0])[0]
                    for k in range(len(test_labels))
                ])
                diam_test[diam_test < 5] = 5.
            denoise_logger.info(">>>> median diameter set to = %d" % model.diam_mean)
    elif rescale:
        diam_train_mean = diameter
        model.diam_labels = diameter
        denoise_logger.info(">>>> median diameter set to = %d" % model.diam_mean)
        diam_train = diameter * np.ones(len(train_labels), "float32")
        if test_data is not None:
            diam_test = diameter * np.ones(len(test_labels), "float32")

    denoise_logger.info(
        f">>>> mean of training label mask diameters (saved to model) {diam_train_mean:.3f}"
    )
    model.net.diam_labels.data = torch.ones(1, device=model.device) * diam_train_mean

    nchan = train_data[0].shape[0]
    denoise_logger.info(">>>> training network with %d channel input <<<<" % nchan)
    denoise_logger.info(">>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f" %
                        (model.learning_rate_const, model.batch_size, weight_decay))

    if test_data is not None:
        denoise_logger.info(f">>>> ntrain = {nimg}, ntest = {len(test_data)}")
    else:
        denoise_logger.info(f">>>> ntrain = {nimg}")

    tic = time.time()

    lavg, nsum = 0, 0

    if save_path is not None:
        _, file_label = os.path.split(save_path)
        file_path = os.path.join(save_path, "models/")

        if not os.path.exists(file_path):
            os.makedirs(file_path)
    else:
        denoise_logger.warning("WARNING: no save_path given, model not saving")

    ksave = 0

    # cannot train with mkldnn
    model.net.mkldnn = False

    # get indices for each epoch for training
    np.random.seed(0)
    inds_all = np.zeros((0,), "int32")
    if nimg_per_epoch is None or nimg > nimg_per_epoch:
        nimg_per_epoch = nimg
    denoise_logger.info(f">>>> nimg_per_epoch = {nimg_per_epoch}")
    while len(inds_all) < n_epochs * nimg_per_epoch:
        rperm = np.random.permutation(nimg)
        inds_all = np.hstack((inds_all, rperm))

    for iepoch in range(model.n_epochs):
        if SGD:
            model._set_learning_rate(model.learning_rate[iepoch])
        np.random.seed(iepoch)
        rperm = inds_all[iepoch * nimg_per_epoch:(iepoch + 1) * nimg_per_epoch]
        for ibatch in range(0, nimg_per_epoch, batch_size):
            inds = rperm[ibatch:ibatch + batch_size]
            imgi, lbl, scale = random_rotate_and_resize_noise(
                [train_data[i] for i in inds], [train_labels[i][1:] for i in inds],
                poisson=poisson, blur=blur, downsample=downsample,
                diams=diam_train[inds], diam_mean=model.diam_mean)
            imgi = imgi[:, :1]  # keep noisy only
            if z_masking:
                nc = imgi.shape[1]
                nb = imgi.shape[0]
                ncmin = (np.random.rand(nb) > 0.25) * (np.random.randint(
                    nc // 2 - 1, size=nb))
                ncmax = nc - (np.random.rand(nb) > 0.25) * (np.random.randint(
                    nc // 2 - 1, size=nb))
                for b in range(nb):
                    imgi[b, :ncmin[b]] = 0
                    imgi[b, ncmax[b]:] = 0

            train_loss = model._train_step(imgi, lbl)
            lavg += train_loss
            nsum += len(imgi)

        if iepoch % 10 == 0 or iepoch == 5:
            lavg = lavg / nsum
            if test_data is not None:
                lavgt, nsum = 0., 0
                np.random.seed(42)
                rperm = np.arange(0, len(test_data), 1, int)
                for ibatch in range(0, len(test_data), batch_size):
                    inds = rperm[ibatch:ibatch + batch_size]
                    imgi, lbl, scale = random_rotate_and_resize_noise(
                        [test_data[i] for i in inds],
                        [test_labels[i][1:] for i in inds], poisson=poisson, blur=blur,
                        downsample=downsample, diams=diam_test[inds],
                        diam_mean=model.diam_mean)
                    imgi = imgi[:, :1]  # keep noisy only
                    test_loss = model._test_eval(imgi, lbl)
                    lavgt += test_loss
                    nsum += len(imgi)

                denoise_logger.info(
                    "Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f" %
                    (iepoch, time.time() - tic, lavg, lavgt / nsum,
                     model.learning_rate[iepoch]))
            else:
                denoise_logger.info(
                    "Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f" %
                    (iepoch, time.time() - tic, lavg, model.learning_rate[iepoch]))

            lavg, nsum = 0, 0

        if save_path is not None:
            if iepoch == model.n_epochs - 1 or iepoch % save_every == 1:
                # save model at the end
                if save_each:  #separate files as model progresses
                    if model_name is None:
                        filename = "{}_{}_{}_{}".format(
                            model.net_type, file_label,
                            d.strftime("%Y_%m_%d_%H_%M_%S.%f"), "epoch_" + str(iepoch))
                    else:
                        filename = "{}_{}".format(model_name, "epoch_" + str(iepoch))
                else:
                    if model_name is None:
                        filename = "{}_{}_{}".format(model.net_type, file_label,
                                                     d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                    else:
                        filename = model_name
                filename = os.path.join(file_path, filename)
                ksave += 1
                denoise_logger.info(f"saving network parameters to {filename}")
                model.net.save_model(filename)
        else:
            filename = save_path

    # reset to mkldnn if available
    model.net.mkldnn = model.mkldnn
    return filename
