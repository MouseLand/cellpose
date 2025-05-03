"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import logging
import numpy as np
from tqdm import trange
from . import transforms, utils

import torch

TORCH_ENABLED = True

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)


def use_gpu(gpu_number=0, use_torch=True):
    """ 
    Check if GPU is available for use.

    Args:
        gpu_number (int): The index of the GPU to be used. Default is 0.
        use_torch (bool): Whether to use PyTorch for GPU check. Default is True.

    Returns:
        bool: True if GPU is available, False otherwise.

    Raises:
        ValueError: If use_torch is False, as cellpose only runs with PyTorch now.
    """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError("cellpose only runs with PyTorch now")


def _use_gpu_torch(gpu_number=0):
    """
    Checks if CUDA or MPS is available and working with PyTorch.

    Args:
        gpu_number (int): The GPU device number to use (default is 0).

    Returns:
        bool: True if CUDA or MPS is available and working, False otherwise.
    """
    try:
        device = torch.device("cuda:" + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info("** TORCH CUDA version installed and working. **")
        return True
    except:
        pass
    try:
        device = torch.device('mps:' + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info('** TORCH MPS version installed and working. **')
        return True
    except:
        core_logger.info('Neither TORCH CUDA nor MPS version not installed/working.')
        return False


def assign_device(use_torch=True, gpu=False, device=0):
    """
    Assigns the device (CPU or GPU or mps) to be used for computation.

    Args:
        use_torch (bool, optional): Whether to use torch for GPU detection. Defaults to True.
        gpu (bool, optional): Whether to use GPU for computation. Defaults to False.
        device (int or str, optional): The device index or name to be used. Defaults to 0.

    Returns:
        torch.device, bool (True if GPU is used, False otherwise)
    """

    if isinstance(device, str):
        if device != "mps" or not(gpu and torch.backends.mps.is_available()):
            device = int(device)
    if gpu and use_gpu(use_torch=True):
        try:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{device}')
                core_logger.info(">>>> using GPU (CUDA)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                core_logger.info(">>>> using GPU (MPS)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
    else:
        device = torch.device('cpu')
        core_logger.info('>>>> using CPU')
        gpu = False
        cpu = True
    
    if cpu:
        device = torch.device("cpu")
        core_logger.info(">>>> using CPU")
        gpu = False
    return device, gpu


def _to_device(x, device):
    """
    Converts the input tensor or numpy array to the specified device.

    Args:
        x (torch.Tensor or numpy.ndarray): The input tensor or numpy array.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The converted tensor on the specified device.
    """
    if not isinstance(x, torch.Tensor):
        X = torch.from_numpy(x).to(device, dtype=torch.float32)
        return X
    else:
        return x


def _from_device(X):
    """
    Converts a PyTorch tensor from the device to a NumPy array on the CPU.

    Args:
        X (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    x = X.detach().cpu().numpy()
    return x


def _forward(net, x):
    """Converts images to torch tensors, runs the network model, and returns numpy arrays.

    Args:
        net (torch.nn.Module): The network model.
        x (numpy.ndarray): The input images.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The output predictions (flows and cellprob) and style features.
    """
    X = _to_device(x, device=net.device)
    net.eval()
    with torch.no_grad():
        y, style = net(X)[:2]
    del X
    y = _from_device(y)
    style = _from_device(style)
    return y, style


def run_net(net, imgi, batch_size=8, augment=False, tile_overlap=0.1, bsize=224,
            rsz=None):
    """ 
    Run network on stack of images.
    
    (faster if augment is False)

    Args:
        net (class): cellpose network (model.net)
        imgi (np.ndarray): The input image or stack of images of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: outputs of network y and style. If tiled `y` is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability. 
            style is a 1D array of size 256 summarizing the style of the image, if tiled `style` is averaged over tiles.
    """
    # run network
    Lz, Ly0, Lx0, nchan = imgi.shape 
    if rsz is not None:
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0
    
    ly, lx = bsize, bsize
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(bsize, bsize))
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])
    
    if augment:
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
    else:
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
    
    
    # run multiple slices at the same time
    ntiles = ny * nx
    nimgs = max(1, batch_size // ntiles) # number of imgs to run in the same batch
    niter = int(np.ceil(Lz / nimgs))
    ziterator = (trange(niter, file=tqdm_out, mininterval=30) 
                    if niter > 10 or Lz > 1 else range(niter))
    for k in ziterator:
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32")
        for i, b in enumerate(inds):
            # pad image for net so Ly and Lx are divisible by 4
            imgb = transforms.resize_image(imgi[b], rsz=rsz) if rsz is not None else imgi[b].copy()
            imgb = np.pad(imgb.transpose(2,0,1), pads, mode="constant")
            IMG, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                imgb, bsize=bsize, augment=augment,
                tile_overlap=tile_overlap)
            IMGa[i * ntiles : (i+1) * ntiles] = np.reshape(IMG, 
                                            (ny * nx, nchan, ly, lx))
        
        # run network
        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            ya0, stylea0 = _forward(net, IMGa[bslc])
            if j == 0:
                nout = ya0.shape[1]
                ya = np.zeros((IMGa.shape[0], nout, ly, lx), "float32")
                stylea = np.zeros((IMGa.shape[0], 256), "float32")
            ya[bslc] = ya0
            stylea[bslc] = stylea0

        # average tiles
        for i, b in enumerate(inds):
            if i==0 and k==0:
                yf = np.zeros((Lz, nout, Ly, Lx), "float32")
                styles = np.zeros((Lz, 256), "float32")
            y = ya[i * ntiles : (i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, 3, ly, lx))
                y = transforms.unaugment_tiles(y)
                y = np.reshape(y, (-1, 3, ly, lx))
            yfi = transforms.average_tiles(y, ysub, xsub, Lyt, Lxt)
            yf[b] = yfi[:, :imgb.shape[-2], :imgb.shape[-1]]
            stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
            stylei /= (stylei**2).sum()**0.5
            styles[b] = stylei
    # slices from padding
    yf = yf[:, :, ypad1 : Ly-ypad2, xpad1 : Lx-xpad2]
    yf = yf.transpose(0,2,3,1)   
    return yf, np.array(styles)


def run_3D(net, imgs, batch_size=8, augment=False,
           tile_overlap=0.1, bsize=224, net_ortho=None,
           progress=None):
    """ 
    Run network on image z-stack.
    
    (faster if augment is False)

    Args:
        imgs (np.ndarray): The input image stack of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.
        net_ortho (class, optional): cellpose network for orthogonal ZY and ZX planes. Defaults to None.
        progress (QProgressBar, optional): pyqt progress bar. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: outputs of network y and style. If tiled `y` is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Z flow; y[...,1] is Y flow; y[...,2] is X flow; y[...,3] is cell probability. 
            style is a 1D array of size 256 summarizing the style of the image, if tiled `style` is averaged over tiles.
    """
    sstr = ["YX", "ZY", "ZX"]
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    cp = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = imgs.shape[:-1]
    yf = np.zeros((*shape, 4), "float32")
    for p in range(3):
        xsl = imgs.transpose(pm[p])
        # per image
        core_logger.info("running %s: %d planes of size (%d, %d)" %
                         (sstr[p], shape[pm[p][0]], shape[pm[p][1]], shape[pm[p][2]]))
        y, style = run_net(net,
                           xsl, batch_size=batch_size, augment=augment, 
                           bsize=bsize, tile_overlap=tile_overlap, 
                           rsz=None)
        yf[..., -1] += y[..., -1].transpose(ipm[p])
        for j in range(2):
            yf[..., cp[p][j]] += y[..., cpy[p][j]].transpose(ipm[p])
        y = None; del y
    
        if progress is not None:
            progress.setValue(25 + 15 * p)
    
    return yf, style
