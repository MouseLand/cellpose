import time
import os
import numpy as np
from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch
from cellpose.transforms import normalize_img
from pathlib import Path
import torch
from torch import nn
from tqdm import trange
from numba import prange

import logging

train_logger = logging.getLogger(__name__)


def _loss_fn_seg(lbl, y, device):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * torch.from_numpy(lbl[:, 1:]).to(device)
    loss = criterion(y[:, :2], veci)
    loss /= 2.
    loss2 = criterion2(y[:, -1], torch.from_numpy(lbl[:, 0] > 0.5).to(device).float())
    loss = loss + loss2
    return loss


def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               channels=None, channel_axis=None, rgb=False,
               normalize_params={"normalize": False}):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        channels (list or None): List of channel indices to extract from images.
        channel_axis (int or None): Axis along which the channels are located.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).

    Returns:
        tuple: A tuple containing two lists: the batch of images and the batch of labels.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, channels=channels, channel_axis=channel_axis,
                             rgb=rgb, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i][1:] for i in inds]
    return imgs, lbls


def pad_to_rgb(img):
    if img.ndim == 2 or np.ptp(img[1]) < 1e-3:
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        img = np.tile(img[:1], (3, 1, 1))
    elif img.shape[0] < 3:
        nc, Ly, Lx = img.shape
        # randomly flip channels
        if np.random.rand() > 0.5:
            img = img[::-1]
        # randomly insert blank channel
        ic = np.random.randint(3)
        img = np.insert(img, ic, np.zeros((3 - nc, Ly, Lx), dtype=img.dtype), axis=0)
    return img


def convert_to_rgb(img):
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
        img = np.tile(img, (3, 1, 1))
    elif img.shape[0] < 3:
        img = img.mean(axis=0, keepdims=True)
        img = transforms.normalize99(img)
        img = np.tile(img, (3, 1, 1))
    return img


def _reshape_norm(data, channels=None, channel_axis=None, rgb=False,
                  normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data.
        channels (int or list, optional): Number of channels or list of channel indices to keep. Defaults to None.
        channel_axis (int, optional): Axis along which the channels are located. Defaults to None.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if channels is not None or channel_axis is not None:
        data = [
            transforms.convert_image(td, channels=channels, channel_axis=channel_axis)
            for td in data
        ]
        data = [td.transpose(2, 0, 1) for td in data]
    if normalize_params["normalize"]:
        data = [
            transforms.normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    if rgb:
        data = [pad_to_rgb(td) for td in data]
    return data


def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}):
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = transforms.convert_image(td, channels=channels,
                                          channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = transforms.normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new
    # else:
    #     train_files = reshape_norm_save(train_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)
    # elif test_files is not None:
    #     test_files = reshape_norm_save(test_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, channels=None, channel_axis=None,
                        rgb=False, normalize_params={"normalize": False
                                                    }, device=torch.device("cuda")):
    """
    Process train and test data.

    Args:
        train_data (list or None): List of training data arrays.
        train_labels (list or None): List of training label arrays.
        train_files (list or None): List of training file paths.
        #train_labels_files (list or None): List of training label file paths.
        train_probs (ndarray or None): Array of training probabilities.
        test_data (list or None): List of test data arrays.
        test_labels (list or None): List of test label arrays.
        test_files (list or None): List of test file paths.
        #test_labels_files (list or None): List of test label file paths.
        test_probs (ndarray or None): Array of test probabilities.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files),
                                          files=train_files, device=device)
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files),
                                              files=test_files, device=device)

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if channels is not None or normalize_params["normalize"]:
        if channels:
            train_logger.info(f">>> using channels {channels}")
        if normalize_params["normalize"]:
            train_logger.info(f">>> normalizing {normalize_params}")
        if train_data is not None:
            train_data = _reshape_norm(train_data, channels=channels,
                                       channel_axis=channel_axis, rgb=rgb,
                                       normalize_params=normalize_params)
            normed = True
        if test_data is not None:
            test_data = _reshape_norm(test_data, channels=channels,
                                      channel_axis=channel_axis, rgb=rgb,
                                      normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, load_files=True, batch_size=8, learning_rate=0.005,
              n_epochs=2000, weight_decay=1e-5, momentum=0.9, SGD=False, channels=None,
              channel_axis=None, rgb=False, normalize=True, compute_flows=False,
              save_path=None, save_every=100, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=True, scale_range=None, bsize=224,
              min_train_masks=5, model_name=None):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train.
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Boolean - whether to use SGD as optimization instead of RAdam. Defaults to False.
        channels (List[int], optional): List of ints - channels to use for training. Defaults to None.
        channel_axis (int, optional): Integer - axis of the channel dimension in the input data. Defaults to None.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to True.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.

    Returns:
        Path: path to saved model weights
    """
    device = net.device

    scale_range0 = 0.5 if rescale else 1.0
    scale_range = scale_range if scale_range is not None else scale_range0

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channels=channels,
                              channel_axis=channel_axis, rgb=rgb,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channels": channels,
            "channel_axis": channel_axis,
            "rgb": rgb
        }

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 100:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")

    if not SGD:
        train_logger.info(
            f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
    else:
        train_logger.info(
            f">>> SGD, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}"
        )
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay, momentum=momentum)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    model_path = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {model_path}")

    lavg, nsum = 0, 0
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, labels_files=train_labels_files,
                                    **kwargs)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
            # augmentations
            imgi, lbl = transforms.random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                            scale_range=scale_range,
                                                            xy=(bsize, bsize))[:2]

            X = torch.from_numpy(imgi).to(device)
            y = net(X)[0]
            loss = _loss_fn_seg(lbl, y, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_loss *= len(imgi)
            lavg += train_loss
            nsum += len(imgi)

        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test),
                                             size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch:ibatch + batch_size]
                        imgs, lbls = _get_batch(inds, data=test_data,
                                                labels=test_labels, files=test_files,
                                                labels_files=test_labels_files,
                                                **kwargs)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(
                            len(diams), "float32")
                        imgi, lbl = transforms.random_rotate_and_resize(
                            imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                            xy=(bsize, bsize))[:2]
                        X = torch.from_numpy(imgi).to(device)
                        y = net(X)[0]
                        loss = _loss_fn_seg(lbl, y, device)
                        test_loss = loss.item()
                        test_loss *= len(imgi)
                        lavgt += test_loss
                lavgt /= len(rperm)
            lavg /= nsum
            train_logger.info(
                f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.4f}, time {time.time()-t0:.2f}s"
            )
            lavg, nsum = 0, 0

        if iepoch > 0 and iepoch % save_every == 0:
            net.save_model(model_path)
    net.save_model(model_path)

    return model_path


def train_size(net, pretrained_model, train_data=None, train_labels=None,
               train_files=None, train_labels_files=None, train_probs=None,
               test_data=None, test_labels=None, test_files=None,
               test_labels_files=None, test_probs=None, load_files=True,
               min_train_masks=5, channels=None, channel_axis=None, rgb=False,
               normalize=True, nimg_per_epoch=None, nimg_test_per_epoch=None,
               batch_size=64, scale_range=1.0, bsize=512, l2_regularization=1.0,
               n_epochs=10):
    """Train the size model.

    Args:
        net (object): The neural network model.
        pretrained_model (str): The path to the pretrained model.
        train_data (numpy.ndarray, optional): The training data. Defaults to None.
        train_labels (numpy.ndarray, optional): The training labels. Defaults to None.
        train_files (list, optional): The training file paths. Defaults to None.
        train_labels_files (list, optional): The training label file paths. Defaults to None.
        train_probs (numpy.ndarray, optional): The training probabilities. Defaults to None.
        test_data (numpy.ndarray, optional): The test data. Defaults to None.
        test_labels (numpy.ndarray, optional): The test labels. Defaults to None.
        test_files (list, optional): The test file paths. Defaults to None.
        test_labels_files (list, optional): The test label file paths. Defaults to None.
        test_probs (numpy.ndarray, optional): The test probabilities. Defaults to None.
        load_files (bool, optional): Whether to load files. Defaults to True.
        min_train_masks (int, optional): The minimum number of training masks. Defaults to 5.
        channels (list, optional): The channels. Defaults to None.
        channel_axis (int, optional): The channel axis. Defaults to None.
        normalize (bool or dict, optional): Whether to normalize the data. Defaults to True.
        nimg_per_epoch (int, optional): The number of images per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): The number of test images per epoch. Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 64.
        l2_regularization (float, optional): The L2 regularization factor. Defaults to 1.0.
        n_epochs (int, optional): The number of epochs. Defaults to 10.

    Returns:
        dict: The trained size model parameters.
    """
    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(
        train_data=train_data, train_labels=train_labels, train_files=train_files,
        train_labels_files=train_labels_files, train_probs=train_probs,
        test_data=test_data, test_labels=test_labels, test_files=test_files,
        test_labels_files=test_labels_files, test_probs=test_probs,
        load_files=load_files, min_train_masks=min_train_masks, compute_flows=False,
        channels=channels, channel_axis=channel_axis, normalize_params=normalize_params,
        device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out

    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channels": channels,
            "channel_axis": channel_axis,
            "rgb": rgb
        }

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    diam_mean = net.diam_mean.item()
    device = net.device
    net.eval()

    styles = np.zeros((n_epochs * nimg_per_epoch, 256), np.float32)
    diams = np.zeros((n_epochs * nimg_per_epoch,), np.float32)
    tic = time.time()
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for ibatch in range(0, nimg_per_epoch, batch_size):
            inds_batch = np.arange(ibatch, min(nimg_per_epoch, ibatch + batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, **kwargs)
            diami = diam_train[inds].copy()
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                imgs, scale_range=scale_range, xy=(bsize, bsize))
            imgi = torch.from_numpy(imgi).to(device)
            with torch.no_grad():
                feat = net(imgi)[1]
            indsi = inds_batch + nimg_per_epoch * iepoch
            styles[indsi] = feat.cpu().numpy()
            diams[indsi] = np.log(diami) - np.log(diam_mean) + np.log(scale)
        del feat
        train_logger.info("ran %d epochs in %0.3f sec" %
                          (iepoch + 1, time.time() - tic))

    l2_regularization = 1.

    # create model
    smean = styles.copy().mean(axis=0)
    X = ((styles.copy() - smean).T).copy()
    ymean = diams.copy().mean()
    y = diams.copy() - ymean

    A = np.linalg.solve(X @ X.T + l2_regularization * np.eye(X.shape[0]), X @ y)
    ypred = A @ X

    train_logger.info("train correlation: %0.4f" % np.corrcoef(y, ypred)[0, 1])

    if nimg_test:
        np.random.seed(0)
        styles_test = np.zeros((nimg_test_per_epoch, 256), np.float32)
        diams_test = np.zeros((nimg_test_per_epoch,), np.float32)
        diams_test0 = np.zeros((nimg_test_per_epoch,), np.float32)
        if nimg_test != nimg_test_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg_test),
                                     size=(nimg_test_per_epoch,), p=test_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg_test))
        for ibatch in range(0, nimg_test_per_epoch, batch_size):
            inds_batch = np.arange(ibatch, min(nimg_test_per_epoch,
                                               ibatch + batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=test_data, labels=test_labels,
                                    files=test_files, labels_files=test_labels_files,
                                    **kwargs)
            diami = diam_test[inds].copy()
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                imgs, Y=lbls, scale_range=scale_range, xy=(bsize, bsize))
            imgi = torch.from_numpy(imgi).to(device)
            diamt = np.array([utils.diameters(lbl0[0])[0] for lbl0 in lbl])
            diamt = np.maximum(5., diamt)
            with torch.no_grad():
                feat = net(imgi)[1]
            styles_test[inds_batch] = feat.cpu().numpy()
            diams_test[inds_batch] = np.log(diami) - np.log(diam_mean) + np.log(scale)
            diams_test0[inds_batch] = diamt

        diam_test_pred = np.exp(A @ (styles_test - smean).T + np.log(diam_mean) + ymean)
        diam_test_pred = np.maximum(5., diam_test_pred)
        train_logger.info("test correlation: %0.4f" %
                          np.corrcoef(diams_test0, diam_test_pred)[0, 1])

    pretrained_size = str(pretrained_model) + "_size.npy"
    params = {"A": A, "smean": smean, "diam_mean": diam_mean, "ymean": ymean}
    np.save(pretrained_size, params)
    train_logger.info("model saved to " + pretrained_size)

    return params
