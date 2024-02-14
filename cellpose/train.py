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
    """ loss function between true labels lbl and prediction y """
    criterion  = nn.MSELoss(reduction='mean')
    criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
    veci = 5. * torch.from_numpy(lbl[:,1:]).to(device)
    loss = criterion(y[:,:2], veci)
    loss /= 2.
    loss2 = criterion2(y[:,-1] , torch.from_numpy(lbl[:,0]>0.5).to(device).float())
    loss = loss + loss2
    return loss

def _get_batch(inds, data=None, labels=None, files=None, labels_files=None, 
              channels=None, channel_axis=None, normalize_params={"normalize": False}):
    if data is None:
        imgs = [io.imread(files[i]) for i in inds]
        if channels is not None:
            imgs = [transforms.convert_image(img, channels=channels, channel_axis=channel_axis) for img in imgs]
            imgs = [img.transpose(2,0,1) for img in imgs]
        if normalize_params["normalize"]:
            imgs = [transforms.normalize_img(img, normalize=normalize_params, axis=0) 
                    for img in imgs]
        lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i][1:] for i in inds]
    return imgs, lbls

def _reshape_norm(data, channels=None, channel_axis=None,
                 normalize_params={"normalize": False}):      
    if channels is not None:
        data = [transforms.convert_image(td, channels=channels, 
                                            channel_axis=channel_axis) for td in data]
        data = [td.transpose(2,0,1) for td in data]
    if normalize_params["normalize"]:
        data = [transforms.normalize_img(td, normalize=normalize_params, axis=0) 
                for td in data]
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
            td = td.transpose(2,0,1)
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
                        train_labels_files=None, train_probs=None,
                        test_data=None, test_labels=None, test_files=None, 
                        test_labels_files=None, test_probs=None, 
                        load_files=True, min_train_masks=5,
                        compute_flows=False,
                        channels=None, channel_axis=None,
                        normalize_params={"normalize": False},
                        device=torch.device("cuda")):

    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
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
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files, device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files, device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files), files=train_files, 
                                          device=device)
        train_labels_files = [os.path.splitext(str(tf))[0] + "_flows.tif" 
                                for tf in train_files]
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files), files=test_files, 
                                          device=device)
            test_labels_files = [os.path.splitext(str(tf))[0] + "_flows.tif" 
                                    for tf in test_files]
            
    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0] if train_labels is not None 
                else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train<5] = 5.
    if test_data is not None:
        diam_test = np.array([utils.diameters(test_labels[k][0])[0] 
                                for k in trange(len(test_labels))])
        diam_test[diam_test<5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([utils.diameters(io.imread(test_labels_files[k])[0])[0] 
                                for k in trange(len(test_labels_files))])
        diam_test[diam_test<5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set")
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
    train_probs = 1./nimg * np.ones(nimg, "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1./nimg_test * np.ones(nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    if channels is not None or normalize_params["normalize"]:
        if channels:
            train_logger.info(f">>> using channels {channels}")
        if normalize_params["normalize"]:
            train_logger.info(f">>> normalizing {normalize_params}")
        if train_data is not None:
            train_data = _reshape_norm(train_data, channels=channels, 
                            channel_axis=channel_axis, normalize_params=normalize_params)
        if test_data is not None:
            test_data = _reshape_norm(test_data, channels=channels, 
                            channel_axis=channel_axis, normalize_params=normalize_params)
        
    return (train_data, train_labels, train_files, train_labels_files, 
            train_probs, diam_train, test_data, test_labels, test_files, 
            test_labels_files, test_probs, diam_test) 

def train_seg(net, train_data=None, train_labels=None, 
              train_files=None, train_labels_files=None, train_probs=None,
              test_data=None, test_labels=None, 
              test_files=None, test_labels_files=None, test_probs=None, 
              load_files=True,
              batch_size=8, learning_rate = 0.005,
              n_epochs = 2000, weight_decay = 1e-5, momentum=0.9, SGD=False,
              channels=None, channel_axis=None, normalize=True, 
              compute_flows=False,
              save_path=None, save_every=100,
              nimg_per_epoch=None, nimg_test_per_epoch=None,
              rescale=True, min_train_masks=5,
              model_name=None):
    """ train net with images train_data 
    
        Parameters
        ------------------

        train_data: list of arrays (2D or 3D)
            images for training

        train_labels: list of arrays (2D or 3D)
            labels for train_data, where 0=no masks; 1,2,...=mask labels
            can include flows as additional images

        train_files: list of strings
            file names for images in train_data (to save flows for future runs)

        test_data: list of arrays (2D or 3D)
            images for testing

        test_labels: list of arrays (2D or 3D)
            labels for test_data, where 0=no masks; 1,2,...=mask labels; 
            can include flows as additional images
    
        test_files: list of strings
            file names for images in test_data (to save flows for future runs)

        channels: list of ints (default, None)
            channels to use for training

        normalize: bool or dictionary (default, True)
            normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

        save_path: string (default, None)
            where to save trained model, if None it is not saved

        save_every: int (default, 100)
            save network every [save_every] epochs

        learning_rate: float or list/np.ndarray (default, 0.2)
            learning rate for training, if list, must be same length as n_epochs

        n_epochs: int (default, 500)
            how many times to go through whole training set during training

        weight_decay: float (default, 0.00001)

        SGD: bool (default, True) 
            use SGD as optimization instead of RAdam

        batch_size: int (optional, default 8)
            number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)

        nimg_per_epoch: int (optional, default None)
            minimum number of images to train on per epoch, 
            with a small training set (< 8 images) it may help to set to 8

        rescale: bool (default, True)
            whether or not to rescale images to diam_mean during training, 
            if True it assumes you will fit a size model after training or resize your images accordingly,
            if False it will try to train the model to be scale-invariant (works worse)

        diameter: int (default, None)
            if not None, fixed diameter that is used to rescale all images 
            - resize factor is diam_mean / diameter

        min_train_masks: int (default, 5)
            minimum number of masks an image must have to use in training set

        model_name: str (default, None)
            name of network, otherwise saved with name as params + training start time

    """

    device = net.device

    scale_range = 0.5 if rescale else 1.0

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError('normalize parameter must be a bool or a dict')
    else:
        normalize_params = models.normalize_default
        normalize_params['normalize'] = normalize
    
    out = _process_train_test(train_data=train_data, train_labels=train_labels, train_files=train_files, 
                             train_labels_files=train_labels_files, train_probs=train_probs,
                             test_data=test_data, test_labels=test_labels, test_files=test_files, 
                             test_labels_files=test_labels_files, test_probs=test_probs, 
                             load_files=load_files, min_train_masks=min_train_masks,
                             compute_flows=compute_flows,
                             channels=channels, channel_axis=channel_axis,
                             normalize_params=normalize_params,
                             device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test) = out

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch
    
    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate*np.ones(max(0, n_epochs-10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1]/2 * np.ones(10))
    elif n_epochs > 100:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1]/2 * np.ones(5))
    n_epochs = len(LR)
    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")

    if not SGD:
        train_logger.info(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, 
                                      weight_decay=weight_decay)
    else:
        train_logger.info(f">>> SGD, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}")
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, 
                                    weight_decay=weight_decay, momentum=momentum)

    t0 = time.time()     
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    model_path =  save_path / "models" / model_name
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
            param_group['lr'] = LR[iepoch]
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k+batch_size, nimg)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels, 
                                   files=train_files, labels_files=train_labels_files, 
                                   channels=channels, channel_axis=channel_axis,
                                   normalize_params=normalize_params)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item()
            # augmentations
            imgi, lbl = transforms.random_rotate_and_resize(imgs, Y=lbls, 
                                                            rescale=rsc, 
                                                            scale_range=scale_range)[:2]

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
            
        if iepoch==5 or iepoch%10==0:
            lavgt = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test), 
                                            size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                for ibatch in range(0,len(rperm),batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch:ibatch+batch_size]
                        imgs, lbls = _get_batch(inds, data=test_data, labels=test_labels, 
                                   files=test_files, labels_files=test_labels_files, 
                                   channels=channels, channel_axis=channel_axis,
                                   normalize_params=normalize_params)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item()
                        imgi, lbl = transforms.random_rotate_and_resize(imgs, Y=lbls, 
                                                                        rescale=rsc, 
                                                                        scale_range=scale_range)[:2]
                        X = torch.from_numpy(imgi).to(device)    
                        y = net(X)[0]        
                        loss = _loss_fn_seg(lbl, y, device)
                        test_loss = loss.item()
                        test_loss *= len(imgi)        
                        lavgt += test_loss
                lavgt /= len(rperm)
            lavg /= nsum
            train_logger.info(f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.4f}, time {time.time()-t0:.2f}s")
            lavg, nsum = 0, 0

        if iepoch > 0 and iepoch % save_every==0:
            net.save_model(model_path)
    net.save_model(model_path)

    return model_path

def train_size(net, pretrained_model, train_data=None, train_labels=None, 
              train_files=None, train_labels_files=None, train_probs=None,
              test_data=None, test_labels=None, 
              test_files=None, test_labels_files=None, test_probs=None, 
              load_files=True, min_train_masks=5,
              channels=None, channel_axis=None, normalize=True, 
              nimg_per_epoch=None, nimg_test_per_epoch=None,
              batch_size=128, l2_regularization=1.0, n_epochs=10):
    """ train size model """
    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError('normalize parameter must be a bool or a dict')
    else:
        normalize_params = models.normalize_default
        normalize_params['normalize'] = normalize
    
    out = _process_train_test(train_data=train_data, train_labels=train_labels, train_files=train_files, 
                             train_labels_files=train_labels_files, train_probs=train_probs,
                             test_data=test_data, test_labels=test_labels, test_files=test_files, 
                             test_labels_files=test_labels_files, test_probs=test_probs, 
                             load_files=load_files, min_train_masks=min_train_masks,
                             compute_flows=False,
                             channels=channels, channel_axis=channel_axis,
                             normalize_params=normalize_params,
                             device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test) = out

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch
    
    diam_mean = net.diam_mean.item()
    device = net.device
    net.eval()

    styles = np.zeros((n_epochs*nimg_per_epoch, 256), np.float32)
    diams = np.zeros((n_epochs*nimg_per_epoch,), np.float32)
    tic = time.time()
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,), 
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for ibatch in range(0,nimg_per_epoch,batch_size):
            inds_batch = np.arange(ibatch, min(nimg_per_epoch, ibatch+batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels, 
                                    files=train_files, labels_files=train_labels_files, 
                                    channels=channels, channel_axis=channel_axis,
                                    normalize_params=normalize_params)
            diami = diam_train[inds].copy()
            imgi,lbl,scale = transforms.random_rotate_and_resize(imgs, scale_range=1, 
                                                                    xy=(512,512)) 
            imgi = torch.from_numpy(imgi).to(device)
            with torch.no_grad():
                feat = net(imgi)[1]
            indsi = inds_batch + nimg_per_epoch * iepoch
            styles[indsi] = feat.cpu().numpy()
            diams[indsi] = np.log(diami) - np.log(diam_mean) + np.log(scale)
        del feat
        train_logger.info('ran %d epochs in %0.3f sec'%(iepoch+1, time.time()-tic))

    l2_regularization = 1.

    # create model
    smean = styles.copy().mean(axis=0)
    X = ((styles.copy() - smean).T).copy()
    ymean = diams.copy().mean()
    y = diams.copy() - ymean

    A = np.linalg.solve(X@X.T + l2_regularization*np.eye(X.shape[0]), X @ y)
    ypred = A @ X

    train_logger.info('train correlation: %0.4f'%np.corrcoef(y, ypred)[0,1])
        
    if nimg_test:
        np.random.seed(0)
        styles_test = np.zeros((nimg_test_per_epoch, 256), np.float32)
        diams_test = np.zeros((nimg_test_per_epoch,), np.float32)
        diam_test = np.zeros((nimg_test_per_epoch,), np.float32)
        if nimg_test != nimg_test_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg_test), 
                                    size=(nimg_test_per_epoch,), p=test_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg_test))
        for ibatch in range(0,nimg_test_per_epoch,batch_size):
            inds_batch = np.arange(ibatch, min(nimg_test_per_epoch, ibatch+batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=test_data, labels=test_labels, 
                                    files=test_files, labels_files=test_labels_files, 
                                    channels=channels, channel_axis=channel_axis,
                                    normalize_params=normalize_params)
            diami = diam_test[inds].copy()
            imgi,lbl,scale = transforms.random_rotate_and_resize(imgs, Y=lbls, scale_range=1, 
                                                                    xy=(512,512)) 
            imgi = torch.from_numpy(imgi).to(device)
            diamt = np.array([utils.diameters(lbl0[0])[0] for lbl0 in lbl])
            diamt = np.maximum(5., diamt)
            with torch.no_grad():
                feat = net(imgi)[1]
            styles_test[inds_batch] = feat.cpu().numpy()
            diams_test[inds_batch] = np.log(diami) - np.log(diam_mean) + np.log(scale)
            diam_test[inds_batch] = diamt

        diam_test_pred = np.exp(A @ (styles_test - smean).T + np.log(diam_mean) + ymean)
        diam_test_pred = np.maximum(5., diam_test_pred)
        train_logger.info('test correlation: %0.4f'%np.corrcoef(diam_test, diam_test_pred)[0,1])

    pretrained_size = str(pretrained_model) + '_size.npy'
    params = {'A': A, 'smean': smean, 'diam_mean': diam_mean, 'ymean': ymean}
    np.save(pretrained_size, params)
    train_logger.info('model saved to '+pretrained_size)

    return params