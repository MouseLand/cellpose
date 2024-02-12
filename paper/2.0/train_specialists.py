"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, argparse, time
import numpy as np
from glob import glob
import torch
from torch import nn
from tqdm import trange, tqdm
from cellpose import models
from cellpose.io import logger_setup, imread, imsave
from cellpose import metrics, transforms, utils, resnet_torch
from datasets import reshape_and_normalize

def get_styles(pretrained_model, train_files, train_types, test_files, test_types,
                device=torch.device('cuda')):
    model = models.CellposeModel(pretrained_model=pretrained_model, gpu=True)
    device = torch.device('cuda')
    all_styles = []
    batch_size = 32

    for types, files in zip([train_types, test_types], [train_files, test_files]):
        nimg = len(files)
        batch_size = 32
        styles = np.zeros((nimg, 256), 'float32')
        #diam_type = np.array([17. if types[i][1]=='nuclei' else 30. for i in range(len(types))])
        model.net.eval()
        for ibatch in trange(0, nimg, batch_size):

            inds = np.arange(ibatch, min(nimg, ibatch+batch_size))

            #lbls = [imread(os.path.splitext(files[i])[0] + f'_flows.tif') for i in inds]
            imgs = [reshape_and_normalize(imread(files[i]), types[i]) for i in inds]
            masks = [imread(os.path.splitext(files[i])[0] + f'_masks.tif') for i in inds]
            diams = np.array([utils.diameters(lbl)[0] for lbl in masks])
            diams[diams < 5.] = 5.
            rsc = diams / 30. #diam_type[inds]

            #[data[i] if diam_type[i]==30. else data[i][::-1] for i in inds]
            
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                                    imgs, 
                                    Y = None,
                                    rescale=rsc, scale_range=0., 
                                    random_crop=False, 
                                    random_rotation=False,
                                    xy=(224, 224))
            imgi = torch.from_numpy(imgi).to(device)
            with torch.no_grad():
                style = model.net(imgi)[1]
            styles[inds] = style.cpu().numpy()
        all_styles.append(styles)

    return all_styles

def cluster_styles(train_files, train_types, train_styles, test_files, test_types, test_styles):
    from openTSNE import TSNE
    from sklearn.decomposition import PCA

    ctypes_train = -1*np.ones(len(train_styles), 'int')
    ctypes_train[np.array([(train_types[i][0] == 'cp') * (train_types[i][1] == 'cyto') for i in range(len(train_types))]).astype(bool)] = 0
    ctypes_train[np.array([train_types[i][0] == 'lc' for i in range(len(train_types))]).astype(bool)] = 1
    ctypes_train[np.array([train_types[i][0] == 'tn' for i in range(len(train_types))]).astype(bool)] = 2

    ctypes_test = -1*np.ones(len(test_styles), 'int')
    ctypes_test[np.array([(test_types[i][0] == 'cp') * (test_types[i][1] == 'cyto') for i in range(len(test_types))]).astype(bool)] = 0
    ctypes_test[np.array([test_types[i][0] == 'lc' for i in range(len(test_types))]).astype(bool)] = 1
    ctypes_test[np.array([test_types[i][0] == 'tn' for i in range(len(test_types))]).astype(bool)] = 2

    inds_sub = np.nonzero(ctypes_train==0)[0]
    inds_sub = np.append(inds_sub, np.nonzero(ctypes_train==1)[0][::5])
    inds_sub = np.append(inds_sub, np.nonzero(ctypes_train==2)[0][::5])

    train_styles_sub = train_styles[inds_sub]
    ctypes_sub = ctypes_train[inds_sub]
    train_files_sub = np.array(train_files)[inds_sub]
    train_types_sub = np.array(train_types)[inds_sub]

    U = PCA(n_components=2).fit_transform(train_styles_sub)
    tsne = TSNE(
                perplexity=30,
                metric='cosine',
                n_jobs=8,
                random_state=42,
                verbose = True,
                n_components = 2,
                initialization = .0001 * U,
            )
    embeddingOPENTSNE = tsne.fit(train_styles_sub)

    import scanpy as sc
    adata = sc.AnnData(train_styles_sub)
    sc.pp.neighbors(adata, n_neighbors=100, use_rep='X')

    sc.tl.leiden(adata, resolution=0.45)
    leiden_labels = np.array(adata.obs['leiden'].astype(int))

    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=5).fit(train_styles_sub, leiden_labels)
    train_labels = classifier.predict(train_styles)
    test_labels = classifier.predict(test_styles)

    return train_labels, test_labels, embeddingOPENTSNE, leiden_labels


def train_style_net(root, style_path, 
                    pretrained_model = None,
                    diam_mean=30.,
                    batch_size = 8,
                    flow_threshold = 0.4,
                    weight_decay = 1e-5,
                    n_epochs = 400,
                    learning_rate = 0.2, 
                    istyle=0,
                    save_results=True,
                    use_gpu=True):
    
    save_path = root

    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate*np.ones(n_epochs-10))
    if n_epochs > 100:
        for i in range(10):
            LR = np.append(LR, LR[-1]/2 * np.ones(10))
    n_epochs = len(LR)

    style_name = os.path.splitext(os.path.split(style_path)[1])[0]
    dat = np.load(style_path, allow_pickle=True).item()
    train_files = dat['train_files']
    train_types = dat['train_types']
    train_labels = dat['train_labels']
    test_files = dat['test_files']
    test_types = dat['test_types']
    test_labels = dat['test_labels']
    if train_files[0][:len(root)] != root:
        for i in range(len(train_files)):
            orig_root = os.path.split(os.path.split(os.path.split(train_files[i])[0])[0])[0]
            new_path = root + train_files[i][len(orig_root):]
            if i==0:
                print(f'changing path from {train_files[i]} to {new_path}')
            train_files[i] = new_path
            
        for i in range(len(test_files)):
            orig_root = os.path.split(os.path.split(os.path.split(test_files[i])[0])[0])[0]
            new_path = root + test_files[i][len(orig_root):]
            test_files[i] = new_path

    nimg_per_epoch = batch_size

    netstr = f'scratch_{style_name}_{istyle}'
    print(netstr)
    model = models.CellposeModel(pretrained_model=pretrained_model, gpu=use_gpu, 
                                 diam_mean=diam_mean)

    itrain = np.nonzero(train_labels==istyle)[0]

    train_istyle = [reshape_and_normalize(imread(train_files[i]), train_types[i]) 
                    for i in tqdm(itrain)]

    lbls_istyle = [imread(os.path.splitext(train_files[i])[0] + f'_flows.tif') for i in tqdm(itrain)]
    model_path = model.train(train_istyle, lbls_istyle, train_files=None,
                            channels=[1,2], normalize=True, 
                            save_path=save_path, save_every=n_epochs,
                            weight_decay=weight_decay, n_epochs=n_epochs, 
                            learning_rate=LR, momentum=0.9,
                            batch_size=batch_size, nimg_per_epoch=nimg_per_epoch,
                            min_train_masks=10,
                            netstr=netstr)

    itest = np.nonzero(test_labels==istyle)[0]
    test_istyle = [reshape_and_normalize(imread(test_files[i]), test_types[i]) 
                for i in tqdm(itest)]
    masks_istyle = [imread(os.path.splitext(test_files[i])[0] + f'_masks.tif') for i in itest]

    diams = np.array([utils.diameters(lbl)[0] for lbl in masks_istyle])
    diams[diams < 5.] = 5.

    rescale = diam_mean / diams

    masks, flows, styles = model.eval(test_istyle, channels=[1,2], rescale=rescale, 
                                resample=True, net_avg=False, batch_size=32,
                                tile_overlap=0.1, diameter=diam_mean,
                                flow_threshold=flow_threshold)

    threshold = np.arange(0.5, 1.0, 0.05)
    ap,tp,fp,fn = metrics.average_precision(masks_istyle, masks, threshold=threshold)    
    print(ap[:,[0, 5, 8]].mean(axis=0))

    if save_results: 
        save_ap_path = os.path.join(save_path, f'models/{netstr}_AP_TP_FP_FN.npy')
        print(f'saving results to {save_ap_path}')
        np.save(save_ap_path, {'threshold':threshold, 'ap':ap,
                                                                     'tp':tp, 'fp':fp, 'fn':fn, 
                                                                     'test_files': [test_files[i] for i in itest], 
                                                                     })
        for k,i in enumerate(itest):
            fname = os.path.splitext(test_files[i])[0]
            imsave(fname + '_' + netstr + '_masks.tif', masks[k])

    return model_path

def loss_fn(lbl, y, device):
    """ loss function between true labels lbl and prediction y """
    criterion  = nn.MSELoss(reduction='mean')
    criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
    veci = 5. * torch.from_numpy(lbl[:,1:]).to(device)
    loss = criterion(y[:,:2], veci)
    loss /= 2.
    loss2 = criterion2(y[:,-1] , torch.from_numpy(lbl[:,0]>0.5).to(device).float())
    loss = loss + loss2
    return loss

def train_general_net(root, style_path, 
                    pretrained_model = None,
                    diam_mean=30.,
                    batch_size = 8,
                    flow_threshold = 0.4,
                    weight_decay = 1e-5,
                    n_epochs = 400,
                    learning_rate = 0.2, 
                    istyle=0,
                    save_results=True,
                    use_gpu=True,
                    style_sampling=True,
                    ):
    
    if use_gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    save_path = root

    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate*np.ones(n_epochs-10))
    if n_epochs > 100:
        for i in range(10):
            LR = np.append(LR, LR[-1]/2 * np.ones(10))
    n_epochs = len(LR)

    style_name = os.path.splitext(os.path.split(style_path)[1])[0]
    dat = np.load(style_path, allow_pickle=True).item()
    train_files = dat['train_files']
    train_types = dat['train_types']
    train_labels = dat['train_labels']
    test_files = dat['test_files']
    test_types = dat['test_types']
    test_labels = dat['test_labels']
    if train_files[0][:len(root)] != root:
        for i in range(len(train_files)):
            orig_root = os.path.split(os.path.split(os.path.split(train_files[i])[0])[0])[0]
            new_path = root + train_files[i][len(orig_root):]
            if i==0:
                print(f'changing path from {train_files[i]} to {new_path}')
            train_files[i] = new_path
            
        for i in range(len(test_files)):
            orig_root = os.path.split(os.path.split(os.path.split(test_files[i])[0])[0])[0]
            new_path = root + test_files[i][len(orig_root):]
            test_files[i] = new_path

    if not style_sampling:
        ctypes = -1*np.ones(len(train_types), 'int')
        ctypes[np.array([(train_types[i][0] == 'cp') * (train_types[i][1] == 'cyto') for i in range(len(train_types))]).astype(bool)] = 0
        ctypes[np.array([train_types[i][0] == 'lc' for i in range(len(train_types))]).astype(bool)] = 1
        ctypes[np.array([train_types[i][0] == 'tn' for i in range(len(train_types))]).astype(bool)] = 2
        sample_labels = ctypes
        sample_probs = np.array([0.6, 0.2, 0.2])
        sample_probs = sample_probs[sample_labels]
        sample_probs /= sample_probs.sum()
        n_classes = sample_labels.max() + 1
        nimg_per_epoch = 200 * n_classes
    else:
        n_counts = np.unique(train_labels, return_counts=True)[1]
        sample_probs = n_counts.sum() / n_counts[train_labels]
        sample_probs /= sample_probs.sum()
        sample_labels = train_labels.copy()
        n_classes = sample_labels.max() + 1
        nimg_per_epoch = 100 * n_classes
        #nimg_per_epoch = 5 * n_classes
        
    nbase = [2, 32, 64, 128, 256]
    nout = 3
    net = resnet_torch.CPnet(nbase=nbase, nout=nout, sz=3).to(device)
    if pretrained_model is not None:
        net.load_model(pretrained_model, device=device)
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr = learning_rate, 
                                weight_decay = weight_decay, 
                                momentum = 0.9)
    nimg = len(train_files)  
    t0 = time.time()     
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,), p=sample_probs)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR[iepoch]
        
        lavg = 0
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k+batch_size, nimg)
            inds = rperm[k:kend]
            
            imgs = [reshape_and_normalize(imread(train_files[i]), train_types[i]) for i in inds]
            lbls = [imread(os.path.splitext(train_files[i])[0] + f'_flows.tif') for i in inds]
            diams = np.array([utils.diameters(lbl[0])[0] for lbl in lbls])
            
            rsc = diams / diam_mean
            
            imgi, lbl, scale = transforms.random_rotate_and_resize(imgs, Y=lbls,
                                                                rescale=rsc, 
                                                                scale_range=.5)
            
            optimizer.zero_grad()
            X = torch.from_numpy(imgi).to(device)    
            y = net(X)[0]
            
            loss = loss_fn(lbl, y, device)
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            train_loss *= len(imgi)        
            lavg += train_loss
            
        lavg = lavg/len(rperm)
        print(f'{iepoch}, train_loss= {lavg:.4f}, LR={LR[iepoch]:.4f}, time {time.time()-t0:.2f}s')
    
    if style_sampling:
        model_path = f'models/general_{style_name}_sampling'
    else:
        model_path = 'models/cp_general_60_20_20'
    net.save_model(model_path)

def eval_model(model_path, test_files, test_types, use_gpu=True,
                flow_threshold=0.4, diameter=30.):

    logger_setup();
    model = models.CellposeModel(pretrained_model=model_path, diam_mean=diameter,
                             gpu=use_gpu)
    print(model.pretrained_model)
    test_data = [reshape_and_normalize(imread(test_files[i]), test_types[i]) 
                    for i in trange(len(test_files))]
    masks_data = [imread(os.path.splitext(test_files[i])[0] + f'_masks.tif') 
                for i in  trange(len(test_files))]

    diams = np.array([utils.diameters(lbl)[0] for lbl in masks_data])
    diams[diams < 5.] = 5.
    #diam_type = np.array([17. if test_types[i][1]=='nuclei' else 30. for i in itest])

    rescale = diameter / diams

    masks, flows, styles = model.eval(test_data, channels=[1,2], rescale=rescale, 
                                resample=True, net_avg=False, batch_size=32,
                                normalize=False,
                                tile_overlap=0.1,
                                flow_threshold=flow_threshold)[:3]

    threshold = np.arange(0.5, 1.0, 0.05)
    ap,tp,fp,fn = metrics.average_precision(masks_data, masks, threshold=threshold)    
    print(ap[:,[0, 5, 8]].mean(axis=0))

    return masks, ap, tp, fp, fn, threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
        
    # settings for locating and formatting images
    parser.add_argument('--root', default=[], type=str, help='root folder containing data to train and test on.')
    parser.add_argument('--style_path', default=[], type=str, help='style data path')
    parser.add_argument('--style_type', default=0, type=int, help='style type to run net on')
    parser.add_argument('--general', default=0, type=int, help='train general model')
    parser.add_argument('--start_pretrained', default=0, type=int, help='start training from pretrained model')
    parser.add_argument('--use_gpu', default=1, type=int, help='use gpu')
    parser.add_argument('--n_epochs', default=400, type=int, help='n_epochs')
    parser.add_argument('--diam_mean', default=30, type=int, help='mean diameter to resize cells to')

    args = parser.parse_args()
    logger,log_path=logger_setup()

    if not args.general:
        train_style_net(args.root, args.style_path, 
                        istyle=args.style_type, use_gpu=args.use_gpu,
                        diam_mean=args.diam_mean, n_epochs=args.n_epochs)
    else:
        train_general_net(args.root, args.style_path, use_gpu=args.use_gpu,
                          diam_mean=args.diam_mean, n_epochs=args.n_epochs)
