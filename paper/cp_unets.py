import sys, os, time, string, shutil
from natsort import natsorted
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import matplotlib.pyplot as plt
from matplotlib import rc
import cv2
from scipy import stats
from cellpose import models, datasets, utils, transforms, io, metrics

thresholds = np.arange(0.5, 1.05, 0.05)

def get_pretrained_models(model_root, unet=0, nclass=3, residual=1, style=1, concatenate=0):
    sstr = ['off', 'on']
    model_str = model_root
    if unet:
        model_str += 'unet' + str(nclass) 
    else:
        model_str += 'cellpose'
    model_str += '_residual_' + sstr[residual]
    model_str += '_style_' + sstr[style]
    model_str += '_concatenation_' + sstr[concatenate]
    model_paths = natsorted(glob(model_str + '*'))
    removed_npy = [x for x in model_paths if not x.endswith('.npy')]
    model_paths = removed_npy
    return model_paths

def make_kfold_data(data_root):
    """ uses cellpose images and splits into 9 folds """
    ntest=68
    ntrain=540
    train_root = os.path.join(data_root, 'train/')
    imgs = [os.path.join(train_root, '%03d_img.tif'%i) for i in range(ntrain)]
    labels = [os.path.join(train_root, '%03d_masks.tif'%i) for i in range(ntrain)]
    flow_labels = [os.path.join(train_root, '%03d_img_flows.tif'%i) for i in range(ntrain)]

    test_root = os.path.join(data_root, 'test/')
    imgs = [os.path.join(test_root, '%03d_img.tif'%i) for i in range(ntest)]
    labels = [os.path.join(test_root, '%03d_masks.tif'%i) for i in range(ntest)]
    flow_labels = [os.path.join(test_root, '%03d_img_flows.tif'%i) for i in range(ntest)]

    all_inds = np.hstack((np.random.permutation(ntrain), np.random.permutation(ntest)+ntrain))
    for j in range(9):
        root_train = os.path.join(data_root, 'train%d/'%j)
        root_test = os.path.join(data_root, 'test%d/'%j)
        os.makedirs(root_train, exist_ok=True)
        os.makedirs(root_test, exist_ok=True)
        train_inds = all_inds[np.arange(j*ntest, j*ntest+ntrain, 1, int)%ntot]
        test_inds = all_inds[np.arange(j*ntest+ntrain, (j+1)*ntest+ntrain, 1, int)%ntot]
        for i,ind in enumerate(train_inds):
            shutil.copyfile(imgs[ind], os.path.join(root_train, '%03d_img.tif'%i))
            shutil.copyfile(labels[ind], os.path.join(root_train, '%03d_masks.tif'%i))
            shutil.copyfile(flow_labels[ind], os.path.join(root_train, '%03d_img_flows.tif'%i))
        for i,ind in enumerate(test_inds):
            shutil.copyfile(imgs[ind], os.path.join(root_test, '%03d_img.tif'%i))
            shutil.copyfile(labels[ind], os.path.join(root_test, '%03d_masks.tif'%i))
            shutil.copyfile(flow_labels[ind], os.path.join(root_test, '%03d_img_flows.tif'%i))

def train_unets(data_root):
    """ train unets with 3 or 2 classes and different architectures (12 networks total) """
    # can also run on command line for GPU cluster
    # python -m cellpose --train --use_gpu --dir images_cyto/train/ --test_dir images_cyto/test/ --img_filter _img --pretrained_model None --chan 2 --chan2 1 --unet "$1" --nclasses "$2" --learning_rate "$3" --residual_on "$4" --style_on "$5" --concatenation "$6"
    device = mx.gpu()
    ntest = len(glob(os.path.join(data_root, 'test/*_img.tif')))
    ntrain = len(glob(os.path.join(data_root, 'train/*_img.tif')))
    channels = [2,1]

    concatenation = [1, 1, 0]
    residual_on = [0, 0, 1]
    style_on = [0, 0, 1]
    nclasses = [3, 2, 3]

    # load images
    train_root = os.path.join(data_root, 'train/')
    train_data = [io.imread(os.path.join(train_root, '%03d_img.tif'%i)) for i in range(ntrain)]
    train_labels = [io.imread(os.path.join(train_root, '%03d_masks.tif'%i)) for i in range(ntrain)]
    
    test_root = os.path.join(data_root, 'test/')
    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]

    # train networks
    for k in range(len(concatenation)):
        # 4 nets for each
        for l in range(4):
            model = models.UnetModel(device=device,
                                        pretrained_model=None, 
                                        diam_mean=30,
                                        residual_on=residual_on[k],
                                        style_on=style_on[k],
                                        concatenation=concatenation[k],
                                        nclasses=nclasses[k])
            model.train(train_data, train_labels, test_data, test_labels, 
                        channels=channels, rescale=True,
                        save_path=train_root)

def test_unets_main(data_root, save_root):
    """ data_root is folder with folders images_.../train/models/, images_cyto/test and images_nuclei/test """
    #model_types = ['cyto', 'cyto_sp', 'nuclei']
    model_types = ['cyto_sp', 'nuclei']
    for model_type in model_types:
        model_root = os.path.join(data_root, 'images_%s/train/models/'%model_type)
        test_root = os.path.join(data_root, 'images_%s/test/'%model_type.split('_')[0])
        test_unets(model_root, test_root, save_root, model_type)

def test_unets(model_root, test_root, save_root, model_type='cyto'):
    """ test trained unets """
    device=mx.gpu()
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    if model_type[:4]=='cyto':
        channels = [2,1]
    else:
        channels = [0,0]

    concatenation = [1, 1, 0]
    residual_on = [0, 0, 1]
    style_on = [0, 0, 1]
    nclasses = [3, 2, 3]
    sstr = ['off', 'on']

    aps = np.zeros((len(concatenation),ntest,len(thresholds)))

    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
    
    if model_type!='cyto_sp':
        dat = np.load(os.path.join(test_root, 'predicted_diams.npy'), allow_pickle=True).item()
        if model_type=='cyto':
            rescale = 30. / dat['predicted_diams']
        else:
            rescale = 17. / dat['predicted_diams']
    else:
        rescale = np.ones(len(test_data))


    for k in range(1):#len(concatenation)):
        pretrained_models = get_pretrained_models(model_root, 1, nclasses[k], 
                                                    residual_on[k], style_on[k], 
                                                    concatenation[k])
        print(pretrained_models)
        model = models.UnetModel(device=device,
                                    pretrained_model=pretrained_models)
        
        
        masks = model.eval(test_data, channels=channels, rescale=rescale, net_avg=True)[0]
        ap = metrics.average_precision(test_labels, masks, 
                                        threshold=thresholds)[0]
        print(ap[:,[0,5,8]].mean(axis=0))
        aps[k] = ap
        np.save(os.path.join(save_root, 
                'unet%d_residual_%s_style_%s_concatenation_%s_%s_masks.npy'%(nclasses[k], sstr[residual_on[k]],
                                                                          sstr[style_on[k]], sstr[concatenation[k]], model_type)),
                masks)

def train_cellpose_nets(data_root):
    """ train networks on 9-folds of data (180 networks total) ... ~1 week on one GPU """
    # can also run on command line for GPU cluster
    # python -m cellpose --train --use_gpu --dir images_cyto/train"$7"/ --test_dir images_cyto/test"$7"/ --img_filter _img --pretrained_model None --chan 2 --chan2 1 --unet "$1" --nclasses "$2" --learning_rate "$3" --residual_on "$4" --style_on "$5" --concatenation "$6"
    device = mx.gpu()
    ntest=68
    ntrain=540
    concatenation = [0, 0, 0, 1, 1]
    residual_on = [1, 1, 0, 1, 0]
    style_on = [1, 0, 1, 1, 0]
    channels = [2,1]

    for j in range(9):
        # load images
        train_root = os.path.join(data_root, 'train%d/'%j)
        train_data = [io.imread(os.path.join(train_root, '%03d_img.tif'%i)) for i in range(ntrain)]
        train_labels = [io.imread(os.path.join(train_root, '%03d_masks.tif'%i)) for i in range(ntrain)]
        train_flow_labels = [io.imread(os.path.join(train_root, '%03d_img_flows.tif'%i)) for i in range(ntrain)]
        train_labels = [np.concatenate((train_labels[i][np.newaxis,:,:], train_flow_labels), axis=0) 
                                   for i in range(ntrain)]
        test_root = os.path.join(data_root, 'test%d/'%j)
        test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
        test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
        test_flow_labels = [io.imread(os.path.join(test_root, '%03d_img_flows.tif'%i)) for i in range(ntest)]
        test_labels = [np.concatenate((test_labels[i][np.newaxis,:,:], test_flow_labels), axis=0) 
                                   for i in range(ntest)]

        # train networks
        for k in range(len(concatenation)):
            # 4 nets for each
            for l in range(4):
                model = models.CellposeModel(device=device,
                                            pretrained_model=None, 
                                            diam_mean=30,
                                            residual_on=residual_on[k],
                                            style_on=style_on[k],
                                            concatenation=concatenation[k])
                model.train(images, labels, test_data=test_images, test_labels=test_labels, 
                            channels=channels, rescale=True,
                            save_path=train_root)

                # train size network on default network once
                if k==0 and l==0:
                    sz_model = models.SizeModel(model, device=device)
                    sz_model.train(train_data, train_labels, test_data, test_labels, channels=channels)
                    
                    predicted_diams, diams_style = sz_model.eval(test_data, channels=channels)
                    tlabels = [lbl[0] for lbl in test_labels]
                    ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
                    cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
                    print('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
                    np.save(os.path.join(test_root, 'predicted_diams.npy'), 
                                {'predicted_diams': predicted_diams, 'diams_style': diams_style})

def test_cellpose_main(data_root, save_root):
    """ data_root is folder with folders images_cyto_sp/train/models/, images_cyto/test and images_nuclei/test """
    #model_types = ['cyto', 'cyto_sp', 'nuclei']
    model_types = ['nuclei']
    for model_type in model_types:
        if model_type=='cyto' or model_type=='nuclei':
            pretrained_models = [str(Path.home().joinpath('.cellpose/models/%s_%d'%(model_type,j))) for j in range(4)]
        else:
            pretrained_models = glob(os.path.join(data_root, 'images_cyto_sp/train/models/cellpose_*'))
        test_root = os.path.join(data_root, 'images_%s/test/'%model_type.split('_')[0])
        print(test_root, pretrained_models)
        test_cellpose(test_root, save_root, pretrained_models, model_type)
        

def test_timing(test_root, save_root):
    itest=14
    test_data = io.imread(os.path.join(test_root, '%03d_img.tif'%itest))
    dat = np.load(os.path.join(test_root, 'predicted_diams.npy'), allow_pickle=True).item()
    rescale = 30. / dat['predicted_diams'][itest]
    Ly, Lx = test_data.shape[1:]
    test_data = cv2.resize(np.transpose(test_data, (1,2,0)), (int(Lx*rescale), int(Ly*rescale)))
    
    devices = [mx.gpu(), mx.cpu()]
    bsize = [256, 512, 1024]
    t100 = np.zeros((2,3,2))
    for d,device in enumerate(devices):
        model = models.CellposeModel(device=device, pretrained_model=None)
        for j in range(3):
            if j==2:
                test_data = np.tile(test_data, (2,2,1))
            img = test_data[:bsize[j], :bsize[j]]
            imgs = [img for i in range(100)]
            for k in [0,1]:
                tic = time.time()
                masks = model.eval(imgs, channels=[2,1], rescale=1.0, net_avg=k)[0]
                print(masks[0].max())
                t100[d,j,k] = time.time()-tic
                print(t100[d,j,k])


def test_cellpose(test_root, save_root, pretrained_models, diam_file=None, model_type='cyto'):
    """ test single cellpose net or 4 nets averaged """
    device = mx.gpu()
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    if model_type[:4]!='nuclei':
        channels = [2,1]
    else:
        channels = [0,0]

    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    
    # saved diameters
    if model_type != 'cyto_sp':
        if diam_file is None:
            dat = np.load(os.path.join(test_root, 'predicted_diams.npy'), allow_pickle=True).item()
        else:
            dat = np.load(diam_file, allow_pickle=True).item()
        if model_type=='cyto':
            rescale = 30. / dat['predicted_diams']
        else:
            rescale = 17. / dat['predicted_diams']
    else:
        rescale = np.ones(len(test_data))

    model = models.CellposeModel(device=device, pretrained_model=pretrained_models)
    masks = model.eval(test_data, channels=channels, rescale=rescale)[0]
    
    np.save(os.path.join(save_root, 'cellpose_%s_masks.npy'%model_type), masks)

def test_nets_3D(stack, model_root, save_root, test_region=None):
    """ input 3D stack and test_region (where ground truth is labelled) """
    device = mx.gpu()

    model_archs = ['unet3']#, 'unet2', 'cellpose']
    # found thresholds using ground truth
    cell_thresholds = [3., 0.25]
    boundary_thresholds = [0., 0.]
    for m,model_arch in enumerate(model_archs):
        if model_arch=='cellpose':
            pretrained_models = [str(Path.home().joinpath('.cellpose/models/cyto_%d'%j)) for j in range(4)]
            model = models.CellposeModel(device=device, pretrained_model=pretrained_models)
            masks = model.eval(stack, channels=[2,1], rescale=30./25., 
                                do_3D=True, min_size=2000)[0]
        else:
            pretrained_models = get_pretrained_models(model_root, unet=1, nclass=int(model_arch[-1]), 
                                                      residual=0, style=0, concatenate=1)
            model = models.UnetModel(device=device, pretrained_model=pretrained_models)
            masks = model.eval(stack, channels=[2,1], rescale=30./25., 
                                do_3D=True, min_size=2000, cell_threshold=cell_thresholds[m],
                                boundary_threshold=boundary_thresholds[m])[0]
        if test_region is not None:
            masks = masks[test_region]
            masks = utils.fill_holes_and_remove_small_masks(masks, min_size=2000)
        np.save(os.path.join(save_root, '%s_3D_masks.npy'%model_arch), masks)

def test_cellpose_kfold_aug(data_root, save_root):
    """ test trained cellpose networks on all cyto images """
    device = mx.gpu()
    ntest = 68
    concatenation = [0]
    residual_on = [1]
    style_on = [1]
    channels = [2,1]

    aps = np.zeros((9,68,len(thresholds)))
    
    for j in range(9):
        train_root = os.path.join(data_root, 'train%d/'%j)
        model_root = os.path.join(train_root, 'models/')

        test_root = os.path.join(data_root, 'test%d/'%j)
        test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
        test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
            
        k=0
        
        pretrained_models = get_pretrained_models(model_root, 0, 3, 
                                                    residual_on[k], style_on[k], 
                                                    concatenation[k])
        print(pretrained_models)
        
        cp_model = models.CellposeModel(device=device,
                                        pretrained_model=pretrained_models)
        
        dat = np.load(test_root+'predicted_diams.npy', allow_pickle=True).item()
        rescale = 30. / dat['predicted_diams']
        
        masks = cp_model.eval(test_data, channels=channels, rescale=rescale, net_avg=True, augment=True)[0]
        ap = metrics.average_precision(test_labels, masks, 
                                        threshold=thresholds)[0]
        print(ap[:,[0,5,8]].mean(axis=0))
        aps[j] = ap

    return aps

def test_cellpose_kfold(data_root, save_root):
    """ test trained cellpose networks on all cyto images """
    device = mx.gpu()
    ntest = 68
    concatenation = [0, 0, 0, 1, 1]
    residual_on = [1, 1, 0, 1, 0]
    style_on = [1, 0, 1, 1, 0]
    channels = [2,1]

    aps = np.zeros((9,9,68,len(thresholds)))
    
    for j in range(9):
        train_root = os.path.join(data_root, 'train%d/'%j)
        model_root = os.path.join(train_root, 'models/')

        test_root = os.path.join(data_root, 'test%d/'%j)
        test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
        test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
            
        for k in range(len(concatenation)):
            
            pretrained_models = get_pretrained_models(model_root, 0, 3, 
                                                      residual_on[k], style_on[k], 
                                                      concatenation[k])
            print(pretrained_models)
            
            cp_model = models.CellposeModel(device=device,
                                            pretrained_model=pretrained_models)
            
            dat = np.load(test_root+'predicted_diams.npy', allow_pickle=True).item()
            rescale = 30. / dat['predicted_diams']
            
            masks = cp_model.eval(test_data, channels=channels, rescale=rescale, net_avg=True, augment=False)[0]
            ap = metrics.average_precision(test_labels, masks, 
                                           threshold=thresholds)[0]
            print(ap[:,[0,5,8]].mean(axis=0))
            aps[j,k] = ap

            if k==0:
                # run single network
                for m, pretrained_model in enumerate(pretrained_models):
                    cp_model = models.CellposeModel(device=device,
                                                    pretrained_model=pretrained_model)
                    masks = cp_model.eval(test_data, channels=channels, rescale=rescale, net_avg=False, augment=False)[0]
                    ap = metrics.average_precision(test_labels, masks, 
                                                   threshold=thresholds)[0]
                    print(ap[:,[0,5,8]].mean(axis=0))
                    aps[j,m+5] = ap
    np.save(os.path.join(save_root, 'ap_cellpose_all.npy'), aps)

def size_distributions(data_root, save_root):
    """ size distributions for all images """
    ntest = 68
    sz_dist = np.zeros((9,ntest))
    
    for j in range(9):
        test_root = os.path.join(data_root, 'test%d/'%j)
        test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
        sz_dist[j] = np.array([utils.size_distribution(lbl) for lbl in test_labels])
    np.save(os.path.join(save_root, 'size_distribution.npy'), sz_dist)
    return sz_dist


            

