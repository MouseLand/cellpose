import sys, os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from natsort import natsorted
from glob import glob
from pathlib import Path
import string
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats
from cellpose import models, datasets, utils, transforms, io, metrics

thresholds = np.arange(0.5, 1.05, 0.05)

def cyto(test_root, save_root, save_figure=False):
    """ cyto performance, main fig 2 """
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
    
    masks = [[], []]
    aps = [[], []]
    model_types = ['cyto_sp', 'cyto']
    for m,model_type in enumerate(model_types):
        masks[m].append(np.load(os.path.join(save_root, 'cellpose_%s_masks.npy'%model_type), allow_pickle=True))
        masks[m].append(np.load(os.path.join(save_root, 'maskrcnn_%s_masks.npy'%model_type), allow_pickle=True))
        masks[m].append(np.load(os.path.join(save_root, 'stardist_%s_masks.npy'%model_type), allow_pickle=True))
        #masks[m].append(np.load(os.path.join(save_root, 'unet3_residual_on_style_on_concatenation_off_%s_masks.npy'%model_type), allow_pickle=True))
        masks[m].append(np.load(os.path.join(save_root, 'unet3_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
        masks[m].append(np.load(os.path.join(save_root, 'unet2_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
        
        for j in range(len(masks[m])):
            aps[m].append(metrics.average_precision(test_labels, masks[m][j], 
                                                    threshold=thresholds)[0])

    ltrf = 10
    rc('font', **{'size': 6})

    fig = plt.figure(figsize=(6.85,3.75/2 * 3),facecolor='w',frameon=True, dpi=300)

    mdl = ['cellpose', 'mask r-cnn', 'stardist',  'unet3', 'unet2']
    col ='mgcbyr'

    iimg = 1
    for j in range(3):
        ax = plt.subplot(3,4,2+j)

        img = test_data[1][1]
        img = np.stack((np.zeros_like(img), img, test_data[iimg][0]), axis=2)
        plt.imshow(np.clip(img[:,75:-75,:], 0, 255))

        outpix1 = utils.outlines_list(masks[0][j][iimg][:,75:-75])
        outpix = utils.outlines_list(test_labels[iimg][:,75:-75])
        for out in outpix:  
            plt.plot(out[:,0],  out[:,1],  color='y', lw=.5)
        for out in outpix1:
            plt.plot(out[:,0], out[:,1], '--', color='r', lw=.5)

        plt.title(mdl[j], color=col[j], loc = 'left')    
        plt.text(.5, 1.05, 'ap@0.5=%.2f'%aps[0][j][iimg,0], transform=ax.transAxes, fontsize=7)

        plt.arrow(305, 110, 7, 7, color='w', head_width = 5)
        plt.arrow(255, 325, -10, 0, color='w', head_width = 5)
        plt.arrow(155, 250, 10, 0, color='w', head_width = 5)
        plt.arrow(315, 50, 0, -10, color='w', head_width = 5)
        plt.arrow(100, 220, -7, -7, color='w', head_width = 5)

        plt.axis('off')

        if j==0:
            plt.text(.0, 1.2, 'specialist model / specialized data', fontsize = 7, style='italic', transform=ax.transAxes)
            plt.text(-.1, 1.2, 'b', fontsize = ltrf, transform=ax.transAxes)

    iimg = 16
    for j in range(3):
        ax = plt.subplot(3,4,6+j)

        img = test_data[iimg][1]
        img = np.stack((img, img, img), axis=2)
        plt.imshow(np.clip(img[:,:,:], 0, 255))

        outpix1 = utils.outlines_list(masks[1][j][iimg][:,:])    
        outpix = utils.outlines_list(test_labels[iimg])
        for out in outpix:
            plt.plot(out[:,0],  out[:,1],  color='y', lw=.5)
        for out in outpix1:
            plt.plot(out[:,0], out[:,1], '--', color='r', lw=.5)

        plt.title(mdl[j], color=col[j], loc = 'left', fontsize=7)    
        plt.text(.5, 1.1, 'ap@0.5=%.2f'%aps[1][j][iimg,0], transform=ax.transAxes, fontsize=6)
        plt.axis('off')

        if j==0:
            plt.text(.0, 1.3, 'generalist model / generalized data', fontsize = 7, style='italic', transform=ax.transAxes)
            plt.text(-.1, 1.3, 'c', fontsize = ltrf, transform=ax.transAxes)



    titles = ['specialist model / \n specialized data', 'specialist model /\n generalized data', 
            'generalist model / \n specialized data', 'generalist model /\n generalized data']

    ltr = 'defg'
    inds = [np.arange(0,11,1,int), np.arange(11,ntest,1,int)]
    for t in range(4):
        ax = fig.add_axes([0.1+.22*t,0.1,0.17,0.25])
        for j in range(len(mdl)):
            ax.plot(thresholds, aps[t//2][j][inds[t%2]].mean(axis=0), color=col[j])
            #print(aps[0][j][:11].mean(axis=0)[0])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1])
        ax.set_xlim([0.5, 1])
        if t==0:
            plt.ylabel('average precision')
        ax.set_xlabel('IoU matching threshold')
        ax.text(0, 1.05, titles[t], fontsize = 7, ha='left', transform=ax.transAxes)

        ax.text(-.25, 1.15, ltr[t], fontsize = 10, transform=ax.transAxes)

        if t==0:
            for j in range(len(mdl)):
                ax.text(.05, .4 - .075*j, mdl[j], color=col[j], fontsize=6, transform=ax.transAxes)

    #fig.tight_layout()

    #img=Image.open('D:/Drive/CODE/fromCarsen/cellpose/figs/training_schematic.png')
    #ax=fig.add_axes([.02,.4, .25, .65])
    #ax.imshow(img)
    #ax.axis('off')
    #plt.text(0, 1.2, 'a', fontsize = 14, transform=ax.transAxes)
    
    if save_figure:
        os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
        fig.savefig(os.path.join(save_root, 'figs/fig_perf2d_cyto.pdf'), bbox_inches='tight')


def nuclei(test_root, save_root, save_figure=False):
    """ nuclei performance, suppfig """
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
    
    masks = []
    aps = []
    model_type = 'nuclei'
    masks.append(np.load(os.path.join(save_root, 'cellpose_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'maskrcnn_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'stardist_%s_masks.npy'%model_type), allow_pickle=True))
    #masks.append(np.load(os.path.join(save_root, 'unet3_residual_on_style_on_concatenation_off_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'unet3_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'unet2_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
    
    for j in range(len(masks)):
        aps.append(metrics.average_precision(test_labels, masks[j], 
                                                threshold=thresholds)[0])

    ltrf = 10
    rc('font', **{'size': 6})

    fig = plt.figure(figsize=(6.85/2,3.85),facecolor='w',frameon=True, dpi=300)

    mdl = ['cellpose', 'mask r-cnn', 'stardist',  'unet3', 'unet2']
    col ='mgcbyr'


    iimg = 25
    for j in range(3):
        ax = plt.subplot(3,2,2*(1+j)-1)

        img = test_data[iimg][1]
        img = np.stack((img, img, img), axis=2)
        plt.imshow(np.clip(img[:,:,:], 0, 255))

        outpix1 = utils.outlines_list(masks[j][iimg])
        outpix = utils.outlines_list(test_labels[iimg])
        for out in outpix:  
            plt.plot(out[:,0],  out[:,1],  color='y', lw=.5)
        for out in outpix1:
            plt.plot(out[:,0], out[:,1], '--', color='r', lw=.5)

        plt.title(mdl[j], color=col[j], loc = 'left')    
        plt.text(.5, 1.05, 'ap@0.5=%.2f'%aps[j][iimg,0], transform=ax.transAxes, fontsize=6)

        plt.axis('off')

        if j==0:
            plt.text(-.1, 1.2, 'b', fontsize = ltrf, transform=ax.transAxes)

    ax=fig.add_axes([.65, .3 ,.33,.4])
    for j in range(len(mdl)):
        ax.plot(thresholds, aps[j].mean(axis=0), color=col[j], lw=1.)
        #print(aps[0][j][:11].mean(axis=0)[0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 1])
    ax.set_xlim([0.5, 1])
    ax.set_ylabel('average precision')
    ax.set_xlabel('IoU matching threshold')
    for j in range(len(mdl)):
        ax.text(.05, .32 - .075*j, mdl[j], color=col[j], fontsize=6, transform=ax.transAxes)
    ax.text(-.4, 1., 'c', fontsize = ltrf, transform=ax.transAxes)
    if save_figure:
        os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
        fig.savefig(os.path.join(save_root, 'figs/suppfig_perf2d_nuclei.pdf'), bbox_inches='tight')
    return masks, aps

def suppfig_cellpose_params(save_root, save_figure=False):
    ap_cellpose_all = np.load(os.path.join(save_root, 'ap_cellpose_all.npy'))

    rc('font', **{'size': 6})
    fig=plt.figure(figsize=(3,1.5), facecolor='w',frameon=True, dpi=300)
    colors = [c for c in plt.get_cmap('Dark2').colors]

    ap_compare = ap_cellpose_all[:,0,:,0].flatten()
    netstr = ['style off', 'residual off', 'concatenation on', 'unet architecture', 'one net']

    bmax = 0.15
    dbin = 0.02
    for i in range(5):#ap_cellpose_all.shape[1]-1):
        ax = fig.add_axes([0.1+i*0.5, 0.1, 0.38, 0.75])
        if i<5:
            apt = ap_cellpose_all[:,i+1,:,0].flatten()
        else:
            apt = ap_cellpose_all[:,5:,:,0].mean(axis=1).flatten()
        diffs = apt - ap_compare
        ax.text(-.1, 1.1, netstr[i], transform=ax.transAxes , fontsize=7) 
        hb = ax.hist(np.clip(diffs, -1*bmax+dbin/2, bmax-dbin/2), bins=np.arange(-bmax, bmax+dbin, dbin), 
                    color=colors[i])
        max_counts = hb[0].max()*1.05
        dm = np.mean(diffs)
        p = stats.wilcoxon(diffs).pvalue
        print(p)
        nstars = np.array([p<0.05, p<0.01, p<0.001]).sum()
        ax.scatter(dm, max_counts*1.025, marker='v', color=colors[i], s=10)
        ax.text(dm, max_counts*1.1, '%0.3f'%dm+'*'*nstars, ha='center', fontsize=6)
        ax.set_xlabel('difference in average precision')
        if i==0:
            ax.set_ylabel('# of test images')
        ax.text(-.3, 1.1, string.ascii_lowercase[i], transform=ax.transAxes, fontsize=11)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-bmax, bmax])
        #ax.set_xticks(np.arange(-0.3,0.4,0.1))
        ax.set_ylim([0, max_counts*1.25])

    if save_figure:
        os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
        fig.savefig(os.path.join(save_root, 'figs/suppfig_cellpose_params.pdf'))


def cyto3d(save_root, save_figure=True):
    thresholds = np.arange(0.25, 1.0, 0.05)

    model_archs = ['ilastik', 
                    'cellpose', 'unet3', 'unet2', 
                    'cellpose_stitch', 'maskrcnn_stitch', 'stardist_stitch']
    model_names = [['\u2014 ilastik'], 
                    ['\u2014 cellpose', '\u2014 unet3', '\u2014 unet2'],
                    ['-- cellpose', '-- mask r-cnn', '-- stardist']]
    titles = ['3D trained', '2D extended', '2D stitched']
    
    colors = ['xkcd:periwinkle', 'm', 'y', 'r', 'm', 'g',  
              'c']
    linestyles = ['-', '-', '-', '-', '--', '--', '--']
    linewidths = [1,1,1,1,1,1,1]
    masks_gt = np.load(os.path.join(save_root, 'ground_truth_3D_masks.npy'))
    masks = []
    aps = []
    for model_arch in model_archs:
        masks.append(np.load(os.path.join(save_root, '%s_3D_masks.npy'%model_arch)))
        thresholds = np.arange(0.25,1.0,0.05)
        aps.append(metrics.average_precision(masks_gt, masks[-1], threshold=thresholds)[0])
    aps = np.array(aps)
    print(aps[:,0])
    
    ltrf = 10
    rc('font', **{'size': 6})
    
    fig = plt.figure(figsize=(2,2),facecolor='w',frameon=True, dpi=300)
    ax = fig.add_subplot(111)

    for m,model_arch in enumerate(model_archs):
        ax.plot(thresholds, aps[m], colors[m], 
                linestyle=linestyles[m], linewidth=linewidths[m])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    j=0
    k=0
    dy=0.08
    for title,mnames in zip(titles,model_names):
        ax.text(0.7, 0.95-dy*j, title, transform=ax.transAxes)
        j+=1
        for mname in mnames:
            ax.text(0.7, 0.95-dy*j, mname, color=colors[k], transform=ax.transAxes)
            k+=1
            j+=1 
    ax.set_ylabel('average precision')
    ax.set_xlabel('IoU matching threshold')   
    ax.set_xlim([0.25,1.0])
    ax.set_ylim([0.,1.0])
    if save_figure:
        os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
        fig.savefig(os.path.join(save_root, 'figs/fig_perf3d.pdf'))

