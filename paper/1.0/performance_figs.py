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
from scipy.ndimage import find_objects
from scipy.optimize import linear_sum_assignment
from cellpose import models, datasets, utils, transforms, io, metrics, plot

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
    col ='mgcyr'

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

    inds = [np.arange(0,11,1,int), np.arange(11,ntest,1,int)]
    for t in range(4):
        ax = fig.add_axes([0.1+.22*t,0.1,0.17,0.25])
        for j in range(len(mdl)):
            ap = aps[t//2][j][inds[t%2]].mean(axis=0)
            #print(titles[t], mdl[j], ap[0])
            ax.plot(thresholds, ap, color=col[j])
            #print(aps[0][j][:11].mean(axis=0)[0])
            if t==2:
                print(mdl[j], aps[t//2][j].mean(axis=0)[0])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1])
        ax.set_xlim([0.5, 1])
        if t==0:
            plt.ylabel('average precision')
        ax.set_xlabel('IoU matching threshold')
        ax.text(0, 1.05, titles[t], fontsize = 7, ha='left', transform=ax.transAxes)

        ax.text(-.25, 1.15, string.ascii_lowercase[t+3], fontsize = 10, transform=ax.transAxes)

        if t==0:
            for j in range(len(mdl)):
                ax.text(.05, .4 - .075*j, mdl[j], color=col[j], fontsize=6, transform=ax.transAxes)

    ax = fig.add_axes([.05,.37,.25,.65])
    img = io.imread(os.path.join(save_root, 'figs/training_schematic_final.PNG'))
    ax.imshow(img)
    ax.axis('off')
    ax.text(0, 1.09, 'a', fontsize = ltrf, transform=ax.transAxes)

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
    col ='mgcyr'


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

def suppfig_metrics(test_root, save_root, save_figure=False):
    """ cyto performance measured with AJI and boundary precision """
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
    
    masks = []
    aji = []
    model_type = 'cyto'
    masks.append(np.load(os.path.join(save_root, 'cellpose_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'maskrcnn_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'stardist_%s_masks.npy'%model_type), allow_pickle=True))
    #masks.append(np.load(os.path.join(save_root, 'unet3_residual_on_style_on_concatenation_off_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'unet3_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
    
    for j in range(len(masks)):
        aji.append(metrics.aggregated_jaccard_index(test_labels, masks[j]))
        fsc.append(metrics.aggregated_jaccard_index(test_labels, masks[j]))


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

def suppfig_cellpose_params(test_root, save_root, save_figure=False):
    ap_cellpose_all = np.load(os.path.join(save_root, 'ap_cellpose_all.npy'))

    rc('font', **{'size': 6})
    fig=plt.figure(figsize=(3,1.5), facecolor='w',frameon=True, dpi=300)
    colors = [c for c in plt.get_cmap('Dark2').colors]

    ap_compare = ap_cellpose_all[:,0,:,0].flatten()
    netstr = ['style off', 'residual off', 'concatenation on', 'unet architecture', 'one net']

    bmax = 0.15
    dbin = 0.02
    for i in range(4):#ap_cellpose_all.shape[1]-1):
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

    # performance without specialized images
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
    masks = []
    aps = []
    model_type = 'cyto'
    masks.append(np.load(os.path.join(save_root, 'cellpose_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'cellpose_%s_wo_sp_masks.npy'%model_type), allow_pickle=True))
    for j in range(len(masks)):
        aps.append(metrics.average_precision(test_labels, masks[j], 
                                             threshold=thresholds)[0])

    i=4
    ax = fig.add_axes([0.1+i*0.5, 0.1, 0.38, 0.75])
    inds = [np.arange(11,ntest,1,int)]
    sstr = ['cellpose', 'cellpose w/o\nspecialized']
    ls = ['-']
    col = ['m', [0.3, 0, 0.3]]
    for t in range(len(inds)):
        for j in range(len(masks)):
            ax.plot(thresholds, aps[j][inds[t]].mean(axis=0), color=col[j], ls=ls[t])
            if t==0:
                ax.text(1., 0.7-j*0.18, sstr[j], color=col[j], ha='right')
            #print(aps[0][j][:11].mean(axis=0)[0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 1])
    ax.set_xlim([0.5, 1])
    ax.set_ylabel('average precision')
    ax.set_xlabel('IoU matching threshold')
    ax.text(-.1, 1., 'remove specialized images\n       from training set', transform=ax.transAxes , fontsize=7) 
    ax.text(-.3, 1.1, string.ascii_lowercase[i], transform=ax.transAxes, fontsize=11)

    if save_figure:
        os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
        fig.savefig(os.path.join(save_root, 'figs/suppfig_cellpose_params.pdf'), 
                    bbox_inches='tight')

def cyto3d(save_root, save_figure=True):
    thresholds = np.arange(0.25, 1.0, 0.05)

    model_archs = ['ilastik', 
                    'cellpose', 'unet3', 'unet2', 
                    'cellpose_stitch', 'maskrcnn_stitch', 'stardist_stitch']
    model_names = [['\u2014 ilastik'], 
                    ['\u2014 cellpose3D', '\u2014 unet3', '\u2014 unet2'],
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
        fig.savefig(os.path.join(save_root, 'figs/fig_perf3d.pdf'), bbox_inches='tight')

def suppfig_metrics(test_root, save_root, save_figure=False):
    """ boundary / aji metrics """
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
    
    masks = []
    bscores = []
    ajis = []
    model_type = 'cyto'
    
    masks.append(np.load(os.path.join(save_root, 'cellpose_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'maskrcnn_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'stardist_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'unet3_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'unet2_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
        
    scales = np.arange(0.025, 0.275, 0.025)
    for j in range(len(masks)):
        bscores.append(metrics.boundary_scores(test_labels, masks[j], scales))
        ajis.append(metrics.aggregated_jaccard_index(test_labels, masks[j]))

    ltrf = 10
    rc('font', **{'size': 6})

    fig = plt.figure(figsize=(6.85,3.85),facecolor='w',frameon=True, dpi=300)

    mdl = ['cellpose', 'mask r-cnn', 'stardist',  'unet3', 'unet2']
    col ='mgcyr'

    titles = ['boundary precision', 'boundary recall', 'boundary F-score', 'aggregated jaccard index']


    s=0
    ttr = ['specialized data', 'generalized data']
    ii = [0,2,1,3]
    inds = [np.arange(0,11,1,int), np.arange(11,ntest,1,int)]
    for t in range(len(inds)):
        for k in range(4):
            ax = fig.add_axes([0.1+.22*k,0.66-t*0.5,0.14,0.28])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if k<3:
                for j in range(len(mdl)):
                    ax.plot(scales*30, bscores[j][k][:,inds[t]].mean(axis=1), color=col[j])
                ax.set_ylim([0.25, 1])
                ax.set_ylabel(titles[k])
                ax.set_xlabel('boundary width (pixels)')
                if k==0:
                    ax.text(-0.2, 1.1, ttr[t], fontsize = 7, ha='left', transform=ax.transAxes)
                    
            else:
                for j in range(len(mdl)):
                    ax.bar(j, ajis[j][inds[t]].mean(), color=col[j])
                    ax.errorbar(j, ajis[j][inds[t]].mean(), 
                            ajis[j][inds[t]].std()/ len(inds[t])**0.5, color='k')
                ax.text(-0.2, 1.1, ttr[t], fontsize = 7, ha='left', transform=ax.transAxes)
                ax.set_ylabel(titles[k])
                ax.set_xticks(np.arange(0,len(mdl)))
                plt.xticks(rotation=90)
                ax.set_xticklabels(mdl) 
                ax.set_ylim([0,1])
                
            #ax.set_xlim([0.5, 1])
            if k==0 or k==3:
                ax.text(-.4, 1.1, string.ascii_lowercase[ii[s]], fontsize = ltrf, transform=ax.transAxes)
                s+=1
            if t==0 and k==0:
                for j in range(len(mdl)):
                    ax.text(.5, .5 - .075*j, mdl[j], color=col[j], fontsize=6, transform=ax.transAxes)


    if save_figure:
        os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
        fig.savefig(os.path.join(save_root, 'figs/suppfig_metrics.pdf'), bbox_inches='tight')

def mask_stats(test_root, save_root, save_figure=False):
    """ cyto performance broken down """
    ntest = len(glob(os.path.join(test_root, '*_img.tif')))
    test_data = [io.imread(os.path.join(test_root, '%03d_img.tif'%i)) for i in range(ntest)]
    test_labels = [io.imread(os.path.join(test_root, '%03d_masks.tif'%i)) for i in range(ntest)]
    
    masks = []
    aps = []
    model_type = 'cyto'
    
    masks.append(np.load(os.path.join(save_root, 'cellpose_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'maskrcnn_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'stardist_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'unet3_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
    masks.append(np.load(os.path.join(save_root, 'unet2_residual_off_style_off_concatenation_on_%s_masks.npy'%model_type), allow_pickle=True))
        
    for j in range(len(masks)):
        aps.append(metrics.average_precision(test_labels, masks[j], 
                                             threshold=thresholds)[0])

    # compute shape measure + bin it
    convexities = np.zeros(0)
    maskinds = np.zeros(0, 'int')
    for i in range(len(test_labels)):
        _,solidity, _ = utils.get_mask_stats(test_labels[i])
        convexities = np.append(convexities, solidity)
        maskinds = np.append(maskinds, i*np.ones(len(solidity), 'int'))
        if i==10:
            ispec = len(convexities)
    bins = np.array([np.percentile(convexities, ip) for ip in np.linspace(0,100,4)])
    digi = np.digitize(np.clip(convexities.copy(),bins[0]+.01, bins[-1]-0.01), 
                        bins=bins) - 1
    
    # compute IoU in shape bins
    nbins = 3
    iou_threshold = 0.5
    ioub = np.zeros((len(masks), nbins))
    ioub_exc = np.zeros((len(masks), nbins))
    ioub_ste = np.zeros((len(masks), nbins))    
    for j in range(len(masks)):
        iouall=np.zeros(0)
        for i in range(len(test_labels)):
            iou = metrics._intersection_over_union(test_labels[i], masks[j][i])[1:,1:]
            n_min = min(iou.shape[0], iou.shape[1])
            costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
            true_ind, pred_ind = linear_sum_assignment(costs)
            iout = np.zeros(test_labels[i].max())
            iout[true_ind] = iou[true_ind,pred_ind]
            iouall = np.append(iouall, iout)

        for d in np.unique(digi):
            iou_d = iouall[ispec:][digi[ispec:]==d]
            ioub_exc[j,d] = (iou_d<=iou_threshold).mean()
            iou_d = iou_d[iou_d > iou_threshold]
            ioub[j,d] = iou_d.mean()
            ioub_ste[j,d] = iou_d.std() / (digi==d).sum()
        
    ltrf = 10
    rc('font', **{'size': 6})

    fig = plt.figure(figsize=(6.85,3.85),facecolor='w',frameon=True, dpi=300)

    mdl = ['cellpose', 'mask r-cnn', 'stardist',  'unet3', 'unet2']
    col ='mgcyr'

    titles = ['Cells : fluorescent', 'Cells : nonfluorescent', 
              'Cell membranes', 'Microscopy : other', 'Non-microscopy']

    
    s=0
    inds = [np.arange(11,28,1,int), np.arange(28,33,1,int), 
            np.arange(33,42,1,int), np.arange(42,55,1,int),
            np.arange(55,ntest,1,int)]
    for t in range(len(inds)):
        ax = fig.add_axes([0.1+.22*t,0.66,0.14,0.28])
        for j in range(len(mdl)):
            ax.plot(thresholds, aps[j][inds[t]].mean(axis=0), color=col[j])
            #print(aps[0][j][:11].mean(axis=0)[0])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1])
        ax.set_xlim([0.5, 1])
        if t==0:
            plt.ylabel('average precision')
        ax.set_xlabel('IoU matching threshold')
        ax.text(-0.2, 1.1, titles[t], fontsize = 7, ha='left', transform=ax.transAxes)
        if t==0:
            ax.text(-.4, 1.1, string.ascii_lowercase[s], fontsize = 10, transform=ax.transAxes)
            s+=1
            for j in range(len(mdl)):
                ax.text(.5, .9 - .075*j, mdl[j], color=col[j], fontsize=6, transform=ax.transAxes)

    # size figs
    # size dist for all images
    sz_dist = np.load(os.path.join(save_root, 'size_distribution.npy'))
    # ap for all images
    aps=np.load(os.path.join(save_root, 'ap_cellpose_all.npy'))
    # example low-high masks
    d=np.load(os.path.join(save_root, 'example_size_masks.npy'), allow_pickle=True).item()
    msks = d['masks']
    imgs = d['imgs']
    inds = d['inds']
    szs = sz_dist.flatten()
    ap5 = aps[:,0,:,0].flatten()
    r,p = stats.pearsonr(szs, ap5)
    print(r,p)
    xb = np.linspace(0,1,3)
    bs = [300, 200]
    for j in range(2):
        ax = fig.add_axes([.07, .2-0.3*(j), 0.14,0.28])
        #inds = np.nonzero(np.logical_and(sz_dist[0,:]>xb[j], sz_dist[0,:]<xb[j+1]))[0]
        ic = inds[j]
        maski = plot.mask_overlay(imgs[j], msks[j])
        patch = plot.interesting_patch(msks[j], bsize=bs[j])
        ax.imshow(maski[np.ix_(patch[0], patch[1])])
        
        ax.axis('off')
        if j==0:
            ax.text(0.1, 1.05, 'low homogeneity', transform=ax.transAxes)
            ax.text(-.15,1.25,string.ascii_lowercase[s],transform=ax.transAxes, fontsize=10)
            ax.text(-.0,1.25,'within-image size variability',transform=ax.transAxes, fontsize=7)

            s+=1
        else:
            ax.text(0.1, 1.05, 'high homogeneity', transform=ax.transAxes)
            
    ax = fig.add_axes([0.285, 0.04, 0.18,0.4])
    ax.scatter(szs, ap5, s=0.5)
    slope, intercept = stats.linregress(szs, ap5)[:2]
    lp = szs*slope + intercept
    ax.plot(szs, lp, color='k', lw=1)
    ax.scatter(szs[inds], ap5[inds], s=40, color='r', marker='+', lw=1)
    ax.set_ylabel('AP @ IoU>0.5')
    ax.set_xlabel('homogeneity of mask areas\n(25th / 75th percentile)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(1.05, 0.63, 'r=%0.2f,\np=%0.3f'%(r, p), ha='right', 
                    transform=ax.transAxes, fontsize=6)
    #ax.text(-.25,1.1,string.ascii_lowercase[s],transform=ax.transAxes, fontsize=10)
    #ax.set_title('Within-image size variability')

    sstr = ['low', 'medium', 'high']
    print(s)
    for k in range(3):
        ic = np.nonzero(np.logical_and(convexities<bins[k+1], convexities>bins[k]))[0]
        np.random.seed(20)
        inds = np.random.permutation(len(ic))
        for l in range(24):
            ax = fig.add_axes([0.51+0.035*(l%8), 0.44-(3.3*k+l//8)*0.06, 0.03, 0.05])
            imask = ic[inds[l]]
            im = maskinds[imask]
            imask -= np.nonzero(maskinds==im)[0][0]
            slices = find_objects(test_labels[im]==(imask+1))
            ax.imshow(test_labels[im][slices[0]]==(imask+1), vmin=0, vmax=1, cmap='gray_r')
            ax.axis('off')
            if l==0:
                ax.text(0.1, 1.3,'%s'%(sstr[k]), ha='left', transform=ax.transAxes)
                if k==0:
                    ax.text(-0.5, 2, string.ascii_lowercase[s], transform=ax.transAxes, fontsize=10)
                    s+=1    
                    ax.text(0.1, 2, 'convexity distributions', transform=ax.transAxes, fontsize=7)

    dx=0.44
    pbins = np.arange(0,3)
    ax = fig.add_axes([0.42+dx, 0.04,.11,.4])
    for j in range(len(masks)):
        ax.plot(pbins+0.04*j, ioub_exc[j], color=col[j], lw=1)
        ax.scatter(pbins+0.04*j, ioub_exc[j], color=col[j], s=5)
        #if k==1 and s==0:
        #    ax.text(0.7,.95-0.06*(4-j), mdl[j], transform=ax.transAxes, color=col[j])
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['low', 'medium' , 'high'])
    ax.set_ylabel('miss rate')
    ax.set_xlabel('mask convexity')
    ax.text(.7, 1.1, 'IoU threshold = 0.5', transform=ax.transAxes)
    ax.set_ylim([0,1.])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    #ax.set_title('generalized data', fontsize=7)
    ax.text(-.5, 1.1, string.ascii_lowercase[s], transform=ax.transAxes, fontsize=10)
    s+=1
    
    ax = fig.add_axes([0.6+dx, 0.04, .11,.4])
    for j in range(len(masks)):
        ax.errorbar(pbins+0.04*j, ioub[j], ioub_ste[j],#np.abs(ioubi[k,j] - ioubi_ste[k,j].T), 
                    color=col[j], lw=1)
        ax.scatter(pbins+0.04*j, ioub[j], s=5, color=col[j])
        
    ax.set_ylim([0.4, 1.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('average IoU of true positives')
    ax.set_xlabel('mask convexity')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['low', 'medium' , 'high'])


    if save_figure:
        os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
        fig.savefig(os.path.join(save_root, 'figs/fig_stats.pdf'), bbox_inches='tight')

