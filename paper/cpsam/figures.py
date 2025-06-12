import sys 
from pathlib import Path
from scipy.stats import wilcoxon
import fastremap
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np 
from cellpose import io, utils, metrics, denoise, transforms
import matplotlib
import matplotlib.gridspec
import matplotlib.transforms as mtransforms
import torch
from natsort import natsorted

from fig_utils import * 
from benchmarks import load_dataset
from semantic import cl_colors, cl_names

def fig2(save_fig=False):
    fig = plt.figure(figsize=(14,9), dpi=150)
    grid = plt.GridSpec(5, 7, hspace=0.1, wspace=0.2, top=0.95, bottom=0.05, left=0.02, right=0.98)
    colors_ia = [0.5 * np.ones(3), "k"]
    colors_tab = plt.get_cmap("tab10").colors
    colors = ["g",  np.maximum(0, np.array(colors_tab[1])-0.1), #[0,1,0],
                [0.5, 0.3, 0], [0.7,0.4,1]] #[0.8,0.8,.3]]

    il = 0

    files, imgs, masks_H1 = load_dataset("cyto2")
    #ind_im = np.arange(91)
    dataset = "cyto2"
    algs = ["cyto3", "cellsam", "samcell", "cellposesam"]
    alg_names = np.array(["Cellpose cyto3", "CellSAM", "SAMCell", "Cellpose-SAM"])
    #algs = ["cyto3", "segformer", "cellsam", "samcell", "cellposesam"]
    #alg_names = np.array(["Cellpose cyto3", "Cellpose segformer", "CellSAM", "SAMCell", "Cellpose-SAM"])
    aps, tps, fps, fns = [], [], [], []
    errors = []
    masks_preds = []
    runtimes = []
    for alg in algs:
        dat = np.load(f"results/{alg}_{dataset}.npy", allow_pickle=True).item()
        errors.append((dat["fp"] + dat["fn"]) / (dat["tp"] + dat["fn"]))
        aps.append(dat["ap"])
        tps.append(dat["tp"])
        fps.append(dat["fp"])
        fns.append(dat["fn"])
        masks_preds.append(dat["masks_pred"])
        if "runtime" in dat:
            runtimes.append(dat["runtime"])
        else:
            runtimes.append(np.nan*np.zeros(len(masks_H1)))

    aps = np.array(aps)
    errors = np.array(errors)
    tps = np.array(tps)
    fps = np.array(fps)
    fns = np.array(fns)
    runtimes = np.array(runtimes)

    print(aps[:,:,0].mean(axis=1))
    print(errors[:,:,0].mean(axis=1))

    if 1:
        dat = np.load("/grive/denoising/styles_cyto3.npy", allow_pickle=True).item()

        train_types = np.array([t for t in dat["train_types"]])
        test_types = np.array([t for t in dat["test_types"]])
        train_types[train_types=="yeast_BF"] = "YeaZ"
        test_types[test_types=="yeast_BF"] = "YeaZ"
        train_types[train_types=="yeast_PhC"] = "YeaZ"
        test_types[test_types=="yeast_PhC"] = "YeaZ"
        train_styles = dat["train_styles"]
        test_styles = dat["test_styles"]

        type_names = np.unique(train_types)
        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, len(type_names)+2, subplot_spec=grid[0, :],
                                                                wspace=0.2, hspace=0.2)
        cc_stats = []
        dsets = ["Cellpose", "Nuclei", "Tissuenet", "Livecell", "YeaZ", "Omnipose\nphase-contrast", "Omnipose\nfluorescent", "DeepBacs"]
        for i, tname in enumerate(["cyto2", "nuclei", "tissuenet", "livecell", "YeaZ", "bact_phase", "bact_fluor", "deepbacs"]): 
            styles = train_styles[train_types==tname].copy()
            styles -= styles.mean(axis=1, keepdims=True)
            styles /= styles.std(axis=1, keepdims=True)

            styles_t = test_styles[test_types==tname].copy()
            styles_t -= styles_t.mean(axis=1, keepdims=True)
            styles_t /= styles_t.std(axis=1, keepdims=True)

            if tname == "cyto2":
                ind_im = np.array([68, 69, 71, 72, 73, 74, 75, 76, 84, 86, 89, 90])
                ind_im = np.hstack((np.arange(55), ind_im))
                styles_t = styles_t[ind_im]
            elif tname == "nuclei":
                itest = np.ones(len(styles_t), "bool")
                itest[76 : 103] = False
                styles_t = styles_t[itest]
                itrain = np.ones(len(styles), "bool")
                itrain[693 : 936] = False
                styles = styles[itrain]
            
            cc = (styles @ styles_t.T) / styles.shape[1]
            cc_stats.append(cc.mean(axis=0))

            ax = plt.subplot(grid1[0,i])
            pos = ax.get_position().bounds
            ax.set_position([pos[0] + (len(type_names) - i - 1)*0.012 - (i==0)*0.02 - 0.05, pos[1]-0.2*pos[2], pos[2]*0.95, pos[2]*2])
            im = ax.imshow(cc, aspect="auto", vmin=-1, vmax=1, cmap="bwr", interpolation="nearest")
            ax.set_title(dsets[i], loc="center", fontsize="medium")
            if i==0:
                ax.set_ylabel("train images")
                ax.set_xlabel("test images")
                cax = ax.inset_axes([1.05, 0.7, 0.05, 0.3])
                plt.colorbar(im, cax=cax)
                ticks = cax.yaxis.get_ticklabels()
                for tick in ticks:
                    tick.set_fontsize("small")
                ax.text(-0.1, 1.25, "Style vector correlation between train and test images", fontsize="large",
                        transform=ax.transAxes, fontstyle="italic")
                ax.text(-0.04, 0.15, "50", rotation=90, ha="center", va="center", fontsize="small",
                    transform=ax.transAxes)
                ax.text(0.25, -0.03, "25", rotation=0, ha="center", va="center", fontsize="small",
                    transform=ax.transAxes)
                ax.text(0.5, -0.15, "test images", rotation=0, ha="center", va="center",
                    transform=ax.transAxes)
                ax.text(-0.18, 0.5, "train images", rotation=90, ha="center", va="center",
                    transform=ax.transAxes)
                transl = mtransforms.ScaledTranslation(-20 / 72, 25 / 72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl, fs_title)
            
            ax.plot(-0.05*len(styles_t)*np.ones(2), [len(styles), len(styles) - 100], color="k")
            ax.plot([0, 25], 1.03*len(styles)*np.ones(2), color="k")
            ax.axis("off")

            dsets = ["Cellpose", "Nuclei", "Tissuenet", "Livecell", "YeaZ", "Omnipose (PhC)", "Omnipose (fluor)", "DeepBacs"]

        ax = plt.subplot(grid1[:,-2:])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + 0.*pos[2], pos[1] + 0.2*pos[3], pos[2]*1, pos[3]*0.9])
        vp = ax.violinplot(cc_stats, positions=np.arange(len(type_names)), widths=0.6, showmeans=True,
                        showextrema=False)
        for i in range(len(type_names)):
            vp["bodies"][i].set_facecolor(0.5*np.ones(3))
            vp["bodies"][i].set_alpha(0.35)
            ax.plot(np.array([-1, 1])*0.3 + i, cc_stats[i].mean(axis=0)*np.ones(2), color=0.5*np.ones(3), lw=3)
        ax.set_xticks(np.arange(len(type_names)))
        ax.set_xticklabels(dsets, rotation=30, ha="right")
        ax.set_ylabel("mean correlation\n per test image")
        ax.set_ylim([-0.1, 0.85])
        ax.set_yticks([0, 0.4, 0.8])
        ax.plot([-0.5, len(cc_stats)-0.5], [0, 0], color="k", lw=1, ls="--")
        ax.set_xlim([-0.5, len(cc_stats)-0.5])
        transl = mtransforms.ScaledTranslation(-60 / 72, 0/ 72, fig.dpi_scale_trans)
        il = plot_label(ltr, il, ax, transl, fs_title)
    else:
        print("Could not load styles_cyto3.npy, skipping style correlation plot")

    try:
        ind_im = np.array([68, 69, 71, 72, 73, 74, 75, 76, 84, 86, 89, 90])
        ind_im = np.concatenate((np.arange(55, dtype = 'int32'), ind_im), 0)
        masks_H2 = [np.load(f"/media/carsen/ssd3/datasets_cellpose/images_cyto2/labels2/{i:03d}_img_seg.npy", allow_pickle=True).item()["masks"] 
            for i in ind_im]
        ap, tp, fp, fn = metrics.average_precision(masks_H1, masks_H2, threshold=0.5)
        ap, tp, fp, fn = ap[:,0], tp[:,0], fp[:,0], fn[:,0]
    except:
        ap, tp, fp, fn = (np.nan*np.zeros(len(masks_H1)), np.nan*np.zeros(len(masks_H1)), 
                            np.nan*np.zeros(len(masks_H1)), np.nan*np.zeros(len(masks_H1)))
        print("no second set of masks found, skipping annotator 2 to 1 plot")
        

    np.random.seed(0)
    masks = masks_H1[40][:300, :300].copy()
    ismall = (fastremap.unique(masks, return_counts=True)[1] < 400).nonzero()[0]
    fastremap.mask(masks, ismall, in_place=True)
    masks = fastremap.renumber(masks)[0]
    print(masks.max())
    iperm = np.random.permutation(masks.max())
    outlines = utils.outlines_list(masks)

    dy = -0.04

    transl = mtransforms.ScaledTranslation(-8 / 72, 30 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid[1:3, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]+dy, pos[2], pos[3]])
    ax.set_title("Annotator 2 to 1", loc="center")
    ax.text(0.05, 1.25, "Simulated annotations", fontsize="large", fontstyle="italic",
            transform=ax.transAxes)
    icorrect = iperm[10:-10]
    imiss = np.hstack((iperm[5:10], iperm[-5:]))
    ibad = np.hstack((iperm[:5], iperm[-10:-5]))
    not_ibad = np.ones(masks.max(), dtype=bool)
    not_ibad[ibad] = False
    imfp = fastremap.mask(masks, not_ibad.nonzero()[0], in_place=False) > 0
    ax.imshow(imfp, cmap="bwr", vmin=-1, vmax=1)
    for i, outline in enumerate(outlines):
        if i+1 in icorrect or i+1 in imiss:
            ax.plot(outline[:,0], outline[:,1], color=colors_ia[0], lw=1.5, zorder=0)
            if i+1 in imiss:
                ax.scatter(np.median(outline[:,0]), np.median(outline[:,1]), color="r", 
                        s=300, marker="x", lw=2, zorder=30)
    ax.axis("off")
    il = plot_label(ltr, il, ax, transl, fs_title)

    axin = ax.inset_axes([1.05, 0.4, 1.05, 0.6])
    axin.scatter(0, 2, marker="o", color=colors_ia[0], s=100, facecolor="none", lw=1.5)
    axin.scatter(0, 1, marker="o", color="r", s=100)
    axin.scatter(0, 0, marker="x", color="r", s=100, lw=2)
    axin.set(xlim=(-0.1, 2.25), ylim=(-2, 2.5))
    axin.axis("off")
    x0 = 0.2
    axin.text(x0, 2, "Annotator 1", color=colors_ia[0], va="center")
    axin.text(x0, 1, "false positives (FP)", color="r", va="center")
    axin.text(x0, 0, "false negatives (FN)", color="r", va="center")


    #fig.savefig("figures/fig2.pdf", dpi=150)


    #fig.savefig("figures/annotator1_to_2.png", dpi=300)

    if 1:
        axin.scatter(2, -1.5, marker="o", color=colors_ia[1], s=100, facecolor="none", lw=1.5)
        axin.text(1.8, -1.5, "human consensus", color=colors_ia[1], va="center", ha="right")

        for k in range(2):
            if k==0:
                icorrect = iperm[5:-10]
                ibad = iperm[-5:]
                imiss = iperm[-10:-5]
            else:
                icorrect = iperm[10:-5]
                ibad = iperm[:5]
                imiss = iperm[5:10]
            
            ax = plt.subplot(grid[1:3, k+2])
            pos = ax.get_position().bounds
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-0.3*pos[2], pos[1]+dy, pos[2], pos[3]])
            not_ibad = np.ones(masks.max(), dtype=bool)
            not_ibad[ibad] = False
            imfp = fastremap.mask(masks, not_ibad.nonzero()[0], in_place=False) > 0
            ax.imshow(imfp, cmap="bwr", vmin=-1, vmax=1)
            for i, outline in enumerate(outlines):
                if i+1 in icorrect or i+1 in imiss:
                    ax.plot(outline[:,0], outline[:,1], "k", lw=1.5, zorder=0)
                    if i+1 in imiss:
                        ax.scatter(np.median(outline[:,0]), np.median(outline[:,1]), color="r", 
                                s=300, marker="x", lw=2, zorder=30)
            ax.axis("off")
            ax.set_title(f"Annotator {k+1}\n to human consensus", loc="center")



    iplot_ia = np.array([0, 1])
    iplot_errors = np.arange(len(aps))

    ax = plt.subplot(grid[1:5, -3:-1])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0]+0.05*pos[2], pos[1]+0.13*pos[3], pos[2]*0.75, pos[3]*(0.43+0.32-0.13)])
    axin = ax.inset_axes([0, 1.02, 1, 0.07])
    errors_ia = np.array([(fp + fn) / (tp + fn), (fp + fn) / (tp + fn) / 2])
    for i, error_ia in enumerate(errors_ia[iplot_ia]):
        vp = ax.violinplot(error_ia, positions=[i], widths=0.6, showmeans=True, showextrema=False)
        vp["bodies"][0].set_facecolor(colors_ia[i])
        vp["bodies"][0].set_alpha(0.5)
        ax.plot(0.3*np.array([-1,1]) + i, error_ia.mean()*np.ones(2), 
                color=colors_ia[i], lw=3)
    ax.set_ylabel("error rate relative to Annotator 1")

    for i in iplot_errors:
        vp = ax.violinplot(errors[i, :, 0], positions=[i+2], widths=0.6, showmeans=True, 
                    showextrema=False)
        vp["bodies"][0].set_facecolor(colors[i])
        vp["bodies"][0].set_alpha(0.35)
        ax.plot(i + 2 + 0.3*np.array([-1,1]), errors[i, :, 0].mean()*np.ones(2), color=colors[i], lw=3)
        if i < len(aps)-1:
            p = wilcoxon(errors[-1, :, 0], errors[i, :, 0]).pvalue 
            print(p)
            axin.plot([i+2, len(aps)-1+2], np.ones(2)*(0.95 + (len(aps)-1-i)*0.02), lw=1, color="k")
            axin.text((len(aps)-2-i+len(aps)-1)/2 + 2, 0.95 + (i+1)*0.02, "***", ha="center", va="center")
    for i in iplot_ia:
        ax.plot([1.5, 6.5], errors_ia[i].mean()*np.ones(2), lw=2, color=colors_ia[i], 
                ls="--" if i==0 else "-")
    ax.set_ylim([0, 0.8])
    ax.set_xlim([-0.75, (len(aps)+1)+.75])    
    axin.set_xlim([-0.75, (len(aps)+1)+.75])    
    axin.axis("off")
    colors_all = [*colors_ia, *colors]
    labels_all = np.hstack((["Annotator 2", "human consensus\n(estimate)"], alg_names))
    ax.set_xticks(np.arange(0, len(aps)+2))
    ax.set_xticklabels(labels_all, rotation=45, ha="right")
    for i, tick in enumerate(plt.gca().xaxis.get_ticklabels()):
        tick.set_color(colors_all[i])
        tick.set_fontweight("bold")

    ax.set_title("Performance on Cellpose test set", fontstyle="italic", y=1.22, x=-0.08)
    transl = mtransforms.ScaledTranslation(-40 / 72, 68 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)

    ax.text(0.3, 1.15, r"= $\frac{FP + FN}{TP + FN}$", transform=ax.transAxes, fontsize="xx-large")
    ax.text(0, 1.15, r"error rate", transform=ax.transAxes, fontsize="large", va="center")
    ax.text(0.05, 1.01, "n = 67 images", transform=ax.transAxes)

    #fig.savefig("figures/error_annotator.png", dpi=300)

    ax = plt.subplot(grid[1:5, -1])
    pos = ax.get_position().bounds 
    # 0.17 + 0.65 = 0.82
    ax.set_position([pos[0]-0.15*pos[2], pos[1]+0.43*pos[3], pos[2]*1.15, pos[3]*0.32])
    axin = ax.inset_axes([0, 1.02, 1, 0.07*0.65/0.35])
    vp = ax.violinplot(ap, positions=[-1], widths=0.6, showmeans=True, showextrema=False)
    vp["bodies"][0].set_facecolor(colors_ia[0])
    vp["bodies"][0].set_alpha(0.5)
    ax.plot(0.3*np.array([-1,1])-1, ap.mean()*np.ones(2), color=colors_ia[0], lw=3)
    for i in iplot_errors:
        vp = ax.violinplot(aps[i, :, 0], positions=[i], widths=0.6, showmeans=True, 
                    showextrema=False)
        vp["bodies"][0].set_facecolor(colors[i])
        vp["bodies"][0].set_alpha(0.35)
        ax.plot(i + 0.3*np.array([-1,1]), aps[i, :, 0].mean()*np.ones(2), color=colors[i], lw=3)
        if i < len(aps)-1:
            p = wilcoxon(aps[-1, :, 0], aps[i, :, 0]).pvalue 
            print(p)
            axin.plot([i, len(aps)-1], np.ones(2)*(0.95 + (len(aps)-1-i)*0.02), lw=1, color="k")
            axin.text((len(aps)-2-i+len(aps)-1)/2, 0.95 + (i+1)*0.02, "***", ha="center", va="center")
    ax.set_ylabel("average precision (AP) @ 0.5 IoU")
    ax.set_ylim([0.3, 1.0])
    ax.set_xlim([-1.5, len(aps)-1 + .5])
    axin.set_xlim([-1.5, len(aps)-1 + .5])
    axin.axis("off")
    ax.set_xticks(np.arange(-1, len(aps)))
    ax.set_xticklabels([])
    transl = mtransforms.ScaledTranslation(-50 / 72, 50 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
        
    ax.text(0.2, 1.27, r"= $\frac{TP}{TP + FN + FP}$", transform=ax.transAxes, fontsize="xx-large")
    ax.text(-0.22, 1.3, "average\nprecision", transform=ax.transAxes, fontsize="large", va="center")

    ax = plt.subplot(grid[-2:, -1:])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.15*pos[2], pos[1]+0.06*pos[3], pos[2]*1.15, pos[3]*0.65])
    #ax.set_position([pos[0]+0.2*pos[2], pos[1]+0.*pos[3], pos[2]*0.75, pos[3]*0.7])
    npix = np.array([m.size for m in masks_H1])**0.5
    for i in iplot_errors:
        ax.scatter(npix+np.random.rand(len(npix))*npix*0.05, runtimes[i], color=colors[i], 
                s=10, lw=1.5, marker="x")
    ax.set_xlabel("# of pixels per dimension")
    ax.text(-0.25, 0.5, "runtime (sec.)", rotation=90, va="center", transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(plt.FixedLocator([*np.arange(200, 1000, 100),
                                                *np.arange(1000, 3000, 1000)]))
    ax.xaxis.set_ticks([200, 1000])
    ax.xaxis.set_ticklabels(["200", "1000"])
    ax.yaxis.set_ticks([0.1, 1, 10])
    ax.yaxis.set_ticklabels(["0.1", "1", "10"])
    ax.set_xlim([180, 2100])
    ax.set_title("per image\nsegmentation time", y=0.9, loc="center")
    transl = mtransforms.ScaledTranslation(-50 / 72, -5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)


    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=grid[3:, :-3],
                                                            wspace=0.0, hspace=0.1)
    ylims = [[70, 300], [0, 200], [0, 700], [50, 200], [0, 200], [150, 450], [0, 300], [0, 150]]
    xlims = [[70, 300], [0, 200], [0, 700], [50, 200], [100, 300], [150, 450], [200, 500], [0, 150]]
    for i, iex in enumerate([22, 19, 60, 15, 16, 20, 40, 47]):##[56, 60, 53, 66]):#, 62]):
        #print(aps[-2,i,0] - aps[-1,i,0])
        img = imgs[iex].copy()
        img = np.tile(img[0], (3,1,1)) if np.ptp(img[1]) < 1e-3 else np.concatenate((np.zeros_like(img[:1]), img), axis=0)
        img = (255 * np.clip(img.transpose(1,2,0), 0, 1)).astype("uint8")
        outcols = [colors[-1]]

        ax = plt.subplot(grid1[i//4, i%4])#grid[1+(i>1), 2*i+k -4*(i>1)])  #2,6,k+1 + 2*i)
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.05*pos[2]*(i%4)-0.01, pos[1]+((i//4)==0)*0.02 - 0.02, pos[2]*1., pos[3]*1.])
        ax.imshow(img)
        for j, masks in enumerate([masks_preds[-1][iex]]):
            outlines = utils.outlines_list(masks)
            for outline in outlines:
                ax.plot(outline[:,0], outline[:,1], color=outcols[j], 
                        lw=1.5 if j==0 else 2, ls="dashed" if j==0 else "-")#, dashes=[2, 3] if j!=0 else [])
        ax.axis("off")
        if i==0:
            ax.set_title("Example segmentations from Cellpose test set", fontstyle="italic", y=1.05, x=-0.0)
            transl = mtransforms.ScaledTranslation(-14 / 72, 12 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl, fs_title)
        elif i==3:
            #ax.text(1., 1.15, "ground truth", color=outcols[0], transform=ax.transAxes, fontweight="bold", ha="right")
            ax.text(1, 1.05, f"{alg_names[-1]}", color=outcols[0], transform=ax.transAxes, fontweight="bold", ha="right")
            

        ax.text(1, -0.1, f"AP@0.5={aps[-1,iex,0]:.2f}", color=outcols[0], transform=ax.transAxes, ha="right")
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])

    if save_fig:
        fig.savefig("figures/fig2.pdf", dpi=150)

def fig3(save_fig=False):
    files, imgs, masks_true = load_dataset("cyto2")
    diam_true = [utils.diameters(m)[0] for m in masks_true]

    iex = 2
    outcols = [[0.8, 0.8,0.3], [0.7,0.4,1]]
    fig = plt.figure(figsize=(14, 7), dpi=150)
    grid = plt.GridSpec(3, 8, hspace=0.4, wspace=0.2, top=0.95, bottom=0.05, left=0.01, right=0.99)
    il = 0
    transl = mtransforms.ScaledTranslation(-14 / 72, 6 / 72, fig.dpi_scale_trans)        

    colors_tab = plt.get_cmap("tab10").colors
    colors = np.array([[0,0.5,0.], [0,1,0], colors_tab[6], 
            np.maximum(0, np.array(colors_tab[1])-0.1), [0.7,0.4,1]])

    dat = np.load("results/color_invariance.npy", allow_pickle=True).item()
    aps = dat["aps"]
    masks_preds = dat["masks_preds"]
    test_data = dat["test_data"]

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[0, :4],
                                                            wspace=0.1, hspace=0.2)
    xlabels = ['RGB', 'BRG', 'GBR', 'Random']
    for i in range(3):
        ax = plt.subplot(grid1[0, i])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.03*i, pos[1], pos[2], pos[3]])
        img_rsz = test_data[i][iex].transpose(1,2,0).copy().transpose(1,0,2)
        masks_true_rsz = masks_true[iex].copy()
        ax.imshow(np.clip(img_rsz[:,:,[2,1,0]]*1.1, 0, 1), interpolation="nearest")  
        outlines = utils.outlines_list(masks_true_rsz)
        for outline in outlines:
            ax.plot(outline[:, 1], outline[:, 0], color=outcols[0], lw=1)

        outlines = utils.outlines_list(masks_preds[i][iex])
        for outline in outlines:
            ax.plot(outline[:, 1], outline[:, 0], color=outcols[1], lw=1, linestyle="--")
        ycent, xcent = img_rsz.shape[0]//2, img_rsz.shape[1]//2 
        ax.set_ylim([ycent-65, ycent+65])
        ax.set_xlim([xcent-45, xcent+45])
        ax.axis('off')
        if i==0:
            ax.set_title(f"invariance to channel order", loc="left", fontstyle="italic")
            il = plot_label(ltr, il, ax, transl, fs_title)
            
        ax.set_title(xlabels[i], loc="center", y=-0.18, fontsize="medium")
    ax.text(1,1.01, "ground-truth", color=outcols[0], fontweight="bold", ha="right", va="bottom", transform=ax.transAxes)


    ax = plt.subplot(grid1[0, 3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.025, pos[1]+0.15*pos[3], pos[2], pos[3]*0.9])
    for j in range(4):
        vp = ax.violinplot(aps[j,:,0], positions=[j], widths=0.6, showmeans=True, showextrema=False)
        vp["bodies"][0].set_facecolor(outcols[1])
        vp["bodies"][0].set_alpha(0.5)
        ax.plot(0.3*np.array([-1,1]) + j, aps[j,:,0].mean()*np.ones(2), 
                color=outcols[1], lw=2)
    ax.text(1, 0.35, "Cellpose-SAM", color=colors[4], transform=ax.transAxes, ha="right")
    ax.set_ylabel("AP @ 0.5 IoU")
    ax.set_xticks([0, 1, 2, 3 ])
    ax.set_xticklabels(xlabels, fontsize="small", rotation=30, ha="right")
    ax.set_ylim([0, 1])
    print(aps[:,:,0].mean(axis=-1))

    dat = np.load("results/size_invariance.npy", allow_pickle=True).item()
    aps = dat["aps"]
    masks_preds = dat["masks_preds"]
    aps = np.array(aps)

    szs = [10, 15, 30, 60, 90]
    import cv2 

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[0, 4:],
                                                            wspace=0.1, hspace=0.2)
    for i in range(3):
        sz = szs[i*2]
        diam = diam_true[iex].copy() * 30. / sz
        ax = plt.subplot(grid1[0, i])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.03*i, pos[1], pos[2], pos[3]])
        img_rsz = transforms.resize_image(imgs[iex].copy().transpose(2,1,0), rsz=30./diam)
        img_rsz = np.concatenate((np.zeros_like(img_rsz[:,:,:1]), img_rsz), axis=-1) 
        masks_true_rsz = transforms.resize_image(masks_true[iex], rsz=30./diam, no_channels=True, interpolation=cv2.INTER_NEAREST)

        ax.imshow(np.clip(img_rsz[:,:,:]*1.1, 0, 1), interpolation="nearest")  
        outlines = utils.outlines_list(masks_true_rsz)
        for outline in outlines:
            ax.plot(outline[:, 1], outline[:, 0], color=outcols[0], lw=1)

        outlines = utils.outlines_list(masks_preds[i*2][iex])
        for outline in outlines:
            ax.plot(outline[:, 1], outline[:, 0], color=outcols[1], lw=1, linestyle="--")
        ycent, xcent = img_rsz.shape[0]//2, img_rsz.shape[1]//2 
        ycent -= 50 if i==2 else 0
        ax.set_ylim([ycent-65, ycent+65])
        ax.set_xlim([xcent-45, xcent+45])
        ax.axis('off')
        #ax.text(1, -0.1, f"AP@0.5={aps[0,i*2,iex,0]:.2f}", color=outcols[1], transform=ax.transAxes, ha="right")
        if i==0:
            ax.set_title(f"invariance to size", loc="left", fontstyle="italic")
            ax.set_title(f"cell diameter={sz}px", loc="center", y=-0.18, fontsize="medium")
            il = plot_label(ltr, il, ax, transl, fs_title)
        else:
            ax.set_title(f"{sz}px", loc="center", y=-0.18, fontsize="medium")

    ax = plt.subplot(grid1[0, 3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.025, pos[1]+0.15*pos[3], pos[2], pos[3]*0.9])
    for j in range(3):
        ax.errorbar(np.arange(5), aps[j, :,:,0].mean(axis=-1).T, aps[j, :,:,0].std(axis=-1).T / (66**0.5),
                    color=colors[[4,0,3]][j])
        ypos = 0.28- j*0.12 if j>0 else 0.95
        xpos = 1 if j<2 else 0.7
        ax.text(xpos, ypos, ["Cellpose-SAM", "Cellpose\ncyto3", "CellSAM"][j], color=colors[[4,0,3]][j], transform=ax.transAxes, ha="right")
    ax.set_ylabel("AP @ 0.5 IoU")
    ax.set_xticks([0, 1, 2, 3 ,4])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticklabels(["10", "15", "30", "60", "90"])
    ax.set_xlabel("cell diameter (pixels)")
    print(aps[:,:,:,0].mean(axis=-1))

    nstr = ["Poisson noise", "blur", "pixel size", "anisotropic blur"]
    nstr = np.array(nstr)[[0, 2, 1, 3]]
    rstr = ["denoising", "deblurring", "upsampling", "anisotropic\ndeconvolution"]
    rstr = np.array(rstr)[[0, 2, 1, 3]]
    xstr = ["noise", "blur", "pixel size", "anisotropic blur"]
    xstr = np.array(xstr)[[0, 2, 1, 3]]
    for ii, noise_type in enumerate(["poisson", "downsample", "blur", "aniso"]):
        dat = np.load(f"results/{noise_type}_invariance.npy", allow_pickle=True).item()
        aps = dat["aps"]
        masks_preds = dat["masks_preds"]

        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[ii//2+1, (ii%2)*4:4+4*(ii%2)],
                                                            wspace=0.1, hspace=0.2)

        if noise_type=="poisson":
            param = np.array([5, 2.5, 0.5])
        elif noise_type=="blur":
            param = np.array([2, 4, 8])# 48])
        elif noise_type=="downsample":
            param = np.array([2, 5, 10])
        else:
            param = np.array([2, 6, 12])

        denoise.deterministic()
        for i in range(3):
            k = i
            ax = plt.subplot(grid1[0, i])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-0.03*i, pos[1], pos[2], pos[3]])
            img = np.maximum(0, imgs[iex].copy().astype("float32").transpose(0,2,1))
            if noise_type=="poisson":
                params = {"poisson": 1.0, "blur": 0.0, "downsample": 0.0, "pscale": param[k]}
            elif noise_type=="blur":
                params = {"poisson": 1.0, "pscale": 120., "blur": 1.0, "downsample": 0.0,
                            "sigma0": param[k], "sigma1": param[k]}
            elif noise_type=="downsample":
                params = {"poisson": 0.0, "pscale": 0., "blur": 1.0, "downsample": 1.0, "ds": param[k],
                            "sigma0": param[k]/2, "sigma1": param[k]/2}
            else:
                params = {"poisson": 0.0, "pscale": 0., "blur": 1.0, "downsample": 1.0, "ds": param[k],
                            "sigma0": param[k]/2, "sigma1": param[k]/2/10, "iso": False}
            img = denoise.add_noise(torch.from_numpy(img).unsqueeze(0), 
                                    **params).cpu().numpy().squeeze()
            if noise_type=="downsample":
                img = img[:,::param[k], ::param[k]]
            elif noise_type=="aniso":
                img = img[:,::param[k]]
            img_rsz = img.transpose(1,2,0).copy()
            img_rsz = np.concatenate((np.zeros_like(img_rsz[:,:,:1]), img_rsz), axis=-1)
            masks_true_rsz = masks_true[iex].copy()

            if ii!=0 or (ii==0 and i==0):
                vmax = 1
            else:
                vmax = 1.1 if i==1 else 1.3
            ax.imshow(np.clip(img_rsz[:,:,:]*vmax, 0, 1), aspect=1 if noise_type!="aniso" else param[k], 
                    interpolation="nearest")#.transpose(1,0,2))  
            ycent, xcent = img_rsz.shape[0]//2, img_rsz.shape[1]//2 
            ax.set_ylim([ycent-65, ycent+65])
            ax.set_xlim([xcent-45, xcent+45])
            if noise_type=="downsample":
                ax.set_ylim([ycent-65/param[k], ycent+65/param[k]])
                ax.set_xlim([xcent-45/param[k], xcent+45/param[k]])
            elif noise_type=="aniso":
                ax.set_ylim([ycent-65/param[k], ycent+65/param[k]])
            ax.axis('off')
            if i==0:
                ax.set_title(f"robustness to {nstr[ii]}", loc="left", fontstyle="italic")
                il = plot_label(ltr, il, ax, transl, fs_title)
            ax.set_title(["low", "medium", "high"][i], loc="center", y=-0.18, fontsize="medium")
            
        ax = plt.subplot(grid1[0, 3])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.025, pos[1]+0.15*pos[3], pos[2], pos[3]*0.9])
        for j in range(4):
            ax.errorbar(np.arange(3), aps[j, :,:,0].mean(axis=-1).T, aps[j, :,:,0].std(axis=-1).T / (66**0.5),
                        color=colors[[4,0,0,3]][j], ls="--" if j==2 else "-")
            if 1:
                xpos = 1.1 if j<2 else 0.05
                ypos = 1.05-j*0.15+0.1*(ii==3) if j<2 else 0.3-(j-2)*0.15
                ax.text(xpos, ypos, ["Cellpose-SAM", f"-- cyto3+{rstr[ii]}", "cyto3", "CellSAM"][j], color=colors[[4,0,0,3]][j], transform=ax.transAxes, 
                    ha="left" if j>1 else "right", va="top")
        ax.set_ylabel("AP @ 0.5 IoU")
        ax.set_xticks([0, 1, 2])
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticklabels(["low", "medium", "high"])
        ax.set_xlabel(xstr[ii])
        print(nstr[ii], aps[:,:,:,0].mean(axis=-1))

    if save_fig:
        fig.savefig("figures/fig3.pdf", dpi=150)

def fig4(save_fig=False):
    
    ntrains_files = np.hstack(([-1], 2**np.arange(0, 9), [0]))
    colors = ["g", [0.7,0.4,1]]
    fig = plt.figure(figsize=(14, 7), dpi=150)
    grid = plt.GridSpec(3, 10, hspace=.4, wspace=0.2, top=0.93, bottom=0.07, left=0.02, right=0.98) 

    iexs = [22, 26, 141]
    iexs_3D = [3, 0, 0]
    ylims = [[50, 425], [0, 130], [40, 420]]
    xlims = [[0, 500], [210, 389], [0, 500]]  
    dset_names = ["BlastoSPIM (Nunley et al 2024)", "PlantSeg: lateral root (Wolny et al 2020)", "PlantSeg: ovules (Wolny et al 2020)"]
    il = 0
    transl = mtransforms.ScaledTranslation(-15 / 72, 18 / 72, fig.dpi_scale_trans)    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=grid[:, :3], wspace=0.05, hspace=0.45)
    grid2 = [matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid[:, 3:5], wspace=0.4, hspace=0.45),
            matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid[:, -2:], wspace=0.4, hspace=0.45)]
    grid3 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=grid[:, -5:-2], wspace=0.4, hspace=0.45)
    for d, dset in enumerate(["blastospim", "root", "ovules"]):
        #if "mito" in dset: #dset!="mitoEM-R":
        #    continue
        iex, ylim, xlim = iexs[d], ylims[d], xlims[d]
        dset_name = dset_names[d]

        dat = np.load(f"results/aps_new_{dset}.npy", allow_pickle=True).item()
        aps = dat["aps"]
        #print(np.nanmean(aps[:,:,:,:,0], axis=(-1,-2)))
        masks_pred_all = dat["masks_pred_all"]
        nrois = dat["nrois"]
        if dset == "root" or dset == "ovules":
            root = Path(f"/home/carsen/dm11_string/datasets_cellpose/root_ovules_wolny/{dset}/models/")
        else:
            root = Path(f"/home/carsen/dm11_string/datasets_cellpose/{dset}/models/")

        img_path = Path(*root.parts[:-1], "test", dat["test_files"][iex].name)
        masks_path = str(img_path)[:-4] + "_masks.tif"
        img = io.imread(img_path)
        masks_gt = io.imread(masks_path)
        
        itrain = [0, np.abs(nrois[0]-400).argmin()]
        print(itrain)
        outcols = [colors[1]]
        for i in range(2):
            ax = plt.subplot(grid1[d, i])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-0.00*i, pos[1], pos[2], pos[3]])
            ax.imshow(transforms.normalize99(img) , cmap="gray", vmin=0, vmax=1.2)
            for k, masks in enumerate([masks_pred_all[1][itrain[i]][iex]]):
                outlines = utils.outlines_list(masks)
                for outline in outlines:
                    ax.plot(outline[:, 0], outline[:, 1], c=outcols[k], 
                            lw=1.5, ls="--")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.text(1, -0.1, f"AP@0.5={aps[1,itrain[i],0,iex,0]:.2f}",
                    transform=ax.transAxes, ha="right")
            ax.axis("off")
            if i==0:
                #ax.set_title(r"n$_{ROIs}$, training = %d"%int(nrois[0, itrain[i]]), fontsize="medium")
                ax.text(0, 1.2, dset_name, transform=ax.transAxes, fontsize="large",
                        fontstyle="italic")
                il = plot_label(ltr, il, ax, transl, fs_title)
            ax.set_title(r"# of train ROIs = %d"%int(nrois[0, itrain[i]]), fontsize="medium") 
            #else:
            #    ax.set_title(f"{int(nrois[0, itrain[i]]):,} ROIs", fontsize="medium", loc="center")

        for k in range(2):
            if k==1:
                dat = np.load(f"results/aps_3D_{dset}.npy", allow_pickle=True).item()
                aps = dat["aps"]
            ax = plt.subplot(grid2[k][d])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]+0.2*pos[2]+0.14*pos[2]*(k==1), pos[1]-0.07*pos[3], pos[2]*0.7, pos[3]*1.1])
            accs = np.nanmean(aps[:,:,:,:,0], axis=-2)
            frac = 1 if d!=2 else 0.1
            for j in range(2):
                ax.errorbar(np.nanmean(nrois[:, 1:], axis=0) * frac, 
                            np.nanmean(accs[j,1:], axis=-1), np.nanstd(accs[j,1:], axis=-1) / (accs.shape[2]-1)**0.5,
                            color=colors[j])
                ax.errorbar(1, np.nanmean(accs[j,0]), marker="o", markersize=5, color=colors[j])
            ax.set_xscale("log")
            if d == 0:
                ax.set_xlabel("# of training ROIs")
                ax.set_ylabel("AP @ 0.5 IoU")
            ax.set_ylim([0, 0.8 if d!=0 or k!=1 else 1.0])
            dd = 2  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -dd), (1, dd)], markersize=12,
                        linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax.plot([1.8], [0], **kwargs)
            ax.plot([2.3], [0], **kwargs)
            xticks = 10**np.arange(0, 4 if d==1 else 5)
            ax.set_xticks(xticks)
            if d!=2:
                ax.set_xticklabels([rf"10$^{t}$" if t!=0 else "0" for t in range(len(xticks))])    
            else:
                ax.set_xticklabels([rf"10$^{t+1}$" if t!=0 else "0" for t in range(len(xticks))])

            for t in np.arange(1.95, 2.16, 0.05):
                ax.plot([t], [0], marker=[(-1, -dd), (1, dd)], markersize=12, 
                    color='w', mec='w', mew=1, clip_on=False, zorder=30)
            ax.set_xlim([0.7, np.nanmax(nrois)*1.1*frac])
            if d==0:
                ax.xaxis.set_minor_locator(plt.FixedLocator(np.hstack([np.arange(1, 10)*j for j in [1, 10, 100, 1000]])))
                ax.text(1., 0.3, "Cellpose-SAM", transform=ax.transAxes, color=colors[1], 
                        fontsize="large", ha="right")
                ax.text(1., 0.15, "Cellpose cyto3", transform=ax.transAxes, color=colors[0], 
                        fontsize="large", ha="right")
            else:
                ax.xaxis.set_minor_locator(plt.FixedLocator(np.hstack([np.arange(1, 10)*j for j in [1, 10, 100, 1000, 10000]])))
            #plt.grid(color=0.9*np.ones(3))

        for j in range(2):    
            im = io.imread(f"figures/{dset}_ntrain_{ntrains_files[itrain[j]]}.png")
            ax = plt.subplot(grid3[d, j])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]+0.07*pos[2]-0.08*pos[2]*(j==1), pos[1]-0.15*pos[3], pos[2]*1.2, pos[3]*1.22])
            ax.imshow(im)
            ax.set_title(f"# of train ROIs = {int(nrois[0,itrain[j]])}", 
                        fontsize="medium", loc="left")
            ax.text(1, -0.1, f"AP@0.5={aps[1,itrain[j],0,iexs_3D[d],0]:.2f}",
                        transform=ax.transAxes, ha="right")
            if d==0 and j==0:
                ax.text(0, 1.22, "3D segmentation w/ 2D model", transform=ax.transAxes, 
                            fontsize="large", fontstyle="italic")
            if d==0:
                ax.set_xlim([120, 880])
            elif d==1:
                ax.set_xlim([180, 900])
            else:
                ax.set_xlim([120, 880])
            ax.axis("off")

    fig.savefig("figures/fig4.pdf", dpi=150)

def fig5(save_fig=False): 

    leader_folders = ["cellposeSAM", "Amirreza_Mahbod", "IIAI",  "SharifHooshPardaz", "SJTU_426",]
    aps = []
    errors = []
    for lfolder in leader_folders:
        dat2 = np.load(f"results/monusac_{lfolder}.npy", allow_pickle=True).item()
        aps.append(np.array(dat2["aps"]))
        errors.append(np.array(dat2["errors"]))
        if lfolder=="cellposeSAM":
            dat = dat2

    aps = np.array(aps)
    errors = np.array(errors)
    print(np.nanmean(errors, axis=1), np.nanmean(errors, axis=(1,2)))
    print(np.nanmean(aps, axis=1), np.nanmean(aps, axis=(1,2)))
    img_files = dat["img_files"]

    folders = natsorted(np.unique([f.name.split("_")[0] for f in img_files]))

    colors_tab = plt.get_cmap("tab10").colors
    colors = [[0.8,0.8,.3], [0.5,0.5,0.5], [0.7,0.5,1]]

    #iexs = np.array([9, 10, 11, 16, 17, 18, 26, 28, 33, 34, 42, 43, 44, 53, 54, 55, 56, 57, 58, 59, 60, 61, 71, 74, 75, 79, 82])
    #iexs = iexs[[12, 3, 7, 11, 6, 13, 15, 17]] #4 = 6, 23 has bad areas
    iexs = np.array([44, 16, 28, 26, 55, 74, 43, 57, 53]) #55, 53
    ylims = [[0, 200], [650, 850], [0, 250], [0, 124], [0, 237],  [32, 300-20],[770, 970], [0, 185], [0, 200]]
    xlims = [[0, 500], [100, 300], [0, 250], [0, 180], [0, 240],  [172, 425],  [300, 500],[0, 221], [0, 200]]
    from scipy.stats import mode

    class_colors_pred = np.minimum(1, cl_colors.copy()/255 + 0.2)
    class_colors_pred[-1] = np.minimum(1, class_colors_pred[-1] + 0.1)
    class_colors_true = np.maximum(0, cl_colors.copy()/255 - 0.2)
    classes = dat["classes"]
    classes_true = dat["classes_true"]
    masks_pred = dat["masks_pred"]
    imgs = dat["imgs"]
    masks_true = dat["masks_true"]
    masks_bad = dat["masks_bad"]
    fig = plt.figure(figsize=(14*2./3,5), dpi=150)
    grid = plt.GridSpec(3, 7, hspace=0., wspace=0.1, top=0.95, bottom=0.05, left=0.01, right=0.99)
    il = 0
    for i, iex in enumerate(iexs):
        iap = [i for i, folder in enumerate(folders) if folder in img_files[iex].name][0]
        ax = plt.subplot(grid[i//3, i%3])
        if i==0:
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1]-0.1*pos[3], pos[2], 0.4*pos[3]])
        #elif i < 4:
        #    pos = ax.get_position().bounds
        #    ax.set_position([pos[0], pos[1]-0.1*pos[3], pos[2], pos[3]])
        class0 = classes[iex].copy()
        class_true = classes_true[iex].copy()
        masks_pred0 = masks_pred[iex].copy()
        masks_true0 = masks_true[iex].copy()
        masks_pred0[class0 == 0] = 0

        xlim, ylim = xlims[i], ylims[i]
        masks_pred0 = masks_pred0[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        masks_true0 = masks_true0[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        class0 = class0[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        class_true = class_true[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        masks_pred0 = fastremap.renumber(masks_pred0)[0]
        masks_true0 = fastremap.renumber(masks_true0)[0]

        cid = np.array([mode(class0[masks_pred0==j])[0] for j in range(1, masks_pred0.max()+1)]) - 1
        tid = np.array([mode(class_true[masks_true0==j])[0] for j in range(1, masks_true0.max()+1)]) - 1
        cid = cid.astype("int")
        tid = tid.astype("int")

        ax.imshow(imgs[iex][ylim[0]:ylim[1], xlim[0]:xlim[1]])
        outlines = utils.outlines_list(masks_true0)
        for j, outline in enumerate(outlines):
            ax.plot(outline[:,0], outline[:,1], color=class_colors_true[tid[j]], lw=1.5, ls="-")

        if 1:
            outlines = utils.outlines_list(masks_pred0)
            for j, outline in enumerate(outlines):
                ax.plot(outline[:,0], outline[:,1], color=class_colors_pred[cid[j]], lw=2.5, ls="--",
                        dashes=(1.5,2))

        ax.axis("off")
        if i==0:
            ax.text(0, 3, "MoNuSAC 2020 challenge: segmentation and classification", fontsize="large", fontstyle="italic", transform=ax.transAxes)
            ax.text(0.15, 2.5, "cell classes:", transform=ax.transAxes)
            for k, cname in enumerate(cl_names):
                ax.text(0.25, 2.2-k*0.3, cname, color=class_colors_true[k], transform=ax.transAxes)
            transl = mtransforms.ScaledTranslation(-0 / 72, 55 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl, fs_title)

    alg_names = ["Cellpose    \nSAM    ", "PL1", "PL2", "PL3", "L2"]
    colors = [[0.7,0.4,1], 0.5*np.ones(3), 0.5*np.ones(3), 0.5*np.ones(3), 0.5*np.ones(3)]
    for k in range(2):
        ax = plt.subplot(grid[:, 3 + k*2: 3 + k*2 + 2])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]+0.3*pos[2], pos[1]+0.05*pos[3], 0.7*pos[2], pos[3]*0.75])
        eps = errors if k==0 else aps
        for j in range(5):
            vp = ax.violinplot(np.nanmean(eps[j], axis=-1), showmeans=True, showmedians=False, showextrema=False, positions=[j])
            vp["bodies"][0].set_facecolor(colors[j])
            vp["bodies"][0].set_alpha(0.35)    
            ax.plot(j + 0.3*np.array([-1,1]), np.nanmean(eps[j], axis=-1).mean()*np.ones(2), color=colors[j], lw=3)
        if k==0:
            ax.set_ylabel("error rate @ 0.5 IoU")
            ax.set_ylim([0., 0.8])
            ax.set_yticks(np.arange(0,0.85,0.2))
        else:
            ax.set_ylabel("AP @ 0.5 IoU")
            ax.set_ylim([0., 1])
            ax.set_yticks(np.arange(0,1.05,0.2))

        ax.set_xticks(np.arange(0, aps.shape[0]))
        ax.set_xticklabels(alg_names, fontsize="small")#, rotation=20, ha="right")
        for j, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(colors[j])
        #ax.set_ylim(0.45, 0.95)
        ax.set_xlim([-0.5, 4.5])    
        transl = mtransforms.ScaledTranslation(-40 / 72, 25 / 72, fig.dpi_scale_trans)
        il = plot_label(ltr, il, ax, transl, fs_title)

        from scipy.stats import wilcoxon
        axin = ax.inset_axes([0, 1.02, 1, 0.15])
        l0 = 0
        for j in range(1,5):
            from scipy.stats import ttest_rel
            p = wilcoxon(np.nanmean(eps[0], axis=-1), np.nanmean(eps[j], axis=-1)).pvalue
            print(k,j,p)
            pstr = "n.s." if p > 0.05 else ("*" if p >= 0.01 else "**" if p >= 0.001 else "***")
            axin.plot([0, j], np.ones(2)*(0.95 + (len(eps)-j)*0.02), lw=1, color="k")
            pstr = "n.s." if p > 0.05 else ("*" if p >= 0.01 else "**" if p >= 0.001 else "***")
            axin.text(j/2, 0.95 + (len(eps)-j)*0.02 + 0.01*(p>0.05), 
                        pstr, ha="center", va="center", fontsize="small" if p > 0.05 else "large")
        axin.axis("off")
        axin.set_xlim([-0.5, 4.5])    
    if save_fig:
        fig.savefig("figures/fig5.pdf")

def supp_bench(save_fig=False):
    algs = ["cellposesam", "cyto3", "omnipose", "cellsam", "microsam", "samcell",  "pathosam"]
    alg_names = np.array(["Cellpose-SAM", "Cellpose\ncyto3", "Omnipose",
                            "CellSAM", "microSAM", "SAMCell", "PathoSAM"])
    aps, tps, fps, fns = [], [], [], []
    errors = []
    masks_preds = []
    runtimes = []
    dsets = ["tissuenet", "livecell", "bact_phase", "bact_fluor", "deepbacs", "monuseg"]
    dset_names = ["Tissuenet", "Livecell", "Omnipose (PhC)", "Omnipose (fluor)", "DeepBacs", "MoNuSeg"]
    aps, tps, fps, fns = [], [], [], []
    for d, dset in enumerate(dsets):
        for i, alg in enumerate(algs):
            if Path(f"results/{alg}_{dset}.npy").exists():
                dat = np.load(f"results/{alg}_{dset}.npy", allow_pickle=True).item()
            else:
                continue
            ap = dat["ap"]
            tp = dat["tp"]
            fp = dat["fp"]
            fn = dat["fn"]
            if i==0:
                aps.append(np.nan*np.zeros((len(algs), *ap.shape)))
                tps.append(np.nan*np.zeros((len(algs), *tp.shape)))
                fps.append(np.nan*np.zeros((len(algs), *fp.shape)))
                fns.append(np.nan*np.zeros((len(algs), *fn.shape)))
                errors.append(np.nan*np.zeros((len(algs), *ap.shape)))
            aps[d][i] = ap
            tps[d][i] = tp
            fps[d][i] = fp
            fns[d][i] = fn
            errors[d][i] = (fp + fn) / (fn + tp)

    fig = plt.figure(figsize=(14,10), dpi=150)
    grid = plt.GridSpec(2, 7, hspace=1, wspace=1, top=0.96, bottom=0.04, left=0.01, right=0.99)
    colors_ia = [0.5 * np.ones(3), "k"]
    colors_tab = plt.get_cmap("tab10").colors
    colors = [[0.7,0.4,1], "g", "y", np.maximum(0, np.array(colors_tab[1])-0.1), "tab:blue",
                [0.5, 0.3, 0], [0, 1, 1]]

    for j in range(2):
        ax = plt.subplot(grid[j,1:])
        pos = ax.get_position().bounds 
        ax.set_position([pos[0], pos[1] - 0.1*(j==0) - 0.02*(j==1), pos[2], pos[3]])
        axin = ax.inset_axes([0., 1.02, 1, 0.1])
        xticks, xticklabels, xtickcolors = [], [], []
        k = 0
        for d in range(len(dsets)):
            ips = np.nonzero(~np.isnan(aps[d][:,0,0]))[0]
            if j==1:
                eps = aps[d][ips,:,0]
            else:
                eps = errors[d][ips,:,0]
            for i in range(len(ips)):
                vp = ax.violinplot(eps[i], positions=[k], widths=0.6, showmeans=True, 
                            showextrema=False)
                vp["bodies"][0].set_facecolor(colors[ips[i]])
                vp["bodies"][0].set_alpha(0.35)
                if i > 0:
                    #p = wilcoxon(eps[0], eps[i], alternative="less" if j==0 else "greater").pvalue 
                    p = wilcoxon(eps[0], eps[i], alternative="two-sided").pvalue 
                    print(dsets[d], ips[i], p)
                    #if "bact" in dsets[d] and ips[i]==3:
                        #print(wilcoxon(eps[0], eps[i], alternative="two-sided").pvalue)#"greater" if j==0 else "less").pvalue )

                    axin.plot([k-i, k], np.ones(2)*(0.95 + (len(ips)-i)*0.02), lw=1, color="k")
                    pstr = "n.s." if p > 0.05 else ("*" if p >= 0.01 else "**" if p >= 0.001 else "***")
                    axin.text(k - i/2, 0.95 + (len(ips)-i)*0.02 + 0.01*(p>0.05), 
                                pstr, ha="center", va="center", fontsize="small" if p > 0.05 else "large")
                ax.plot(k + 0.3*np.array([-1,1]), eps[i].mean()*np.ones(2), color=colors[ips[i]], lw=3)
                xticks.append(k)
                xticklabels.append(alg_names[ips[i]].replace("\n", " "))
                xtickcolors.append(colors[ips[i]])
                k += 1
            if j==0:
                ax.text(k - len(ips)/2 - 0.5, 0.98, f"{dset_names[d]}\nn={len(eps[0]):,}", ha="center", va="bottom", 
                fontsize="large")
            k+=1.5   
        if j==1:
            ax.set_ylabel("average precision (AP) @ 0.5 IoU", fontsize="large")
            ax.set_ylim([0.3, 1.0])
            ax.set_xticks([])
        else:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
            for i, color in enumerate(xtickcolors):
                ax.get_xticklabels()[i].set_color(color)
            ax.set_ylabel("error rate @ 0.5 IoU", fontsize="large")
            ax.set_ylim([0, 0.8])
            k = 0
            y = np.array([7, 5.85, 0.7, 4.7, 4, 2, 1.3]) * 1/8
            for i in range(len(algs)):
                ax.text(-0.16, y[i], alg_names[i], ha="left", va="center", fontsize="large", 
                        color=colors[i], transform=ax.transAxes, fontweight="bold")
                #k += 1 if i != 1 and i != 2 else 1.7
            ax.text(-0.17, 0.98, "generalists", ha="left", va="center", fontsize="large",
                    color="k", transform=ax.transAxes, fontstyle="italic", fontweight="bold")
            ax.text(-0.17, 0.35, "specialists", ha="left", va="center", fontsize="large",
                    color="k", transform=ax.transAxes, fontstyle="italic", fontweight="bold")
        ax.spines["bottom"].set_visible(False)
        ax.grid(True, color=0.8*np.ones(3), lw=0.5, ls="--", axis="y")
        axin.axis("off")

    fig.savefig("figures/supp_bench.pdf", dpi=150)