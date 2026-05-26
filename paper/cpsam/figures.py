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

colors_tab = plt.get_cmap("tab10").colors
colors = ["g",  np.maximum(0, np.array(colors_tab[1])-0.1), #[0,1,0],
                [0.5, 0.3, 0], "c", [0.7,0.4,1], [1, 0.7, 1], [0.9,0.4,1], 
                "y", "tab:blue"]
algs = ["cyto3", "cellsam", "samcell", "microsam", "cpsam", "cpdino-vitb", "cpdino", "omnipose", "pathosam"]
alg_names = np.array(["Cellpose cyto3", "CellSAM", "SAMCell", "MicroSAM", "CellposeSAM",
                       "CellposeDINO-ViTB", "CellposeDINO", "Omnipose", "PathoSAM"])

outcols = [[0.8, 0.8,0.3], [0.7,0.4,1]]

alg_dict = {alg: {"name": name, "color": color} for alg, name, color in zip(algs, alg_names, colors)}


def fig2(root, save_fig=False):
    fig = plt.figure(figsize=(14,9), dpi=150)
    grid = plt.GridSpec(5, 7, hspace=0.1, wspace=0.2, top=0.95, bottom=0.05, left=0.02, right=0.98)
    colors_ia = [0.5 * np.ones(3), "k"]
    algs = ["cyto3", "cellsam", "samcell", "microsam", "cpdino-vitb", "cpdino", "cpsam"]
    il = 0

    files, imgs, masks_H1 = load_dataset("cyto2", root=root / "../")
    dataset = "cyto2"
    aps, tps, fps, fns = [], [], [], []
    errors = []
    masks_preds = []
    runtimes = []
    for alg in algs:
        dat = np.load(root / f"results/{alg}_{dataset}.npy", allow_pickle=True).item()
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

    try:
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

            dsets = ["Cellpose", "Nuclei", "Tissuenet", "Livecell", "YeaZ", "Omnipose\n(PhC)", "Omnipose\n(fluor)", "DeepBacs"]

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
        ax.set_xticklabels([d.replace("\n", " ") for d in dsets], rotation=30, ha="right")
        ax.set_ylabel("mean correlation\n per test image")
        ax.set_ylim([-0.1, 0.85])
        ax.set_yticks([0, 0.4, 0.8])
        ax.plot([-0.5, len(cc_stats)-0.5], [0, 0], color="k", lw=1, ls="--")
        ax.set_xlim([-0.5, len(cc_stats)-0.5])
        transl = mtransforms.ScaledTranslation(-60 / 72, 0/ 72, fig.dpi_scale_trans)
        il = plot_label(ltr, il, ax, transl, fs_title)
    except:
        print("Could not load styles_cyto3.npy, skipping style correlation plot")

    try:
        ind_im = np.array([68, 69, 71, 72, 73, 74, 75, 76, 84, 86, 89, 90])
        ind_im = np.concatenate((np.arange(55, dtype = 'int32'), ind_im), 0)
        masks_H2 = [np.load(root / f"../images_cyto2/labels2/{i:03d}_img_seg.npy", allow_pickle=True).item()["masks"] 
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
    ax.set_position([pos[0]+0.03*pos[2], pos[1]+0.13*pos[3], pos[2]*0.79, pos[3]*(0.43+0.32-0.2)])
    axin = ax.inset_axes([0, 1.02, 1, 0.16])
    errors_ia = np.array([(fp + fn) / (tp + fn), (fp + fn) / (tp + fn) / 2])
    for i, error_ia in enumerate(errors_ia[iplot_ia]):
        vp = ax.violinplot(error_ia, positions=[i], widths=0.6, showmeans=True, showextrema=False)
        vp["bodies"][0].set_facecolor(colors_ia[i])
        vp["bodies"][0].set_alpha(0.5)
        ax.plot(0.3*np.array([-1,1]) + i, error_ia.mean()*np.ones(2), 
                color=colors_ia[i], lw=3)
    ax.set_ylabel("error rate relative to Annotator 1")

    for i, alg in enumerate(algs):
        vp = ax.violinplot(errors[i, :, 0], positions=[i+2], widths=0.6, showmeans=True, 
                    showextrema=False)
        vp["bodies"][0].set_facecolor(alg_dict[alg]["color"])
        vp["bodies"][0].set_alpha(0.35)
        ax.plot(i + 2 + 0.3*np.array([-1,1]), errors[i, :, 0].mean()*np.ones(2), color=alg_dict[alg]["color"], lw=3)
        if i != len(aps)-1:
            p = wilcoxon(errors[-1, :, 0], errors[i, :, 0]).pvalue 
            print(p)
            axin.plot([i+2, len(aps)+1], np.ones(2)*(0.95 + (len(aps)-1-i)*0.02), lw=1, color="k")
            pstr = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            axin.text((len(aps)+1 - (i+2))/2 + (i+2), 0.95 + (len(aps)-1-i)*0.02 - 0.015 + 0.01*(pstr=="n.s."), 
                      pstr, ha="center", va="bottom")
    for i in iplot_ia:
        ax.plot([1.5, len(aps)+1.5], errors_ia[i].mean()*np.ones(2), lw=2, color=colors_ia[i], 
                ls="--" if i==0 else "-")
    ax.set_ylim([0, 0.8])
    ax.set_xlim([-0.5, (len(aps)+1)+.5])    
    axin.set_xlim([-0.5, (len(aps)+1)+.5])    
    axin.axis("off")
    colors_all = [*colors_ia, *[alg_dict[alg]["color"] for alg in algs]]
    labels_all = ["Annotator 2", "human consensus\n(estimate)"] + [alg_dict[alg]["name"] for alg in algs]
    labels_all = [label.replace("MicroSAM", "MicroSAM (2026)") for label in labels_all]
    ax.set_xticks(np.arange(0, len(aps)+2))
    ax.set_xticklabels(labels_all, rotation=45, ha="right")
    for i, tick in enumerate(plt.gca().xaxis.get_ticklabels()):
        tick.set_color(colors_all[i])
        tick.set_fontweight("bold")

    ax.set_title("Performance on Cellpose test set", fontstyle="italic", y=1.34, x=-0.08)
    transl = mtransforms.ScaledTranslation(-40 / 72, 95 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)

    ax.text(0.3, 1.25, r"= $\frac{FP + FN}{TP + FN}$", transform=ax.transAxes, fontsize="xx-large")
    ax.text(0, 1.28, r"error rate", transform=ax.transAxes, fontsize="large", va="center")
    ax.text(0.05, 1.01, "n = 67 images", transform=ax.transAxes)

    #fig.savefig("figures/error_annotator.png", dpi=300)

    ax = plt.subplot(grid[1:5, -1])
    pos = ax.get_position().bounds 
    # 0.17 + 0.65 = 0.82
    ax.set_position([pos[0]-0.15*pos[2], pos[1]+0.43*pos[3], pos[2]*1.05, pos[3]*0.32])
    axin = ax.inset_axes([0, 1.02, 1, 0.28])
    vp = ax.violinplot(ap, positions=[-1], widths=0.6, showmeans=True, showextrema=False)
    vp["bodies"][0].set_facecolor(colors_ia[0])
    vp["bodies"][0].set_alpha(0.5)
    ax.plot(0.3*np.array([-1,1])-1, ap.mean()*np.ones(2), color=colors_ia[0], lw=3)
    for i, alg in enumerate(algs):
        vp = ax.violinplot(aps[i, :, 0], positions=[i], widths=0.6, showmeans=True, 
                    showextrema=False)
        vp["bodies"][0].set_facecolor(alg_dict[alg]["color"])
        vp["bodies"][0].set_alpha(0.35)
        ax.plot(i + 0.3*np.array([-1,1]), aps[i, :, 0].mean()*np.ones(2), color=alg_dict[alg]["color"], lw=3)
        if i < len(aps)-1:
            p = wilcoxon(aps[-1, :, 0], aps[i, :, 0]).pvalue 
            print(p)
            axin.plot([i, len(aps)-1], np.ones(2)*(0.95 + (len(aps)-1-i)*0.02), lw=1, color="k")
            pstr = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            axin.text((len(aps)-1-i)/2 + i, 0.95 + (len(aps)-1-i)*0.02 - 0.015 + 0.01*(pstr=="n.s."), 
                      pstr, ha="center", va="bottom")
    ax.set_ylabel("average precision (AP) @ 0.5 IoU")
    ax.set_ylim([0.4, 1.0])
    ax.set_xlim([-1.5, len(aps)-1 + .5])
    axin.set_xlim([-1.5, len(aps)-1 + .5])
    axin.axis("off")
    ax.set_xticks(np.arange(-1, len(aps)))
    ax.set_xticklabels([])
    transl = mtransforms.ScaledTranslation(-50 / 72, 62 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
        
    ax.text(0.2, 1.4, r"= $\frac{TP}{TP + FN + FP}$", transform=ax.transAxes, fontsize="xx-large")
    ax.text(-0.22, 1.43, "average\nprecision", transform=ax.transAxes, fontsize="large", va="center")

    ax = plt.subplot(grid[-2:, -1:])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.15*pos[2], pos[1]+0.06*pos[3], pos[2]*1.15, pos[3]*0.65])
    #ax.set_position([pos[0]+0.2*pos[2], pos[1]+0.*pos[3], pos[2]*0.75, pos[3]*0.7])
    npix = np.array([m.size for m in masks_H1])**0.5
    for i, alg in enumerate(algs):
        ax.scatter(npix+np.random.rand(len(npix))*npix*0.05, runtimes[i], color=alg_dict[alg]["color"], 
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
        outcols = [alg_dict[algs[-1]]["color"]]

        ax = plt.subplot(grid1[i//4, i%4])#grid[1+(i>1), 2*i+k -4*(i>1)])  #2,6,k+1 + 2*i)
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.05*pos[2]*(i%4)-0.01, pos[1]+((i//4)==0)*0.02 - 0.02, pos[2]*1., pos[3]*1.])
        ax.imshow(img)
        for j, masks in enumerate([masks_preds[-1][iex]]):
            outlines = utils.outlines_list(masks, multiprocessing=False)
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
            ax.text(1, 1.05, f"{alg_dict[algs[-1]]['name']}", color=outcols[0], transform=ax.transAxes, fontweight="bold", ha="right")
            
        ax.text(1, -0.1, f"AP@0.5={aps[-1,iex,0]:.2f}", color=outcols[0], transform=ax.transAxes, ha="right")
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])

    if save_fig:
        fig.savefig("figures/fig2.pdf", dpi=150)

    return errors


def supp_models(root, save_fig=False):
    algs = ["cpsam", "cpdino", "cpdino-vitb", "cpdino_ps16", "cpsam_linear", 
        "cpdino_linear", "cpdino-vitb_linear"]

    dataset = "cyto2"

    files, imgs, masks_H1 = load_dataset("cyto2", root=root / "../")

    aps, tps, fps, fns = [], [], [], []
    errors = []
    masks_preds = []
    runtimes = []
    for alg in algs:
        dat = np.load(root / f"results/{alg}_{dataset}.npy", allow_pickle=True).item()
        errors.append((dat["fp"] + dat["fn"]) / (dat["tp"] + dat["fn"]))
        aps.append(dat["ap"])
        tps.append(dat["tp"])
        fps.append(dat["fp"])
        fns.append(dat["fn"])
        masks_preds.append(dat["masks_pred"])
        if "runtime" in dat:
            runtimes.append(dat["runtime"])
        if alg == "cpsam_linear": 
            flows = dat["flows"]
        
    aps = np.array(aps)
    errors = np.array(errors)
    tps = np.array(tps)
    fps = np.array(fps)
    fns = np.array(fns)
    runtimes = np.array(runtimes)

    fig = plt.figure(figsize=(14,6), dpi=150)
    grid = plt.GridSpec(2, 8, hspace=0.2, wspace=0.35, top=0.95, bottom=0.11, left=0.05, right=0.98)
    il = 0

    print(aps[:,:,0].mean(axis=1))
    print(errors[:,:,0].mean(axis=1))

    colors = [alg_dict[alg.split("_")[0]]["color"] for alg in algs]
    colors[3] = [1, 0, 0.2]
    names = [alg_dict[alg.split("_")[0]]["name"] for alg in algs]
    names[3] += " patch-size=16"
    names = [name + (" linear-probe" if "linear" in alg else "") for name, alg in zip(names, algs)]

    ax = plt.subplot(grid[0, 0])
    pos = ax.get_position().bounds 
    ax.set_position([pos[0], pos[1]+0.*pos[3], pos[2], pos[3]*0.75])
    axin = ax.inset_axes([0, 1.02, 1, 0.25])
    for i, alg in enumerate(algs):
        vp = ax.violinplot(aps[i, :, 0], positions=[i], widths=0.6, showmeans=True, 
                    showextrema=False)
        vp["bodies"][0].set_facecolor(colors[i])
        vp["bodies"][0].set_alpha(0.35)
        ax.plot(i + 0.3*np.array([-1,1]), aps[i, :, 0].mean()*np.ones(2), color=colors[i], lw=3)
        if i > 0:
            p = wilcoxon(aps[0, :, 0], aps[i, :, 0]).pvalue 
            print(p)
            axin.plot([0, i], np.ones(2)*(0.95 + (len(aps)-1-i)*0.02), lw=1, color="k")
            pstr = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            axin.text((i)/2, 0.95 + (len(aps)-1-i)*0.02 - 0.015 + 0.01*(pstr=="n.s."), 
                      pstr, ha="center", va="bottom")
    ax.set_ylabel("average precision (AP) @ 0.5 IoU")
    ax.set_ylim([0., 1.0])
    ax.set_xlim([-.5, len(aps)-1 + .5])
    axin.set_xlim([-.5, len(aps)-1 + .5])
    axin.axis("off")
    ax.set_xticks(np.arange(len(aps)))
    ax.set_xticklabels(names, rotation=90, ha="center", fontsize="small")
    transl = mtransforms.ScaledTranslation(-40 / 72, 40 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    for i, xtick in enumerate(ax.xaxis.get_ticklabels()):
        xtick.set_color(colors[i])        

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=grid[:, 1:3], 
                                                        wspace=0.1, hspace=0.2)
    
    iex = 15
    ylim = [50, 180]
    xlim = [50, 200]
    ax = plt.subplot(grid1[0, 0])
    ax.imshow(imgs[iex][0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title("image", fontsize="medium", loc="center")
    transl = mtransforms.ScaledTranslation(-20 / 72, 10 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    
    ax = plt.subplot(grid1[1, 0])
    ax.imshow(flows[iex][0])
    ax.axis("off")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title("flows from\nlinear probe", fontsize="medium", loc="center")
    
    ax = plt.subplot(grid1[2, 0])
    ax.imshow(flows[iex][2])
    ax.axis("off")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title("cell-prob. from\nlinear probe", fontsize="medium", loc="center")

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=grid[:, 2:], 
                                                        wspace=0.4, hspace=0.5)
    for nd in range(2, 4):
        if nd==2:
            ntiles_all = 2 ** np.arange(0, 8)# np.array([1, 16, 32, 64, 128])
            print(ntiles_all)
        else:
            ntiles_all = np.array([1, 2, 4, 6, 8])
            print(ntiles_all)

        algs = ["cpsam", "cpdino", "cpdino-vitb", "cyto3"]
        gpus = ["r4070s", "p6000", "a100", "h100"]
        gpu_names = ["RTX 4070S", "RTX PRO 6000", "A100", "H100"]
        mem_max = np.nan * np.zeros((len(algs), len(gpus), len(ntiles_all)))
        mem_gpu_max = np.nan * np.zeros((len(algs), len(gpus), len(ntiles_all)))
        runtime = np.nan * np.zeros((len(algs), len(gpus), len(ntiles_all)))
        for i, ntiles in enumerate(ntiles_all):
            for j, alg in enumerate(algs):
                for k, gpu in enumerate(gpus):
                    if Path(f"ramlogs/mem_{nd}d_{ntiles}_{alg}_{gpu}.out").exists():
                        with open(f"ramlogs/mem_{nd}d_{ntiles}_{alg}_{gpu}.out", "r") as f:
                            out = f.read()
                            out = out.split("\n")[1:-1]
                            mem = np.array([float(x.split(" ")[1]) for x in out if x[:3]=="MEM"])
                        mem_max[j, k, i] = mem.max()/1e3

                        with open(f"ramlogs/log_{nd}d_{ntiles}_{alg}_{gpu}.out", "r") as f:
                            out = f.read()
                            out = out.split("\n")[-6:-4]
                            try:
                                mr = np.array([float(x) for x in out])
                            except:
                                mr = np.array([np.nan, np.nan])
                        mem_gpu_max[j, k, i] = mr[-1]
                        runtime[j, k, i] = mr[0]

        for k, gpu in enumerate(gpus):
            ax = plt.subplot(grid1[nd-2, k])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]+0.018*(3-k), *pos[1:]])
            print(f"nd={nd}, gpu={gpu}")
            print("RAM")
            print(mem_max[:, k])
            print("GPU RAM")
            print(mem_gpu_max[:, k])
            print("Runtime")
            print(runtime[:, k])
            for j in range(len(algs)):
                ax.loglog((ntiles_all*150)**nd, runtime[j, k], label=algs[j], marker="o",
                        color=alg_dict[algs[j]]["color"], markersize=3, lw=1)
                if k==0 and nd==2:
                    ax.text(0.35, 0.27-0.08*j, alg_dict[algs[j]]["name"], color=alg_dict[algs[j]]["color"], 
                            transform=ax.transAxes)
            ax.set_ylim([1.e-2, 3.55*60] if nd==2 else [60e-2, 46*60])
            slc = slice(0, len(ntiles_all)) if k>0 else slice(0, len(ntiles_all)-1)
            ax.set_xticks((ntiles_all[slc]*150)**nd)
            ax.set_xticklabels([rf"{ntiles*150:,d}$^{{{nd}}}$" for ntiles in ntiles_all[slc]], 
                               rotation=45, ha="right", va="top")
            if k==0:
                ax.set_ylabel("runtime (sec.)")
                ax.set_xlabel("number of pixels", labelpad=0)
                ax.text(-0.2, 1.02, f"{nd}D segmentation", fontsize="large", fontstyle="italic", 
                        transform=ax.transAxes, va="bottom")
                transl = mtransforms.ScaledTranslation(-45 / 72, 5 / 72, fig.dpi_scale_trans)
                il = plot_label(ltr, il, ax, transl, fs_title)
            # turn off minor ticks
            ax.xaxis.set_minor_locator(plt.NullLocator())
            #if nd==3:
            ax.set_title(f"{gpu_names[k]}", fontsize="medium", loc="center", y=0.88)
    
    if save_fig:
        fig.savefig("figures/supp_models.pdf", dpi=150)
    
def cp4_text(ax, x, y):
    ax.text(x, y, "CellposeSAM" + 27*" ", color=alg_dict["cpsam"]["color"], transform=ax.transAxes, ha="right")
    ax.text(x, y, "CellposeDINO" + 0*" ", color=alg_dict["cpdino"]["color"], transform=ax.transAxes, ha="right")
    ax.text(x-0., y-0.13, "CellposeDINO-ViTB", color=alg_dict["cpdino-vitb"]["color"], transform=ax.transAxes, ha="right")


def fig3(root, save_fig=False):
    files, imgs, masks_true = load_dataset("cyto2", root=root / "..")
    diam_true = [utils.diameters(m)[0] for m in masks_true]

    iex = 2
    fig = plt.figure(figsize=(14, 7), dpi=150)
    grid = plt.GridSpec(3, 8, hspace=0.4, wspace=0.2, top=0.95, bottom=0.05, left=0.01, right=0.99)
    il = 0
    transl = mtransforms.ScaledTranslation(-14 / 72, 6 / 72, fig.dpi_scale_trans)        

    xlabels = ['RGB', 'BRG', 'GBR', 'random']
    cperms = [[0,1,2], [1,2,0], [2,0,1], np.random.permutation(3)]

    aps, masks_preds = [], []
    algs = ["cpsam", "cpdino", "cpdino-vitb"]
    for alg in algs:
        aps.append([])
        masks_preds.append([])
        for xl in xlabels:
            dat = np.load(root / f"results/{alg}_cyto2_{xl}.npy", allow_pickle=True).item()
            aps[-1].append(dat["ap"])
            masks_preds[-1].append(dat["masks_pred"])
    aps = np.array(aps)
    print(aps.shape)
    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[0, :4],
                                                            wspace=0.1, hspace=0.2)
    for i in range(3):
        ax = plt.subplot(grid1[0, i])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.03*i, pos[1], pos[2], pos[3]])
        img_rsz = imgs[iex].transpose(1,2,0).transpose(1,0,2).copy()
        img_rsz = np.concatenate((img_rsz, np.zeros_like(img_rsz[:,:,:1])), axis=-1)
        masks_true_rsz = masks_true[iex].copy()
        ax.imshow(np.clip(img_rsz[:,:,cperms[i]]*1.1, 0, 1), interpolation="nearest")  
        outlines = utils.outlines_list(masks_true_rsz, multiprocessing=False)
        for outline in outlines:
            ax.plot(outline[:, 1], outline[:, 0], color=outcols[0], lw=1)

        outlines = utils.outlines_list(masks_preds[0][i][iex], multiprocessing=False)
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
    xp = 1.3
    for j in range(len(algs)):
        for i in range(len(xlabels)):
            vp = ax.violinplot(aps[j,i,:,0], positions=[len(algs)*i*xp + j], widths=0.6, showmeans=True, showextrema=False)
            vp["bodies"][0].set_facecolor(alg_dict[algs[j]]["color"])
            vp["bodies"][0].set_alpha(0.5)
            ax.plot(0.3*np.array([-1,1]) + len(algs)*i*xp + j, aps[j,i,:,0].mean()*np.ones(2), 
                    color=alg_dict[algs[j]]["color"], lw=2)
        ax.text(1, 0.35-j*0.13, alg_dict[algs[j]]["name"], color=alg_dict[algs[j]]["color"], 
                transform=ax.transAxes, ha="right")
    ax.set_ylabel("AP @ 0.5 IoU")
    ax.set_xticks(np.arange(len(xlabels)*len(algs)*xp, step=len(algs)*xp) + 1)
    ax.set_xticklabels(xlabels, fontsize="small", rotation=30, ha="right")
    ax.set_ylim([0, 1])
    print(aps[:,:,0].mean(axis=-1))

    szs = [10, 15, 30, 60, 90]
    import cv2 
    algs = ["cpsam", "cpdino", "cpdino-vitb", "microsam", "cellsam", "cyto3"]
    aps, masks_preds = [], []
    for alg in algs:
        aps.append([])
        masks_preds.append([])
        for sz in szs:
            if (root / f"results/{alg}_cyto2_sz{sz}.npy").exists():
                dat = np.load(root / f"results/{alg}_cyto2_sz{sz}.npy", allow_pickle=True).item()
                aps[-1].append(dat["ap"])
                masks_preds[-1].append(dat["masks_pred"])
            else:
                aps[-1].append(np.nan*np.zeros(dat["ap"].shape))
    
    aps = np.array(aps)
    
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
        outlines = utils.outlines_list(masks_true_rsz, multiprocessing=False)
        for outline in outlines:
            ax.plot(outline[:, 1], outline[:, 0], color=outcols[0], lw=1)

        outlines = utils.outlines_list(masks_preds[0][i*2][iex], multiprocessing=False)
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
    cp4_text(ax, 1.1, 1.1)
    for j in range(len(algs)):
        ax.errorbar(np.arange(5), aps[j, :,:,0].mean(axis=-1).T, aps[j, :,:,0].std(axis=-1).T / (66**0.5),
                    color=alg_dict[algs[j]]["color"], lw=1)
        if j>2:
            xpos = 1.1
            ypos = 0.35-(j-3)*0.1
            ax.text(xpos, ypos, alg_dict[algs[j]]["name"], color=alg_dict[algs[j]]["color"], 
                    transform=ax.transAxes, ha="right", va="top")
    ax.set_ylabel("AP @ 0.5 IoU")
    ax.set_xticks([0, 1, 2, 3 ,4])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticklabels(["10", "15", "30", "60", "90"])
    ax.set_xlabel("cell diameter (pixels)")
    print(aps[:,:,:,0].mean(axis=-1))

    algs = ["cpsam", "cpdino", "cpdino-vitb", "microsam", "cellsam", "cyto3", "cyto3_restore"]

    nstr = ["Poisson noise", "blur", "pixel size", "anisotropic blur"]
    nstr = np.array(nstr)[[0, 2, 1, 3]]
    rstr = ["denoising", "deblurring", "upsampling", "anisotropic\ndeconvolution"]
    rstr = np.array(rstr)[[0, 2, 1, 3]]
    xstr = ["noise", "blur", "pixel size", "anisotropic blur"]
    xstr = np.array(xstr)[[0, 2, 1, 3]]
    for ii, noise_type in enumerate(["poisson", "downsample", "blur", "aniso"]):
        aps, masks_preds = [], []
        for alg in algs:
            aps.append([])
            masks_preds.append([])
            for nl in range(3):
                dat = np.load(root / f"results/{alg}_cyto2_{noise_type}_{nl}.npy", allow_pickle=True).item()
                aps[-1].append(dat["ap"])
                masks_preds[-1].append(dat["masks_pred"])
        aps = np.array(aps)

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

        for i in range(3):
            k = i
            ax = plt.subplot(grid1[0, i])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-0.03*i, pos[1], pos[2], pos[3]])
            img = np.maximum(0, imgs[iex].copy().astype("float32"))
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
            denoise.deterministic()
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
        alg_names = [alg_dict[alg]["name"] if alg!="cyto3_restore" else f"-- cyto3+{rstr[ii]}" for alg in algs]
        ax.set_position([pos[0]-0.025, pos[1]+0.15*pos[3], pos[2], pos[3]*0.9])
        if ii==0:
            cp4_text(ax, 1.1, 1.05+0.1*(ii==3))
        for j in range(len(algs)):
            alg0 = algs[j] if algs[j]!="cyto3_restore" else "cyto3"
            ax.errorbar(np.arange(3), aps[j, :,:,0].mean(axis=-1).T, aps[j, :,:,0].std(axis=-1).T / (66**0.5),
                        color=alg_dict[alg0]["color"], 
                        ls="--" if algs[j]=="cyto3_restore" else "-")
            if j == len(algs)-1 or (ii==0 and j>2):
                xpos = 1.1 if j==len(algs)-1 else 0.02
                ypos = 0.32-(j-3)*0.1 if j!=len(algs)-1 else 0.9+0.1*(ii==3)
                ax.text(xpos, ypos, alg_names[j], 
                        color=alg_dict[alg0]["color"], transform=ax.transAxes, 
                    ha="left" if j!=len(algs)-1 else "right", va="top")
        ax.set_ylabel("AP @ 0.5 IoU")
        ax.set_xticks([0, 1, 2])
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticklabels(["low", "medium", "high"])
        ax.set_xlabel(xstr[ii])
        print(nstr[ii], aps[:,:,:,0].mean(axis=-1))        

    if save_fig:
        fig.savefig("figures/fig3.pdf", dpi=150)


def supp_invariance(root, save_fig=False):
    files, imgs, masks_true = load_dataset("cyto2", root=root / "..")
    aps, masks_preds = [], []
    algs = ["cpsam", "cpdino", "cpdino-vitb"]
    xlabels = ['', 'BGR', 'ud', 'lr']
    xnames = ['BGR (channel swap)', 'vertical flip', 'horizontal flip']
    for alg in algs:
        aps.append([])
        masks_preds.append([])
        for i, xl in enumerate(xlabels):
            xl = f"_{xl}" if len(xl) > 0 else ""
            dat = np.load(root / f"results/{alg}_cyto2{xl}.npy", allow_pickle=True).item()
            #aps[-1].append(dat["ap"])
            masks_preds[-1].append(dat["masks_pred"])
            if i > 0:
                ap, tp, fp, fn = metrics.average_precision(masks_preds[-1][0], masks_preds[-1][-1], np.arange(0.5, 1, 0.05))
                aps[-1].append(ap)
    aps = np.array(aps)
    print(aps.shape)

    fig = plt.figure(figsize=(14, 2.5))
    grid = plt.GridSpec(1, 8, hspace=0.4, wspace=0.35, top=0.85, bottom=0.17, left=0.02, right=0.96)
    il = 0
    transl = mtransforms.ScaledTranslation(-14 / 72, 6 / 72, fig.dpi_scale_trans)        

    iex = 19

    outcols = ['r', 'b', [0, 0.7, 1]]
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[0, :5],
                                                            wspace=0.1, hspace=0.2)
    for i, xl in enumerate(xlabels[1:4]):
        ax = plt.subplot(grid1[0, i])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.01*i, pos[1], pos[2], pos[3]])
        img_rsz = imgs[iex].copy().transpose(1,2,0)
        img_rsz = np.concatenate((img_rsz, np.zeros_like(img_rsz[:,:,:1])), axis=-1)
        masks_true_rsz = masks_preds[0][0][iex].copy()
        masks_pred = masks_preds[0][i+1][iex].copy()
        if xl == "BGR":
            img_rsz = img_rsz[:,:,[2,1,0]]
        elif xl == "ud":
            img_rsz = img_rsz[::-1]
            masks_true_rsz = masks_true_rsz[::-1]
            masks_pred = masks_pred[::-1]
        elif xl == "lr":
            img_rsz = img_rsz[:,::-1]
            masks_true_rsz = masks_true_rsz[:,::-1]
            masks_pred = masks_pred[:,::-1]
        print(img_rsz.shape)
        # if i>0:
        #     img_rsz = img_rsz[:,:,:2].mean(axis=-1)
        ax.imshow(np.clip(img_rsz*1.1, 0, 1), interpolation="nearest")  
        outlines = utils.outlines_list(masks_true_rsz, multiprocessing=False)
        for outline in outlines:
            ax.plot(outline[:, 0], outline[:, 1], color='w', lw=1)

        outlines = utils.outlines_list(masks_pred, multiprocessing=False)
        for outline in outlines:
            ax.plot(outline[:, 0], outline[:, 1], color=outcols[i], lw=2, linestyle=":")
        ycent, xcent = img_rsz.shape[0]//2, img_rsz.shape[1]//2
        ax.set_ylim([ycent-55, ycent+55])
        ax.set_xlim([xcent-80, xcent+80])
        ax.axis('off')
        if i==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
        ax.text(1, -0.01, f"AP@0.9={aps[0][i][iex,-2]:.2f}", transform=ax.transAxes, ha="right", va="top")
        ax.set_title(xnames[i], loc="center", color=outcols[i])

    transl = mtransforms.ScaledTranslation(-40 / 72, 6 / 72, fig.dpi_scale_trans)
    for j in range(len(algs)):
        ax = plt.subplot(grid[0, -(3-j)])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1]+0.03, pos[2], pos[3]-0.03])
        for i, xl in enumerate(xlabels[1:4]):
            ax.plot(np.arange(0.5, 1, 0.05), aps[j][i].mean(axis=0), color=outcols[i], label=xl)
        ax.set_ylim([0, 1])
        ax.set_xlabel("IoU threshold")
        ax.set_title(alg_dict[algs[j]]["name"], loc="center")
        ax.set_xlim([0.5, 0.95])
        ax.set_xticks(np.arange(0.5, 1, 0.1))
        if j==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.set_ylabel("average precision (AP)")

    if save_fig:
        fig.savefig("figures/supp_invariance.pdf", dpi=150)


from tqdm import trange

def finetune_results(root, dset, load_3D=False, load_all=False):
    masks_pred_all = [[[], [], [], [], [], [], [], [], [], [], []], 
                    [[], [], [], [], [], [], [], [], [], [], []]]
    if dset == "root" or dset == "ovules":
        folder = Path(root / f"root_ovules_wolny/{dset}/")
    else:
        folder = Path(root / dset)
        
    files = natsorted(list((folder / "models/").glob("*.npy")))
    
    if load_3D:
        files = [f for f in files if "3D" in f.name and f.name[:3] != "sam"]
    else:
        files = [f for f in files if "3D" not in f.name and f.name[:3] != "sam"]
    ntrains_files = np.hstack(([-1], 2**np.arange(0, 9), [0]))
    nrois = np.nan * np.zeros((5, len(ntrains_files)), "int")
    print(ntrains_files)
    ntrains = ntrains_files.copy()
    i0 = 0
    for i in trange(len(files)):
        file = files[i]
        fparams = str(file.stem).split("_")
        model_type = fparams[0] 
        seed = int(fparams[2])
        ntrain = int(fparams[4])
        dat = np.load(file, allow_pickle=True).item()
        itrain = (ntrains_files == ntrain).nonzero()[0]
        if len(itrain) == 0:
            continue 
        else:
            itrain = itrain[0]
        
        if not load_all and model_type[:6] == "cpdino": 
            continue

        itype = 0 if model_type == "cyto3" else 1 if model_type == "cpsam" else 2 if model_type == "cpdino" else 3
    
        if i0 == 0:
            ntest = dat["ap"].shape[0]
            aps = np.nan * np.zeros((4, len(ntrains_files), 5, ntest, 10))
            tps = np.nan * np.zeros((4, len(ntrains_files), 5, ntest, 10))
            fps = np.nan * np.zeros((4, len(ntrains_files), 5, ntest, 10))
            fns = np.nan * np.zeros((4, len(ntrains_files), 5, ntest, 10))
    
        aps[itype, itrain, seed] = dat["ap"]
        tps[itype, itrain, seed] = dat["tp"]
        fps[itype, itrain, seed] = dat["fp"]
        fns[itype, itrain, seed] = dat["fn"]
        if ntrain == 0 and "3D" not in file.name:
            ntrains[-1] = dat["ntrain_masks"]
            nrois[:, -1] = dat["nrois"]
        if seed == 0 and itype==1 and "3D" not in file.name:
            masks_pred_all[itype][itrain].extend(dat["test_masks_pred"])
        if itype==1 and "3D" not in file.name:
            nrois[seed, itrain] = dat["nrois"]
        
        i0 += 1

    ntrains[0] = 0
    nrois[:,0] = 0

    return folder, aps, tps, fps, fns, masks_pred_all, ntrains, nrois



dset_names = ["BlastoSPIM (Nunley et al 2024)", "PlantSeg: lateral root (Wolny et al 2020)", "PlantSeg: ovules (Wolny et al 2020)"]

def fig4(root, save_fig=False):
    
    ntrains_files = np.hstack(([-1], 2**np.arange(0, 9), [0]))
    colors = ["g", [0.7,0.4,1]]
    fig = plt.figure(figsize=(14, 7), dpi=150)
    grid = plt.GridSpec(3, 10, hspace=.4, wspace=0.2, top=0.93, bottom=0.07, left=0.02, right=0.98) 

    iexs = [22, 26, 141]
    iexs_3D = [3, 0, 0]
    ylims = [[50, 425], [0, 130], [40, 420]]
    xlims = [[0, 500], [210, 389], [0, 500]]  
    il = 0
    transl = mtransforms.ScaledTranslation(-15 / 72, 18 / 72, fig.dpi_scale_trans)    
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=grid[:, :3], wspace=0.05, hspace=0.45)
    grid2 = [matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid[:, 3:5], wspace=0.4, hspace=0.45),
            matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid[:, -2:], wspace=0.4, hspace=0.45)]
    grid3 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=grid[:, -5:-2], wspace=0.4, hspace=0.45)
    algs = ["cyto3","cpsam", "cpdino", "cpdino-vitb"]
    for d, dset in enumerate(["blastospim", "root", "ovules"]):
        iex, ylim, xlim = iexs[d], ylims[d], xlims[d]
        dset_name = dset_names[d]

        folder, aps, tps, fps, fns, masks_pred_all, ntrains, nrois = finetune_results(root, dset, load_3D=False, load_all=True)
        print(folder)
        test_files = (folder / "test").glob("*.tif")  
        test_files = natsorted([tf for tf in test_files if "_masks" not in str(tf)])
        img_path = test_files[iex]
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
                outlines = utils.outlines_list(masks, multiprocessing=False)
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
                folder, aps, tps, fps, fns, masks_pred_all, _, _ = finetune_results(root, dset, load_3D=True, load_all=True)
            ax = plt.subplot(grid2[k][d])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]+0.2*pos[2]+0.14*pos[2]*(k==1), pos[1]-0.07*pos[3], pos[2]*0.7, pos[3]*1.1])
            accs = np.nanmean(aps[:,:,:,:,0], axis=-2)
            frac = 1 if d!=2 else 0.1
            yy = [3, 0, 1, 2]
            for j in range(len(algs)):
                ax.errorbar(np.nanmean(nrois[:, 1:], axis=0) * frac, 
                            np.nanmean(accs[j,1:], axis=-1), np.nanstd(accs[j,1:], axis=-1) / (accs.shape[2]-1)**0.5,
                            color=alg_dict[algs[j]]["color"], lw=1)
                ax.errorbar(1, np.nanmean(accs[j,0]), marker="o", markersize=5, 
                            color=alg_dict[algs[j]]["color"])
                if d==0:
                    ax.text(1.05, 0.5-yy[j]*0.13, alg_dict[algs[j]]["name"], color=alg_dict[algs[j]]["color"], 
                            transform=ax.transAxes, ha="right")
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


def fig5(root, save_fig=False): 

    leader_folders = ["cellposeSAM", "Amirreza_Mahbod", "IIAI",  "SharifHooshPardaz", "SJTU_426",]
    aps = []
    errors = []
    pqs = []
    for lfolder in leader_folders:
        dat2 = np.load(root / f"results/monusac_{lfolder}.npy", allow_pickle=True).item()
        aps.append(np.array(dat2["aps"]))
        errors.append(np.array(dat2["errors"]))
        pqs.append(np.array(dat2["pqs"]))
        if lfolder=="cellposeSAM":
            dat = dat2

    aps = np.array(aps)
    errors = np.array(errors)
    pqs = np.array(pqs)
    print(np.nanmean(errors, axis=-1).mean(axis=-1))
    print(np.nanmean(aps, axis=-1).mean(axis=-1))
    print(np.nanmean(pqs, axis=-1).mean(axis=-1))
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
    grid = plt.GridSpec(3, 7, hspace=0., wspace=0.1, top=0.99, bottom=0.01, left=0.01, right=0.99)
    il = 0
    for i, iex in enumerate(iexs):
        iap = [i for i, folder in enumerate(folders) if folder in img_files[iex].name][0]
        ax = plt.subplot(grid[i//3, i%3])
        if i==0:
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1]-0.1*pos[3], pos[2], 0.4*pos[3]])
        else:
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-0.0*(i%3), pos[1]-0.05*pos[3]*(i<6), pos[2], pos[3]])
        
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
        outlines = utils.outlines_list(masks_true0, multiprocessing=False)
        for j, outline in enumerate(outlines):
            ax.plot(outline[:,0], outline[:,1], color=class_colors_true[tid[j]], lw=1.5, ls="-")

        if 1:
            outlines = utils.outlines_list(masks_pred0, multiprocessing=False)
            for j, outline in enumerate(outlines):
                ax.plot(outline[:,0], outline[:,1], color=class_colors_pred[cid[j]], lw=2.5, ls="--",
                        dashes=(1.5,2))

        ax.axis("off")
        if i==0:
            #ax.text(0.1, 3, "MoNuSAC 2020 challenge: segmentation and classification", fontsize="large", fontstyle="italic", transform=ax.transAxes)
            ax.text(0.15, 2.5, "cell classes:", transform=ax.transAxes)
            for k, cname in enumerate(cl_names):
                ax.text(0.25, 2.2-k*0.3, cname, color=class_colors_true[k], transform=ax.transAxes)
            transl = mtransforms.ScaledTranslation(-0 / 72, 55 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl, fs_title)

    alg_names = ["CellposeSAM", "PL1", "PL2", "PL3", "L2"]
    colors = [[0.7,0.4,1], 0.5*np.ones(3), 0.5*np.ones(3), 0.5*np.ones(3), 0.5*np.ones(3)]
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[:, 3:], wspace=0.8, hspace=0.45)
    for k in range(3):
        ax = plt.subplot(grid1[0, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]+0.03*(2-k), pos[1]+0.25*pos[3], pos[2], pos[3]*0.5])
        eps = errors if k==0 else aps if k==1 else pqs
        for j in range(5):
            vp = ax.violinplot(np.nanmean(eps[j], axis=-1), showmeans=True, showmedians=False, showextrema=False, positions=[j])
            vp["bodies"][0].set_facecolor(colors[j])
            vp["bodies"][0].set_alpha(0.35)    
            ax.plot(j + 0.3*np.array([-1,1]), np.nanmean(eps[j], axis=-1).mean()*np.ones(2), color=colors[j], lw=3)
        if k==0:
            ax.set_ylabel("error rate @ 0.5 IoU")
            ax.set_ylim([0., 0.8])
            ax.set_yticks(np.arange(0,0.85,0.2))
        elif k==1:
            ax.set_ylabel("average precision @ 0.5 IoU")
            ax.set_ylim([0., 1])
            ax.set_yticks(np.arange(0,1.05,0.2))
        else:
            ax.set_ylabel("panoptic quality @ 0.5 IoU")
            ax.set_ylim([0., 1])
            ax.set_yticks(np.arange(0,1.05,0.2))

        ax.set_xticks(np.arange(0, aps.shape[0]))
        ax.set_xticklabels(alg_names, fontsize="small", rotation=90)#, rotation=20, ha="right")
        for j, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(colors[j])
        #ax.set_ylim(0.45, 0.95)
        ax.set_xlim([-0.5, 4.5])    
        transl = mtransforms.ScaledTranslation(-40 / 72, 25 / 72, fig.dpi_scale_trans)
        il = plot_label(ltr, il, ax, transl, fs_title)

        from scipy.stats import wilcoxon
        axin = ax.inset_axes([0, 1.02, 1, 0.14])
        l0 = 0
        for j in range(1,5):
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

def supp_sim(root, save_fig=False):
    nc = 300
    ntrue = 100
    nfalse = nc - ntrue
    np.random.seed(0)
    err_results = []
    for ntrue in np.arange(25, 401, 5):
        nc = int(ntrue*2)
        nfalse = nc - ntrue
        for ptrue in np.arange(0.8, 1.0, 0.0025):
            pfp = 1 - ptrue
        
            nann = 2
            ntrue_keep = min(int(ntrue * ptrue), ntrue - 1)
            nfp = max(int(ntrue * pfp), 1)

            itrue = np.argpartition(np.random.random((nann, ntrue)), ntrue_keep - 1, axis=1)[:, :ntrue_keep]
            ifalse = np.argpartition(np.random.random((nann, nfalse)), nfp - 1, axis=1)[:, :nfp] + ntrue
            #print(ifalse.max(), ifalse.min())
            all_anns = np.concatenate((itrue, ifalse), axis=1)

            tps = np.zeros((nann, nann), int)
            fps = np.zeros((nann, nann), int)
            fns = np.zeros((nann, nann), int)
            for i in range(nann):
                for j in range(nann):
                    if i == j:
                        continue
                    tps[i, j] = np.isin(all_anns[j], all_anns[i]).sum()
                    fps[i, j] = len(all_anns[j]) - tps[i, j]
                    fns[i, j] = len(all_anns[i]) - tps[i, j]
                    
            errs = (fps + fns) / (tps + fns)
            err_ha = ((ntrue - ntrue_keep) + nfp) / ntrue
            err_results.append((ntrue, ptrue, np.nanmean(errs[1,0]), err_ha))

    err_results = np.array(err_results)

    x, y = err_results[:, -2:].T.copy()
    a, b = np.polyfit(x, y, 1)
    print(a, b)
    res = y - (a * x + b)

    ii = x < 0.15
    a, b = np.polyfit(x[ii], y[ii], 1)
    print(a, b)

    fig = plt.figure(figsize=(14, 5.5))
    il = 0
    grid = plt.GridSpec(2, 4, wspace=0.6, hspace=0.6)

    ax = plt.subplot(grid[0, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2]*0.9, pos[3]])
    il = plot_label(ltr, il, ax, mtransforms.ScaledTranslation(-15 / 72, 10 / 72, fig.dpi_scale_trans))

    pos = np.random.choice(15**2, size=40, replace=False)
    pos = np.vstack((pos // 15, pos % 15)).T 
    post = pos[:20] + np.random.randn(20, 2) * 0.25
    posf = pos[20:] + np.random.randn(20, 2) * 0.25

    ax.scatter(post[:, 0], post[:, 1], s=40, edgecolor='k', 
            facecolor='none', marker='o', lw=2, label='ground-\ntruth')
    col = [0.65, 0, 0]
    ax.scatter(posf[:, 0], posf[:, 1], s=40, edgecolor=col, 
            facecolor=col, marker='o', lw=2, label='possible\nfalse\npositives')
    ax.set_title('Simulation', loc='center')
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    leg = ax.legend(loc=(1, 0.4), frameon=False, handletextpad=0.1, borderpad=0.1)
    leg.get_texts()[1].set_color(col)

    for j in range(2):
        ax = plt.subplot(grid[0, j+1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]-0.01*(j+1), pos[1], pos[2]*0.9, pos[3]])
        if j==0:
            ntp = 17
            nfp = 3
            tps = post[:ntp]
            fps = posf[:nfp]
            fns = post[ntp:]
        else:
            ntp = 14
            nfp = 6
            tps = post[:ntp]
            fps = np.vstack((post[ntp:ntp+3], posf[:nfp-3]))
            fns = np.vstack((post[ntp+3:], posf[-3:]))
        ax.scatter(tps[:, 0], tps[:, 1], s=40, edgecolor='k', 
                facecolor='none', marker='o', lw=2, label='')
        ax.scatter(fps[:, 0], fps[:, 1], s=40, edgecolor='r', 
                facecolor='r', marker='o', lw=2, label='false\npositives')
        ax.scatter(fns[:, 0], fns[:, 1], s=40, color='r', marker='x', lw=2, 
                label='false\nnegatives')
        ax.set_yticks([]); ax.set_xticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        if j == 0:
            leg = ax.legend(loc=(1, 0.5), frameon=False, handletextpad=0.1, borderpad=0.1)
            leg.get_texts()[0].set_color('r')
            leg.get_texts()[1].set_color('r')
            ax.set_title('Annotator 1 to ground-truth', loc='center')
        else:
            ax.set_title('Annotator 1 to Annotator 2', loc='center')


    ax = plt.subplot(grid[0, 3])
    il = plot_label(ltr, il, ax, mtransforms.ScaledTranslation(-30 / 72, 10 / 72, fig.dpi_scale_trans))
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.03, *pos[1:]])
    ax.scatter(x, y, alpha=0.05, s=3, color=0.5*np.ones(3))
    ax.plot([0, 0.8], [0, 0.4], 'k--')
    ax.scatter(0.26, 0.13, s=80, color='y', marker='x', lw=3, zorder=50)
    ax.text(0.3, 0.2, 'Cellpose\ntest set', fontsize='small', 
            va='bottom', ha='right', color='y')
    ax.text(1, 0.25, 'y = 0.5 x', fontsize='small',
            transform=ax.transAxes, ha='right')
    ax.axis('square')
    ax.set_ylabel('Annotator 1 to ground-truth')
    ax.set_xlabel('Annotator 1 to\nAnnotator 2')
    ax.set_title('Error rate', loc='center')

    ax = plt.subplot(grid[1:5, -3:-1])
    il = plot_label(ltr, il, ax, mtransforms.ScaledTranslation(-20 / 72, 18 / 72, fig.dpi_scale_trans))
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.32*pos[2], pos[1], pos[2]*1.58, pos[3]])
    dataset = "cyto2"
    algs = ["cyto3", "cellsam", "samcell", "microsam", "cpdino-vitb", "cpdino", "cpsam"]
    alg_names = [alg_dict[alg]["name"].replace(" ", "\n").replace("-", "\n") for alg in algs]
    ind_im = np.array([68, 69, 71, 72, 73, 74, 75, 76, 84, 86, 89, 90])
    ind_im = np.concatenate((np.arange(55, dtype = 'int32'), ind_im), 0)
    masks_H2 = [np.load(f"/media/carsen/disk1/datasets_cellpose/images_cyto2/labels2/{i:03d}_img_seg.npy", allow_pickle=True).item()["masks"] 
        for i in ind_im]
    aps, tps, fps, fns = [], [], [], []
    errors = []
    masks_preds = []
    runtimes = []

    from pathlib import Path
    for alg in algs:
        dat = np.load(root / f"results/{alg}_{dataset}.npy", allow_pickle=True).item()
        threshold = np.arange(0.5, 1., 0.05)
        masks_true = dat["masks_true"]
        masks_pred = dat["masks_pred"]
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=threshold)
        err = (fp + fn) / (fn + tp)
        ap2, tp2, fp2, fn2 = metrics.average_precision(masks_H2, masks_pred, threshold=threshold)
        err2 = (fp2 + fn2) / (fn2 + tp2)
        errors.append(np.stack((err[:,0], err2[:,0]), axis=-1))
    errors = np.array(errors)
    print(errors.mean(axis=1))

    for i in range(len(algs)):
        for j in range(2):
            vp = ax.violinplot(errors[i, :, j], positions=[2*i+j], widths=0.6, showmeans=False, 
                                                showextrema=False)
            vp["bodies"][0].set_facecolor(alg_dict[algs[i]]["color"])
            vp["bodies"][0].set_alpha(0.35 if j==0 else 0.25)
            ax.plot(2*i + j + 0.3*np.array([-1,1]), errors[i, :, j].mean()*np.ones(2), 
                    color=alg_dict[algs[i]]["color"], lw=3, ls='-' if j==0 else ':')

    ax.set_ylim([0, 0.8])

    # use xticks and xticks minor to label the algorithm and the annotator
    ax.set_xticks(np.arange(0, len(algs)*2, 1), minor=True)
    ax.set_xticks(np.arange(0, len(algs)*2, 2)+0.5, minor=False)
    labels = []
    for i in range(len(algs)):
        labels.extend(['Ann1', alg_names[i], 'Ann2'])
    ax.set_xticklabels(['Ann1', 'Ann2']*len(algs), rotation=0, minor=True)
    ax.set_xticklabels(alg_names, rotation=0, minor=False)
    # ax.set_xticks(np.arange(1, len(algs)*2, 2))

    va = [ 0, -.05, 0, -.05, -.05, -.05 ]
    for i, tl in enumerate(ax.get_xticklabels(minor=False)):
        tl.set_y(-0.15)
        tl.set_color(alg_dict[algs[i]]["color"])

    ax.tick_params(axis='x', which='major', length=0)

    ax.set_ylabel('Error rate')
    ax.set_title("Performance on Cellpose test set", fontstyle="italic", y=1.1, x=-0.0)
    
    if save_fig:
        fig.savefig("figures/supp_sim.pdf", dpi=150)

def supp_bench(root, save_fig=False):
    algs = ["cpsam", "cpdino", "cpdino-vitb", "cyto3", "cellsam", "microsam", "samcell", "omnipose",  "pathosam"]
    
    for bk in range(2):
        if bk==1:
            dsets = ["tissuenet", "livecell", "bact_phase", "bact_fluor", "deepbacs", "monuseg"]
            dset_names = ["Tissuenet", "Livecell", "Omnipose (PhC)", "Omnipose (fluor)", "DeepBacs", "MoNuSeg"]
        else:
            dsets = ["root", "ovules", "blastospim"]
            dset_names = ["PlantSeg", "Blastospim"]
        aps, tps, fps, fns, errors = [], [], [], [], []
        masks_preds = []
        runtimes = []
        for d, dset in enumerate(dsets):
            # if k==0:
            #     import benchmarks
            #     files, imgs, masks_true = benchmarks.load_dataset(dset, root=root / "..")
            algs0 = algs.copy()#algs[:4].copy() if k==0 else algs.copy()
            for i, alg in enumerate(algs0):
                if Path(root / f"results/{alg}_{dset}.npy").exists():
                    dat = np.load(root / f"results/{alg}_{dset}.npy", allow_pickle=True).item()
                else:
                    continue
                ap0 = dat["ap"]
                tp0 = dat["tp"]
                fp0 = dat["fp"]
                fn0 = dat["fn"]
                
                inds = np.arange(len(ap0))
                ap = ap0[inds]
                tp = tp0[inds]
                fp = fp0[inds]
                fn = fn0[inds]
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
        if bk==0:
            dsets = dset_names
            aps[0] = np.concatenate((aps[0], aps[1]), axis=1)
            tps[0] = np.concatenate((tps[0], tps[1]), axis=1)
            fps[0] = np.concatenate((fps[0], fps[1]), axis=1)
            fns[0] = np.concatenate((fns[0], fns[1]), axis=1)
            errors[0] = np.concatenate((errors[0], errors[1]), axis=1)
            del aps[1], tps[1], fps[1], fns[1], errors[1]
                                    

        if bk==1:
            fig = plt.figure(figsize=(14,8), dpi=150)
            grid = plt.GridSpec(2, 7, hspace=1, wspace=1, top=0.96, bottom=0.04, left=0.04, right=0.96)
        else:
            fig = plt.figure(figsize=(14*0.4,8), dpi=150)
            grid = plt.GridSpec(2, 7, hspace=1, wspace=1, top=0.96, bottom=0.04, left=0.1, right=0.98)
        
        for j in range(2):
            ax = plt.subplot(grid[j,:])
            pos = ax.get_position().bounds 
            ax.set_position([pos[0]+0.035, pos[1] - 0.1*(j==0) - 0.02*(j==1), pos[2]*0.965, pos[3]])
            axin = ax.inset_axes([0., 1.02, 1, 0.24 if bk==1 else 0.18])
            xticks, xticklabels, xtickcolors = [], [], []
            k = 0
            for d in range(len(dsets)):
                ips = np.nonzero(~np.isnan(aps[d][:,0,0]))[0]
                algd = np.array(algs)[ips]
                if j==1:
                    eps = aps[d][ips,:,0]
                else:
                    eps = errors[d][ips,:,0]
                for i in range(len(ips)):
                    vp = ax.violinplot(eps[i], positions=[k], widths=0.6, showmeans=True, 
                                showextrema=False)
                    vp["bodies"][0].set_facecolor(alg_dict[algd[i]]["color"])
                    vp["bodies"][0].set_alpha(0.35)
                    if i > 0:
                        p = wilcoxon(eps[0], eps[i], alternative="two-sided").pvalue 
                        print(dsets[d], alg_dict[algd[i]]["name"], f"p={p:.1e}", f"{eps[i].mean():.4f}")
                        #if "bact" in dsets[d] and ips[i]==3:
                            #print(wilcoxon(eps[0], eps[i], alternative="two-sided").pvalue)#"greater" if j==0 else "less").pvalue )

                        axin.plot([k-i, k], np.ones(2)*(0.95 + (len(ips)-i)*0.02), lw=1, color="k")
                        pstr = "n.s." if p > 0.05 else ("*" if p >= 0.01 else "**" if p >= 0.001 else "***")
                        axin.text(k - i/2, 0.95 + (len(ips)-i)*0.02 + 0.01*(p>0.05), 
                                    pstr, ha="center", va="center", fontsize="small" if p > 0.05 else "large")
                    else:
                        print(f"{eps[0].mean():.4f}")
                    ax.plot(k + 0.3*np.array([-1,1]), eps[i].mean()*np.ones(2), color=alg_dict[algd[i]]["color"], lw=3)
                    xticks.append(k)
                    xticklabels.append(alg_dict[algd[i]]["name"].replace("\n", " "))
                    xtickcolors.append(alg_dict[algd[i]]["color"])
                    k += 1
                if j==0:
                    ax.text(k - len(ips)/2 - 0.5, 0.98 if bk==1 else 1.8, f"{dset_names[d]}\nn={len(eps[0]):,}", ha="center", va="bottom", 
                    fontsize="large")
                k+=1.5   
            if j==1:
                ax.set_ylabel("     average precision (AP) @ 0.5 IoU", fontsize="large")
                if bk==1:
                    ax.set_ylim([0.4, 1.0])
                else:
                    ax.set_ylim([0.2, 0.9])
                ax.set_xticks([])
            else:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, rotation=45, ha="right")
                for i, color in enumerate(xtickcolors):
                    text = xticklabels[i]
                    ax.get_xticklabels()[i].set_color(color)
                    ax.get_xticklabels()[i].set_fontweight("bold" if "SAMCell" in text or "Omnipose" in text or "PathoSAM" in text else "normal")
                ax.set_ylabel("error rate @ 0.5 IoU", fontsize="large")
                if bk==1:
                    ax.set_ylim([0, 0.8])
                else:
                    ax.set_ylim([0., 1.5])
                k = 0
            
            ax.spines["bottom"].set_visible(False)
            ax.grid(True, color=0.8*np.ones(3), lw=0.5, ls="--", axis="y")
            axin.axis("off")

        if bk==1:
            fig.savefig("figures/supp_bench.pdf", dpi=150)
        else:
            fig.savefig("figures/supp_ood.pdf", dpi=150)

        plt.show()
        

def supp_noise(root, save_fig=False):
    files, imgs, masks_true = load_dataset("cyto2", root=root / "..")
    
    fig = plt.figure(figsize=(14*0.6, 7*2./3), dpi=150)
    grid = plt.GridSpec(2, 8, hspace=0.4, wspace=0.2, top=0.93, bottom=0.07, left=0.03, right=0.97)
    il = 0
    transl = mtransforms.ScaledTranslation(-14 / 72, 6 / 72, fig.dpi_scale_trans)        

    algs = ["cpsam"]#, "cpdino", "cpdino-vitb", "cellsam", "microsam"]

    iex = 2
    nstr = ["Poisson noise", "blur", "pixel size", "anisotropic blur"]
    nstr = np.array(nstr)[[0, 2, 1, 3]]
    rstr = ["denoising", "deblurring", "upsampling", "anisotropic\ndeconvolution"]
    rstr = np.array(rstr)[[0, 2, 1, 3]]
    xstr = ["noise", "blur", "pixel size", "anisotropic blur"]
    xstr = np.array(xstr)[[0, 2, 1, 3]]
    for ii, noise_type in enumerate(["poisson", "downsample", "blur", "aniso"]):
        aps, masks_preds = [], []
        for alg in algs:
            aps.append([])
            masks_preds.append([])
            for nl in range(3):
                dat = np.load(root / f"results/{alg}_cyto2_{noise_type}_{nl}.npy", allow_pickle=True).item()
                aps[-1].append(dat["ap"])
                masks_preds[-1].append(dat["masks_pred"])
                
        aps = np.array(aps)

        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[ii//2, (ii%2)*4:4+4*(ii%2)],
                                                            wspace=0.1, hspace=0.2)

        if noise_type=="poisson":
            param = np.array([5, 2.5, 0.5])
        elif noise_type=="blur":
            param = np.array([2, 4, 8])# 48])
        elif noise_type=="downsample":
            param = np.array([2, 5, 10])
        else:
            param = np.array([2, 6, 12])

        for i in range(3):
            k = i
            ax = plt.subplot(grid1[0, i])
            pos = ax.get_position().bounds
            ax.set_position([pos[0]-0.01*i, pos[1], pos[2], pos[3]])
            img = np.maximum(0, imgs[iex].copy().astype("float32"))
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
            denoise.deterministic()
            img = denoise.add_noise(torch.from_numpy(img).unsqueeze(0), 
                                    **params).cpu().numpy().squeeze()
            if noise_type=="downsample":
                img = img[:,::param[k], ::param[k]]
            elif noise_type=="aniso":
                img = img[:,::param[k]]
            img_rsz = img.transpose(1,2,0).copy()
            Ly, Lx = img_rsz.shape[:2]
            img_rsz = np.concatenate((np.zeros_like(img_rsz[:,:,:1]), img_rsz), axis=-1)
            #import cv2
            # masks_true_rsz = cv2.resize(masks_true[iex], (Lx, Ly), interpolation=cv2.INTER_NEAREST)
            # masks_pred_rsz = cv2.resize(masks_preds[0][i][iex], (Lx, Ly), interpolation=cv2.INTER_NEAREST)

            masks_true0 = masks_true[iex] 
            masks_pred0 = masks_preds[0][i][iex]
            
            if ii!=0 or (ii==0 and i==0):
                vmax = 1
            else:
                vmax = 1.1 if i==1 else 1.3
            ax.imshow(np.clip(img_rsz[:,:,:]*vmax, 0, 1), aspect=1 if noise_type!="aniso" else param[k], 
                    interpolation="nearest")#.transpose(1,0,2))  
            for o in range((i>0), 2):
                outlines_list = utils.outlines_list(masks_true0, multiprocessing=False) if o==0 else utils.outlines_list(masks_pred0, multiprocessing=False)
                for outline in outlines_list:
                    rsz_Y = param[k] if noise_type=="downsample" else 1. 
                    rsz_X = param[k] if noise_type=="downsample" or noise_type=="aniso" else 1.
                    ax.plot(outline[:, 0]/rsz_Y, outline[:, 1]/rsz_X, color=outcols[o], lw=1.5 if o==0 else 2, linestyle="-" if o==0 else "--")
                    
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
            
        # ax = plt.subplot(grid1[0, 3])
        # pos = ax.get_position().bounds
        # ax.set_position([pos[0]-0.025, pos[1]+0.15*pos[3], pos[2], pos[3]*0.9])
        # for j in range(len(algs)):
        #     ax.errorbar(np.arange(3), aps[j, :,:,0].mean(axis=-1).T, aps[j, :,:,0].std(axis=-1).T / (66**0.5),
        #                 color=alg_dict[algs[j]]["color"])
        #     if ii==0:
        #         xpos = 1.2 if j<3 else 0.05
        #         ypos = 1.07-j*0.15 if j<3 else 0.3-(j-3)*0.15
        #         ax.text(xpos, ypos, alg_dict[algs[j]]["name"].replace("-","\n"), fontsize="medium",
        #                 color=alg_dict[algs[j]]["color"], transform=ax.transAxes, 
        #                 ha="right" if j<3 else "left", va="top")
        # ax.set_ylabel("AP @ 0.5 IoU")
        # ax.set_xticks([0, 1, 2])
        # ax.set_ylim([0, 1])
        # ax.set_yticks([0, 0.5, 1.0])
        # ax.set_xticklabels(["low", "medium", "high"])
        # ax.set_xlabel(xstr[ii])
        # print(nstr[ii], aps[:,:,:,0].mean(axis=-1))

    if save_fig:
        fig.savefig("figures/supp_noise.pdf", dpi=150)

def supp_realnoise(root, save_fig=False):
    
    dsets = ["flywing", "tribolium", "ribo"]

    fig = plt.figure(figsize=(14*2/3, 8), dpi=150)
    grid = plt.GridSpec(3, 5, hspace=0.3, wspace=0.05, top=0.98, bottom=0.09, left=0.05, right=0.99)
    il = 0

    dset_names = ["Drosophila wing epithelia", "Tribolium nuclei", "Two-photon calcium imaging"]
    for d, dset in enumerate(dsets):
        folder = root / f"../{dset}"
        dat0 = np.load(folder / "cp_masks.npy", allow_pickle=True).item()
        clean = dat0["clean"]
        noisy = dat0["noisy"]
        masks_true = dat0["masks_clean"].copy()

        dat = np.load(root / f"results/cpsam_{dset}.npy", allow_pickle=True).item()
        ap = dat["ap"].reshape(len(dat0["noisy"]), -1, dat["ap"].shape[-1])

        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=grid[d, :],
                                                        wspace=0.07, hspace=0.0)

        iex = 6 if d==0 else 0 if d==1 else 0
        ylim = [350, 500]
        xlim = [0+(d==1)*200, 520+(d==1)*200]
        transl = mtransforms.ScaledTranslation(-18 / 72, 18 / 72, fig.dpi_scale_trans)
        maskst = masks_true[iex].copy().T if d!=1 else masks_true[iex].copy()
        outlines_gt = utils.outlines_list(maskst, multiprocessing=False)
        titlesd = (["High laser power (20mW)", "Low laser power (0.2mW)"] if d<2 else 
                    ["Clean (300 frames averaged)", "Noisy (single frame)"])
        
        nl = 0
        #print(ap[nl][:,0])
        for k in range(2):
            if k == 0:
                img = clean[iex].copy().T
            elif k == 1:
                img = noisy[nl][iex].copy().T
                maskk = dat["masks_pred"][iex].copy().T
                ap_iex = ap[nl][iex, 0]
            if d==1:
                img = img.T 
                maskk = maskk.T
            img = transforms.normalize99(img)
            #print(img.shape)

            ax = plt.subplot(grid1[0, k])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1] - 0.03, pos[2], pos[3]])
            ax.imshow(img, vmin=0, vmax=1, cmap="gray")
            ax.set_title(titlesd[k], color="k", fontsize="medium")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axis("off")
            if k == 0:
                ax.text(0, 1.33, dset_names[d], fontsize="large",
                        fontstyle="italic", transform=ax.transAxes)
                il = plot_label(ltr, il, ax, transl, fs_title)

            ax = plt.subplot(grid1[1, k])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1] - 0.03, pos[2], pos[3]])
            ax.imshow(img, vmin=0, vmax=1, cmap="gray")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axis("off")
            #ax.set_title("segmentation")
            if k == 0:
                for o in outlines_gt:
                    ax.plot(o[:, 0], o[:, 1], color=outcols[0], lw=1, ls="--")
                ax.text(0, -0.1, "ground-truth (Cellpose cyto3)", ha="left", 
                        va="top", color=outcols[0], transform=ax.transAxes)
            else:
                outlines = utils.outlines_list(maskk, multiprocessing=False)
                for o in outlines:
                    ax.plot(o[:, 0], o[:, 1], color=outcols[1], lw=1, ls="--")
                ax.text(1, -0.1, f"AP@0.5 = {ap_iex:.2f}", ha="right", va="top", transform=ax.transAxes)
                ax.text(0., -0.1, "CellposeSAM", ha="left", va="top", 
                        color=alg_dict["cpsam"]["color"], transform=ax.transAxes)

        algs = ["cpsam", "cpdino", "cpdino-vitb", "cellsam", "microsam", "cyto3", "cyto3_restore"]
        aps = []
        for alg in algs:
            if alg[:5]!="cyto3":
                try:
                    dat = np.load(root / f"results/{alg}_{dset}.npy", allow_pickle=True).item()
                except:
                    print(f"missing {alg} {dset}")
                    dat = {"ap": np.nan*dat["ap"]}
                ap = dat["ap"].reshape(len(dat0["noisy"]), -1, dat["ap"].shape[-1])
                aps.append(ap)
            elif alg=="cyto3_restore":
                aps.append(dat0["ap_denoised"] if "ap_denoised" in dat0 else dat0["ap_dn"])
            else:
                aps.append(dat0["ap_noisy"])

        
        ax = plt.subplot(grid1[:, -1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + 0.05, pos[1] - 0.03, pos[2] * 0.7, pos[3]])
        kk = [0, 1, 2, 6, 5, 4, 3]
        norder = [0, 2, 1] if d==0 else np.arange(len(aps[0]))
        x = np.arange(0, 3) if d < 2 else dat0["navgs"]
        print(x)
        for k in range(len(aps)):
            x0 = x.copy() + 0.01 * np.random.randn(len(x))
            alg0 = algs[k] if algs[k] != "cyto3_restore" else "cyto3"
            means = np.array([aps[k][nl][:, 0].mean(axis=0) for nl in norder])
            sems = np.array([aps[k][nl][:, 0].std(axis=0) / ((aps[k][nl][:, 0].shape[0]-1)**0.5) for nl in norder])
            print(dsets[d], alg0, means, sems)
            ax.errorbar(x0, means, sems, color=alg_dict[alg0]["color"], 
                        label=alg_dict[alg0]["name"], lw=2, zorder=30 if kk[k]==0 else 0,
                        ls="--" if algs[k]=="cyto3_restore" else "-", alpha=0.9)
            if d==0:
                alg_name = alg_dict[alg0]["name"] if algs[k]!="cyto3_restore" else "-- cyto3 + denoising"
                xpos = 1 #if kk[k] > 3 else 0.05
                ypos = 0.55 - (kk[k])*0.08 #if kk[k] > 3 else 1.2 - kk[k]*0.08
                ax.text(xpos, ypos, alg_name, fontsize="small",
                        color=alg_dict[alg0]["color"], transform=ax.transAxes, 
                        ha="right", va="top", fontweight="bold")

        ax.set_ylabel("AP @ 0.5 IoU threshold")
        ax.set_ylim([0, 1.0])
        ax.set_xlabel("laser power (mW)" if d < 2 else "# of frames averaged")
        if d==2:
            ax.set_xscale("log")
        ax.set_xlim([-0.1, 2.1] if d < 2 else [dat0["navgs"][0]*0.8, dat0["navgs"][-1]*2])
        ax.set_xticks([0, 1, 2] if d < 2 else dat0["navgs"])
        ax.set_xticklabels(["0.2", "0.3", "0.5"] if d < 2 else dat0["navgs"])
        # turn off minor ticks
        ax.tick_params(axis='x', which='minor', bottom=False)

    if save_fig:
        fig.savefig("figures/supp_realnoise.pdf", dpi=150)


        