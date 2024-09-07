"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os
from cellpose import io, metrics, transforms, denoise, models, plot
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm

from fig_utils import *

cols = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 1, 0.8], [0.5, 0.5, 0],
        [0, 0.2, 0.8], [0, 0.5, 0], [0, 0, 0]]
lss = ["-", "-", "-", "-", "-", "-", "-", "-", "--"]
titles = [
    "clean image", "noisy image", "noise2void", "noise2self", "reconstruction loss",
    "perceptual loss", "segmentation loss", "per. + seg. loss", "retrain w/ noisy"
]
legstr = [u"\u2013 %s" % titles[k] for k in range(len(titles))]
legstr[-1] = f"-- {titles[-1]}"


def load_benchmarks(folder, noise_type="poisson", ctype="cyto2",
                    thresholds=np.arange(0.5, 1.05, 0.05), others=True):
    nimg_test = 68 if ctype == "cyto2" else 111
    folder_name = ctype
    root = Path(f"{folder}/images_{folder_name}/")

    masks_all = []
    imgs_all = []
    dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                  allow_pickle=True).item()
    test_data = dat["test_data"][:nimg_test]
    test_noisy = dat["test_noisy"][:nimg_test]
    masks_noisy = dat["masks_noisy"][:nimg_test]
    masks_true = dat["masks_true"][:nimg_test]
    masks_data = dat["masks_orig"][:nimg_test]
    diam_test = dat["diam_test"][:nimg_test]
    noise_levels = dat["noise_levels"][:nimg_test]
    if noise_type == "downsample":
        test_noisy = [
            cv2.resize(tn[0],
                       (int(tn.shape[-1] / nl), int(tn.shape[-2] / nl)))[np.newaxis]
            for tn, nl in zip(test_noisy, noise_levels)
        ]
    if "flows_true" in dat:
        flows_true = dat["flows_true"][:nimg_test]
    else:
        flows_true = None
    masks_all.append(masks_data)
    masks_all.append(masks_noisy)
    imgs_all.append(test_data)
    imgs_all.append(test_noisy)

    if others:
        dat = np.load(root / "noisy_test" / f"test_{noise_type}_n2v.npy",
                      allow_pickle=True).item()
        test_n2v = dat["test_n2v"][:nimg_test]
        masks_n2v = dat["masks_n2v"][:nimg_test]
        imgs_all.append(test_n2v)
        masks_all.append(masks_n2v)

        dat = np.load(root / "noisy_test" / f"test_{noise_type}_n2s.npy",
                      allow_pickle=True).item()
        test_n2s = dat["test_n2s"][:nimg_test]
        masks_n2s = dat["masks_n2s"][:nimg_test]
        imgs_all.append(test_n2s)
        masks_all.append(masks_n2s)

    dat = np.load(root / "noisy_test" / f"test_{noise_type}_cp3_all.npy",
                  allow_pickle=True).item()
    istr = ["rec", "per", "seg", "perseg"]
    for k in range(len(istr)):
        if f"test_{istr[k]}" in dat:
            test_dn = dat[f"test_{istr[k]}"][:nimg_test]
            masks_dn = dat[f"masks_{istr[k]}"][:nimg_test]
            imgs_all.append(test_dn)
            masks_all.append(masks_dn)
            if istr[k] == "perseg":
                flows_perseg = dat["flows_perseg"][:nimg_test]

    if others:
        dat = np.load(root / "noisy_test" / f"test_{noise_type}_cp_retrain.npy",
                      allow_pickle=True).item()
        masks_retrain = dat["masks_retrain"][:nimg_test]
        masks_all.append(masks_retrain)

    # benchmarking
    aps = []
    tps = []
    fps = []
    fns = []
    for k in range(len(masks_all)):
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_all[k],
                                                   threshold=thresholds)
        aps.append(ap)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    aps, tps, fps, fns = np.array(aps), np.array(tps), np.array(fps), np.array(fns)

    return (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true,
            diam_test, flows_perseg, flows_true)


def draw_net(fig, ax, arrow_in=True, arrow_out=True):
    ys = [[0.25, 0.5, 0.75], np.arange(0, 1.25, 0.25), [0]]
    for l in range(3):
        for b in range((l == 1), 5 - (l == 1)):
            x = (l + 1) / 2 if l < 2 else l / 2
            for y in ys[l]:
                yi = b / 4 if l == 2 else y
                ax.annotate(
                    f" ", (x, yi), (l / 2, b / 4), xycoords=ax.transAxes,
                    textcoords=ax.transAxes, bbox=dict(boxstyle="circle", fc="w"),
                    arrowprops=dict(width=0.25, color="k", headlength=3, headwidth=4),
                    ha="center", va="center", annotation_clip=True)
    ax.axis("off")
    pos = ax.get_position().bounds
    dl = [-0.35, 1.1]
    for l in range(2):
        if (l == 0 and arrow_in) or (l == 1 and arrow_out):
            ax = fig.add_axes([
                pos[0] + dl[l] * pos[2], pos[1] + 0.1 * pos[3], pos[2] * 0.2,
                pos[3] * 0.8
            ])
            ax.annotate(f" ", (1, 0.5), (0, 0.5), xycoords=ax.transAxes,
                        textcoords=ax.transAxes,
                        arrowprops=dict(width=2, color="k", headlength=5, headwidth=7),
                        ha="center", va="center", annotation_clip=True)
            ax.axis("off")


def fig1(folder, save_fig=False):
    thresholds = np.arange(0.5, 1.05, 0.05)
    out = load_benchmarks(folder, noise_type="poisson", ctype="cyto2",
                          thresholds=thresholds)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
     flows_perseg, flows_true) = out

    fig = plt.figure(figsize=(14, 10), dpi=100)
    yratio = 14 / 10
    grid = plt.GridSpec(4, 5, figure=fig, left=0.02, right=0.98, top=0.96, bottom=0.04,
                        wspace=0.25, hspace=0.65)
    transl = mtransforms.ScaledTranslation(-15 / 72, 7 / 72, fig.dpi_scale_trans)

    il = 0

    iex = 2
    ylim = [110, 290]  #[10,310]
    xlim = [130, 360]  #[100,500]
    #iex = 17
    #ylim = [50, 340]
    #xlim = [0, 430]

    imgs = [imgs_all[k][iex] for k in range(len(imgs_all))]
    masks = [masks_all[k][iex] for k in range(len(masks_all))]

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=grid[:, :-1],
                                                        wspace=0.15, hspace=0.15)
    k = 0
    imgk = imgs[k].squeeze().copy() * 1.2
    titlek = titles[k]
    ax = plt.subplot(grid1[0, k])
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.imshow(imgk, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.text(0.5, 1.05, titlek, color=cols[k], ha="center", transform=ax.transAxes,
            fontsize="large")

    x = [0, 0, 2, 3, 3]
    y = [1, 3, 0, 0, 2]
    kk = [1, 2, 4, 6, 7]
    titlesd = titles.copy()
    titlesd[7] = "perceptual +\nsegmentation loss"
    mask_gt = masks_true[iex].copy()
    #outlines_gt = utils.outlines_list(mask_gt, multiprocessing=False)
    for j in range(len(kk)):
        grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=grid1[x[j], y[j]:y[j] + 2], wspace=0.08, hspace=0.15)
        k = kk[j]
        imgk = transforms.normalize99(imgs[k].squeeze().copy()) * 1.1
        titlek = titlesd[k]
        ax = plt.subplot(grid2[0, 0])
        pos = ax.get_position().bounds
        if k == 4:
            ax.set_position([pos[0], pos[1] - 0.04, pos[2], pos[3]])
        elif k == 7:
            ax.set_position([pos[0] + 0.05, pos[1] - 0.02, pos[2], pos[3]])
        elif k == 6:
            ax.set_position([pos[0], pos[1] - 0.02, pos[2], pos[3]])
        il = plot_label(ltr, il, ax, transl, fs_title)
        ax.imshow(imgk, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        #if k!=7:
        ax.text(0.5, 1.05, titlek, color=cols[k], ha="center", transform=ax.transAxes,
                fontsize="large")

        maskk = masks[k].copy()
        outlines = utils.outlines_list(maskk, multiprocessing=False)
        ax = plt.subplot(grid2[0, 1])
        pos = ax.get_position().bounds
        if k == 4:
            ax.set_position([pos[0], pos[1] - 0.04, pos[2], pos[3]])
        elif k == 7:
            ax.set_position([pos[0] + 0.05, pos[1] - 0.02, pos[2], pos[3]])
        elif k == 6:
            ax.set_position([pos[0], pos[1] - 0.02, pos[2], pos[3]])
        ax.imshow(imgk, cmap="gray", vmin=0, vmax=1)
        #for o in outlines_gt:
        #    ax.plot(o[:,0], o[:,1], color=[0.7,0.4,1], lw=1, ls="-")
        for o in outlines:
            ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5, ls="--")
        ax.axis("off")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.text(0.5, 1.05, "segmentation", ha="center", transform=ax.transAxes,
                fontsize="large")
        ax.text(1, -0.05, f"AP@0.5 = {aps[k,iex,0] : 0.2f}", va="top", ha="right",
                transform=ax.transAxes)
        if k == 2:
            il += 3

    transl = mtransforms.ScaledTranslation(-50 / 72, 8 / 72, fig.dpi_scale_trans)
    for j in range(2):
        il = 3 if j == 0 else 5
        ks = [2, 3] if j == 0 else [4, 6]
        ax = plt.subplot(grid[j, -1])
        pos = ax.get_position().bounds
        ax.set_position([
            pos[0] + 0.02, pos[1] - 0.06 - j * 0.01, pos[2] * 0.82,
            pos[2] * 0.82 * yratio
        ])
        il = plot_label(ltr, il, ax, transl, fs_title)
        for k in ks:
            ax.scatter(aps[1, :, 0], aps[k, :, 0], color=cols[k], s=3)
            ax.text(1, 0.15 - (k == ks[-1]) * 0.1, titles[k], color=cols[k], ha="right",
                    transform=ax.transAxes)
        ax.plot([0, 1.01], [0, 1.01], color="k")
        ax.set_ylim([0, 1.01])
        ax.set_xlim([0, 1.01])
        ticks = [0, 0.5, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_ylabel("AP@0.5, denoised image")
        ax.set_xlabel("AP@0.5, noisy image")
        il += 1
    pos = ax.get_position().bounds

    ax = plt.subplot(grid[-2:, -1])
    il += 2
    transl = mtransforms.ScaledTranslation(-50 / 72, 3 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    pos1 = ax.get_position().bounds
    ax.set_position([pos[0], pos1[1] + 0.02, pos[2], pos1[3] * 0.8])
    #theight = [0, 1, 2, 3, 5, 6, 7, 8, 4]
    theight = [0, 1, 2, 3, 5, 6, 6, 7, 4]
    for k in [1, 2, 3, 4, 6, 7, 8]:
        ax.plot(thresholds, aps[k, :, :].mean(axis=0), color=cols[k], ls=lss[k],
                lw=2.5 if k == 6 else 1.5)
        ax.text(1.13, 0.53 + 0.06 * theight[k], legstr[k], color=cols[k],
                transform=ax.transAxes, ha="right")
    ax.set_ylim([0, 0.7])
    ax.set_xlim([0.5, 1.0])
    ax.set_ylabel("average precision (AP)")
    ax.set_xlabel("IoU threshold")
    ax.set_xticks(np.arange(0.5, 1.05, 0.1))
    ax.set_yticks(np.arange(0.0, 0.71, 0.1))

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 8,
                                                        subplot_spec=grid[1:3, :-1],
                                                        wspace=0.05, hspace=0.5)

    gray = 0.92
    ax = plt.subplot(grid1[0, :])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.03, pos[1] - 0.02, pos[2] * 0.95, pos[3] * 1.38])
    pos = ax.get_position().bounds
    ax.spines["left"].set_visible(0)
    ax.spines["bottom"].set_visible(0)
    ax.patch.set_facecolor(gray * np.ones(3))
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(grid1[1, -5:])
    pos1 = ax.get_position().bounds
    ax.set_position([
        pos1[0] + 0.03, pos1[1] - 0.04, pos[0] + pos[2] - pos1[0] - 0.03, pos[3] * 1.38
    ])
    ax.spines["left"].set_visible(0)
    ax.spines["bottom"].set_visible(0)
    ax.patch.set_facecolor(gray * np.ones(3))
    ax.set_xticks([])
    ax.set_yticks([])

    inet = [20, 52, 17]
    ni = len(inet)
    dy, dx = 30, 30
    ly, lx = 130, 130
    stitles = [["noisy images", "denoised images", "predicted flows"],
               ["", "clean images", "ground-truth flows"]]
    nettitles = [
        "denoising network\n(weights learned)", "segmentation network\n(weights fixed)"
    ]
    lstr = ["reconstruction", "segmentation", "perceptual"]
    fs = "medium"
    for r in range(2):
        for j in range(r, 3):
            if j == 0:
                imset = imgs_all[1].copy()
            elif j == 1:
                imset = imgs_all[7].copy() if r == 0 else imgs_all[0].copy()
            else:
                imset = flows_perseg.copy() if r == 0 else flows_true.copy()
            ax = plt.subplot(grid1[r, 3 * j:3 * j + 2])
            if r == 0 and j == 0:
                transl = mtransforms.ScaledTranslation(-58 / 72, 34 / 72,
                                                       fig.dpi_scale_trans)
                il -= 6
                il = plot_label(ltr, il, ax, transl, fs_title)
                ax.text(-0.4, 1.3, "Training protocol", fontsize="large",
                        fontstyle="italic", transform=ax.transAxes)
            pos = ax.get_position().bounds
            ax.set_position([pos[0] + 0.01, pos[1], pos[2], pos[3]])
            pos = ax.get_position().bounds
            img0 = gray * np.ones(
                (ly + (ni - 1) * dy, lx +
                 (ni - 1) * dx)) if j < 2 else int(255 * gray) * np.ones(
                     (ly + (ni - 1) * dy, lx + (ni - 1) * dx, 3), "uint8")
            for k, i in enumerate(inet):
                im0 = imset[i].squeeze()[50:50 + ly, 50:50 + lx]
                im0 *= 1.1 if j < 2 else 1
                img0[k * dy:k * dy + ly, (ni - k - 1) * dx:(ni - k - 1) * dx + lx] = im0
            ax.imshow(img0, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            ax.text(0.5, 1.08 if r == 0 else -0.15, stitles[r][j],
                    transform=ax.transAxes, ha="center", fontsize="medium")
            if r == 0 and j > 0:
                # loss arrows
                if j == 1:
                    posloss = pos
                ax = fig.add_axes([pos[0], pos[1] - 0.5 * pos[3], pos[2], pos[3] * 0.5])
                for l in range(2):
                    ax.annotate(
                        f"{lstr[j-1]} loss", (0.5, 1 - l), (0.5, 0.5),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        bbox=dict(boxstyle="round", fc="w",
                                  edgecolor=cols[4 + 2 * (j - 1)]),
                        color=cols[4 + 2 * (j - 1)],
                        arrowprops=dict(width=2, color="k", headlength=5, headwidth=7),
                        ha="center", va="center", annotation_clip=True, fontsize=fs)
                ax.axis("off")

                # network
                ax = plt.subplot(grid1[r, 3 * (j - 1) + 2:3 * (j - 1) + 3])
                pos = ax.get_position().bounds
                ax.set_position([
                    pos[0] - 0.05 * pos[2] + 0.01, pos[1] + 0.1 * pos[3], pos[2] * 1.1,
                    pos[3] * 0.8
                ])
                ax.text(0.5, 1.15, nettitles[j - 1], ha="center",
                        transform=ax.transAxes, fontsize=fs)
                draw_net(fig, ax, arrow_in=True, arrow_out=True)

    ax = plt.subplot(grid1[1, -3])
    pos = ax.get_position().bounds
    ax.set_position([
        pos[0] - 0.05 * pos[2] + 0.01, pos[1] + 0.1 * pos[3], pos[2] * 1.1, pos[3] * 0.8
    ])
    draw_net(fig, ax, arrow_in=True, arrow_out=False)
    ax = fig.add_axes([
        posloss[0] + 0.14, posloss[1] - 0.5 * posloss[3], posloss[2], posloss[3] * 0.5
    ])
    for l in range(2):
        ax.annotate(f"{lstr[-1]} loss", (0.5, 1 - l), (0.5, 0.5), xycoords=ax.transAxes,
                    textcoords=ax.transAxes, bbox=dict(boxstyle="round", fc="w",
                                                       edgecolor=cols[5]),
                    color=cols[5], arrowprops=dict(width=2, color="k", headlength=5,
                                                   headwidth=7), ha="center",
                    va="center", annotation_clip=True, fontsize=fs)
    ax.axis("off")

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/fig1.pdf", dpi=150)


def denoising_ex(folder, grid, il, transl, kk=[0, 1, 5], seg=[0, 0],
                 noise_type="poisson", ctype="cyto2", j0=0, diams=None, dy=0):
    diam_mean = 30. if ctype == "cyto2" else 17.
    out = load_benchmarks(folder, ctype=ctype, noise_type=noise_type,
                          thresholds=np.ones(1) * 0.5, others=False)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
     flows_perseg, flows_true) = out
    titlesd = np.array(titles)[[0, 1, 4, 5, 6, 7]]
    titlesd[0] = "clean"
    if noise_type == "poisson":
        titlesd[1] = "noisy"
        titlesd[5] = "denoised"
    elif noise_type == "blur":
        titlesd[1] = "blurry"
        titlesd[5] = "deblurred"
    else:
        titlesd[1] = "downsampled"
        titlesd[5] = "upsampled"

    colsd = np.array(cols)[[0, 1, 4, 5, 6, 7]]

    wseg = True if len(seg) % 2 == 0 and seg[0] == 0 and seg[1] == 1 else False

    if ctype == "cyto2":
        ylim = [0, 165]
        xlim = [50, 275]
        yr = 165
        xr = 225
        iexs = [11, 15, 18, 19, 31, 21, 23, 30]
    else:
        ylim = [0, 100]
        xlim = [0, 100]  #150]
        yr = 100
        xr = 100
        iexs = [25, 23, 33, 37, 107, 52, 45, 110]  #43

    if noise_type == "poisson":
        title = "Denoising examples (test set)" if seg[
            0] == 0 else f"{ctype} model segmentation"
    elif noise_type == "blur":
        title = "Deblurring examples (test set)" if seg[
            0] == 0 else f"{ctype} model segmentation"
    elif noise_type == "downsample":
        title = "Upsampling examples (test set)" if seg[
            0] == 0 else f"{ctype} model segmentation"
    if ctype == "nuclei" and seg[0] == 0:
        title += " - nuclei"

    for i, iex in enumerate(iexs):
        ratio = diams[iex] / diam_mean if diams is not None else diam_test[
            iex] / diam_mean
        yri = int(yr * ratio)
        xri = int(xr * ratio)
        ylimi = np.array([0, yri])
        xlimi = np.array([0, xri])
        #print(ylimi, xlimi)

        if ctype == "cyto2":
            xlimi += int(150 * ratio) if iex == 15 or iex == 21 or iex == 23 else 0
            ylimi += 100 if iex == 11 else 0
            ylimi += 50 if iex == 21 else 0
            xlimi += 270 if iex == 30 else 0
            ylimi += 160 if iex == 30 else 0
        else:
            xlimi += 130 + 50 if iex == 107 else 0
            ylimi += 110 if iex == 107 else 0
            xlimi += 150 if iex == 33 else 0

        nl = noise_levels[iex]
        if diams is not None:
            ratio = diam_mean / diams[iex]
            y0, x0 = int(ylimi[0] * ratio), int(xlimi[0] * ratio)
            y1 = y0 + int((ylimi[1] - ylimi[0]) * ratio)
            x1 = x0 + int((xlimi[1] - xlimi[0]) * ratio)
            ylimi, xlimi = [y0, y1], [x0, x1]
            #nl = noise_levels[iex]
            #print(nl)
            #ylimi = [y0//nl, y1//nl]
            #xlimi = [x0//nl, x1//nl]

        mask_gt = masks_true[iex].copy()
        #outlines_gt = utils.outlines_list(mask_gt, multiprocessing=False)
        for j in range(len(kk)):
            k = kk[j]
            imgk = transforms.normalize99(imgs_all[k][iex].squeeze().copy()) * 1.1
            Ly, Lx = imgk.shape
            if noise_type == "downsample" and k == 1:
                nl = noise_levels[iex]
                imgk = cv2.resize(imgk, (Lx * nl, Ly * nl),
                                  interpolation=cv2.INTER_NEAREST)
                #imgk = np.tile(imgk[:,np.newaxis], (1,nl,1)).reshape(-1, Lx)
                #imgk = np.tile(imgk[:,:,np.newaxis], (1,1,nl)).reshape(imgk.shape[0], -1)
            titlek = titlesd[k]
            ax = plt.subplot(grid[j0 + j, i])
            if dy > 0:
                pos = ax.get_position().bounds
                ax.set_position([pos[0], pos[1] + dy * j, pos[2], pos[3]])
            if wseg and j % 2 == 1:
                pos = ax.get_position().bounds
                ax.set_position([pos[0], pos[1] + 0.005, pos[2], pos[3]])

            ax.imshow(imgk, cmap="gray", vmin=0, vmax=1)
            if seg[j]:
                maskk = masks_all[k][iex].copy()
                outlines = utils.outlines_list(maskk, multiprocessing=False)
                #for o in outlines_gt:
                #    ax.plot(o[:,0], o[:,1], color=[0.7,0.4,1], lw=1, ls="-")
                for o in outlines:
                    ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5, ls="--")

            if j == 0 and i == 0:
                il = plot_label(ltr, il, ax, transl, fs_title)
                ax.text(-0., 1.09, title, fontsize="large", transform=ax.transAxes,
                        fontstyle="italic")
            ax.set_ylim(ylimi)
            ax.set_xlim(xlimi)
            ax.axis("off")
            if i == 0:
                ax.text(-0.15, 0.5,
                        titlek if not wseg or j % 2 == 0 else "segmentation",
                        transform=ax.transAxes, va="center", rotation=90,
                        color=colsd[k], fontsize="medium")
            #if j==2:
            if not wseg or j % 2 == 1:
                ax.text(1, -0.015, f"AP@0.5 = {aps[k,iex,0] : 0.2f}", va="top",
                        ha="right", transform=ax.transAxes, fontsize="small")
    return il


def suppfig_seg(folder, save_fig=False):
    thresholds = np.arange(0.5, 1.05, 0.05)
    out = load_benchmarks(folder, ctype="cyto2", thresholds=thresholds)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
     flows_perseg, flows_true) = out

    fig = plt.figure(figsize=(14, 8), dpi=100)
    yratio = 14 / 8
    grid = plt.GridSpec(5, 8, figure=fig, left=0.02, right=0.98, top=0.96, bottom=0.1,
                        wspace=0.05, hspace=0.1)
    transl = mtransforms.ScaledTranslation(-15 / 72, 10 / 72, fig.dpi_scale_trans)
    il = 0
    il = denoising_ex(folder, grid, il, transl, ctype="cyto2", noise_type="poisson",
                      kk=[0, 1, 5], seg=[1, 1, 1])

    transl = mtransforms.ScaledTranslation(-45 / 72, -2.5 / 72, fig.dpi_scale_trans)
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[-2:, :],
                                                        wspace=0.25, hspace=0.1)
    nn = fps.shape[1]
    xs = [1, 2, 3, 4, 6, 7, 8]  #np.arange(1, len(fps))
    nt = tps[:, :, 0] + fns[:, :, 0]
    nps = [fps[:, :, 0] / nt, tps[:, :, 0] / nt, fns[:, :, 0] / nt]
    ylabels = ["false positive rate", "true positive rate", "false negative rate"]
    for j in range(3):
        ax = plt.subplot(grid1[0, j])
        pos = ax.get_position().bounds
        ax.set_position(
            [pos[0] + 0.045, pos[1] - 0.03 + pos[3] * 0.1, pos[2] * 0.75,
             pos[3] * 0.9])  #+pos[3]*0.15-0.03, pos[2], pos[3]*0.7])
        il = plot_label(ltr, il, ax, transl, fs_title)
        ys = nps[j]
        kk = 0
        for k in xs:
            ax.scatter(kk * np.ones(nn) + np.random.randn(nn) * 0.05, ys[k, :],
                       alpha=0.2, s=5, color=cols[k], rasterized=True)
            ax.scatter(kk, ys[k, :].mean(), marker="X", lw=1, s=80, facecolor='w',
                       edgecolor=cols[k])
            kk += 1
        ax.set_xticks(np.arange(0, len(xs)))
        ax.set_xticklabels(
            np.array(titles)[[1, 2, 3, 4, 6, 7, 8]], rotation=25, ha="right")
        ax.set_ylim([0, 1])
        ax.set_ylabel(ylabels[j])

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/suppfig_cells.pdf", dpi=100)


def suppfig_nuclei(folder, save_fig=False):
    thresholds = np.arange(0.5, 1.05, 0.05)
    out = load_benchmarks(folder, ctype="nuclei", thresholds=thresholds)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
     flows_perseg, flows_true) = out

    fig = plt.figure(figsize=(14, 8), dpi=100)
    yratio = 14 / 8
    grid = plt.GridSpec(5, 8, figure=fig, left=0.02, right=0.98, top=0.96, bottom=0.1,
                        wspace=0.01, hspace=0.15)
    transl = mtransforms.ScaledTranslation(-15 / 72, 10 / 72, fig.dpi_scale_trans)
    il = 0
    il = denoising_ex(folder, grid, il, transl, ctype="nuclei", noise_type="poisson",
                      kk=[0, 1, 5], seg=[1, 1, 1])

    transl = mtransforms.ScaledTranslation(-45 / 72, -2.5 / 72, fig.dpi_scale_trans)
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[-2:, :],
                                                        wspace=0.25, hspace=0.1)
    nn = fps.shape[1]
    xs = [1, 2, 3, 4, 6, 7, 8]  #np.arange(1, len(fps))
    nt = tps[:, :, 0] + fns[:, :, 0]
    nps = [fps[:, :, 0] / nt, tps[:, :, 0] / nt, fns[:, :, 0] / nt]
    ylabels = ["false positive rate", "true positive rate", "false negative rate"]
    for j in range(3):
        ax = plt.subplot(grid1[0, j])
        pos = ax.get_position().bounds
        ax.set_position(
            [pos[0] + 0.045, pos[1] - 0.03 + pos[3] * 0.1, pos[2] * 0.75,
             pos[3] * 0.9])  #+pos[3]*0.15-0.03, pos[2], pos[3]*0.7])
        il = plot_label(ltr, il, ax, transl, fs_title)
        ys = nps[j]
        kk = 0
        for k in xs:
            ax.scatter(kk * np.ones(nn) + np.random.randn(nn) * 0.05, ys[k, :],
                       alpha=0.2, s=5, color=cols[k], rasterized=True)
            ax.scatter(kk, ys[k, :].mean(), marker="X", lw=1, s=80, facecolor='w',
                       edgecolor=cols[k])
            kk += 1
        ax.set_xticks(np.arange(0, len(xs)))
        ax.set_xticklabels(
            np.array(titles)[[1, 2, 3, 4, 6, 7, 8]], rotation=25, ha="right")
        ax.set_ylim([0, 1])
        ax.set_ylabel(ylabels[j])

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/suppfig_nuclei.pdf", dpi=100)


def fig2(folder, folder2="/media/carsen/ssd4/denoising/Projection_Flywing/test_data",
         folder3="/media/carsen/ssd4/denoising/ribo_denoise/",
         save_fig=False):
    thresholds = np.arange(0.5, 1.05, 0.05)

    fig = plt.figure(figsize=(14, 12), dpi=100)
    yratio = 14 / 8
    grid = plt.GridSpec(7, 8, figure=fig, left=0.02, right=0.97, top=0.98, bottom=0.08,
                        wspace=0.12, hspace=0.25)
    transl = mtransforms.ScaledTranslation(-18 / 72, 10 / 72, fig.dpi_scale_trans)
    il = 0

    if 1:
        out = load_benchmarks(folder, ctype="cyto2", thresholds=thresholds)
        (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
         flows_perseg, flows_true) = out
        il = denoising_ex(folder, grid, il, transl, ctype="cyto2", noise_type="poisson",
                          kk=[0, 1, 5], seg=[0, 0, 0], dy=0.015)

    dat = np.load(f"{folder2}/cp_masks.npy", allow_pickle=True).item()
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=grid[-4:-2, :],
                                                        wspace=0.05, hspace=0.1)

    iex = 10
    nl = 0
    ylim = [250, 400]
    xlim = [250, 550]

    transl = mtransforms.ScaledTranslation(-18 / 72, 26 / 72, fig.dpi_scale_trans)
    outlines_gt = utils.outlines_list(dat["masks_clean"][iex].copy(),
                                      multiprocessing=False)
    titlesd = ["high laser power (20%)", "low laser power (2%)", "denoised (Cellpose3)"]
    for k in range(3):
        if k == 0:
            img = dat["clean"][iex].copy()
        elif k == 1:
            img = dat["noisy"][nl][iex].copy()
            maskk = dat["masks_noisy"][nl][iex].copy()
            ap = dat["ap_noisy"][nl][iex, 0]
        else:
            img = dat["denoised"][nl][iex].copy()
            maskk = dat["masks_denoised"][nl][iex].copy()
            ap = dat["ap_denoised"][nl][iex, 0]
        img = transforms.normalize99(img)

        ax = plt.subplot(grid1[0, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - 0.01, pos[2], pos[3]])
        ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        ax.set_title(titlesd[k], color="k" if k < 2 else cols[-2], fontsize="medium")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        if k == 0:
            ax.text(0, 1.25, "Denoising drosophila wing epithelia", fontsize="large",
                    fontstyle="italic", transform=ax.transAxes)
            il = plot_label(ltr, il, ax, transl, fs_title)

        ax = plt.subplot(grid1[1, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - 0.01, pos[2], pos[3]])
        ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        #ax.set_title("segmentation")
        if k == 0:
            for o in outlines_gt:
                ax.plot(o[:, 0], o[:, 1], color=[1, 0, 1], lw=1, ls="--")
            ax.text(1, -0.15, "ground-truth", ha="right", transform=ax.transAxes)
        else:
            outlines = utils.outlines_list(maskk, multiprocessing=False)
            for o in outlines:
                ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1, ls="--")
            ax.text(1, -0.15, f"AP@0.5 = {ap:.2f}", ha="right", transform=ax.transAxes)

    aps = []
    aps.append(dat["ap_noisy"])
    dat2 = np.load(f"{folder2}/n2v_masks.npy", allow_pickle=True).item()
    aps.append(dat2["ap_n2v"])
    dat2 = np.load(f"{folder2}/n2s_masks.npy", allow_pickle=True).item()
    aps.append(dat2["ap_n2s"])
    aps.append(dat["ap_denoised"])

    transl = mtransforms.ScaledTranslation(-40 / 72, 15 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid1[:, 3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.05, pos[1] - 0.0, pos[2] * 0.7, pos[3]])
    nl = 0
    titlesd = titles.copy()
    titlesd[7] = "Cellpose3"
    titlesd[1] = "noisy \n(2% laser \npower)"
    kk = [1, 2, 3, 7]
    theight = [-2, 2, 1, 3]
    for k in range(len(aps)):
        means = aps[k][nl].mean(axis=0)
        ax.plot(thresholds, means, color=cols[kk[k]])
        ax.text(1.1, 0.7 + theight[k] * 0.08, titlesd[kk[k]], color=cols[kk[k]],
                transform=ax.transAxes, ha="right")
    il = plot_label(ltr, il, ax, transl, fs_title)
    #ax.set_title("2% laser power", fontsize="medium")
    ax.text(-0.18, 1.07, "Segmentation performance", fontstyle="italic",
            transform=ax.transAxes, fontsize="large")
    ax.set_ylabel("average precision (AP)")
    ax.set_xlabel("IoU threshold")
    ax.set_xticks(np.arange(0.5, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim([0, 1.])
    ax.set_xlim([0.5, 1.0])

    ax = plt.subplot(grid1[:, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.05, pos[1] - 0.0, pos[2] * 0.7, pos[3]])
    kk = [1, 2, 3, 7]
    for k in range(len(aps)):
        means = np.array([aps[k][nl][:, 0].mean(axis=0) for nl in [0, 2, 1]])
        sems = np.array([aps[k][nl][:, 0].std(axis=0) / (25**0.5) for nl in [0, 2, 1]])
        ax.errorbar(np.arange(0, 3), means, sems, color=cols[kk[k]])

    ax.set_ylabel("AP @ 0.5 IoU threshold")
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim([0, 1.])
    ax.set_xlim([-0.1, 2.1])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2%", "3%", "5%"])
    ax.set_xlabel("laser power")


    dat = np.load(f"{folder3}/ribo_denoise_n2v.npy", allow_pickle=True).item()
    ap_n2v = dat["ap_n2v"]
    dat = np.load(f"{folder3}/ribo_denoise_n2s.npy", allow_pickle=True).item()
    ap_n2s = dat["ap_n2s"]
    dat = np.load(f"{folder3}/ribo_denoise.npy", allow_pickle=True).item()
    navgs  = dat["navgs"]

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=grid[-2:, :],
                                                        wspace=0.05, hspace=0.1)
    iex = 3
    nl = 2
    ylim = [350, 500]
    xlim = [200, 500]

    transl = mtransforms.ScaledTranslation(-18 / 72, 26 / 72, fig.dpi_scale_trans)
    outlines_gt = utils.outlines_list(dat["masks_clean"][iex].copy(),
                                        multiprocessing=False)
    titlest = ["clean (300 frames averaged)", "noisy (4 frames averaged)", "denoised (Cellpose3)"]
    for k in range(3):
        if k == 0:
            img = dat["clean"][iex].copy()
        elif k == 1:
            img = dat["noisy"][nl][iex].copy()
            maskk = dat["masks_noisy"][nl][iex].copy()
            ap = dat["ap_noisy"][nl][iex, 0]
        else:
            img = dat["imgs_dn"][nl][iex].copy()
            maskk = dat["masks_dn"][nl][iex].copy()
            ap = dat["ap_dn"][nl][iex, 0]
        img = transforms.normalize99(img)

        ax = plt.subplot(grid1[0, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - 0.05, pos[2], pos[3]])
        ax.imshow(img, vmin=0., vmax=0.75, cmap="gray")
        ax.set_title(titlest[k], color="k" if k < 2 else [0,0.5,0], fontsize="medium")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        if k == 0:
            ax.text(0, 1.25, "Denoising two-photon imaging in mice", fontsize="large",
                    fontstyle="italic", transform=ax.transAxes)
            il = plot_label(ltr, il, ax, transl, fs_title)

        ax = plt.subplot(grid1[1, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - 0.05, pos[2], pos[3]])
        ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        #ax.set_title("segmentation")
        if k == 0:
            for o in outlines_gt:
                ax.plot(o[:, 0], o[:, 1], color=[1, 0, 1], lw=1, ls="--")
            ax.text(1, -0.15, "ground-truth", ha="right", transform=ax.transAxes)
        else:
            outlines = utils.outlines_list(maskk, multiprocessing=False)
            for o in outlines:
                ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1, ls="--")
            ax.text(1, -0.15, f"AP@0.5 = {ap:.2f}", ha="right", transform=ax.transAxes)

    grid11 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid1[:, -2:],
                                                        wspace=0.3, hspace=0.1)
    

    transl = mtransforms.ScaledTranslation(-35 / 72, 25 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid11[:, 0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.05, pos[1] - 0.04, pos[2] * 0.8, pos[3]*0.9])
    aps = [dat["ap_noisy"], ap_n2v, ap_n2s, dat["ap_dn"]]
    theight = [-0.9, 3, 2, 4]
    kk = [1, 2, 3, 7]
    titlesd[1] = "noisy\n(4 frames\naveraged)"
    for k in range(len(aps)):
        means = aps[k][nl, :12].mean(axis=0)
        ax.plot(thresholds, means, color=cols[kk[k]])
        ax.text(1.15, 0.62   + theight[k] * 0.09, titlesd[kk[k]], 
                color=cols[kk[k]],
                transform=ax.transAxes, ha="right")
    ax.set_ylim([0, 0.8])
    ax.text(-0.18, 1.13, "Segmentation performance", fontstyle="italic",
            transform=ax.transAxes, fontsize="large")
    ax.set_ylabel("average precision (AP)")
    ax.set_xlabel("IoU threshold")
    ax.set_xticks(np.arange(0.5, 1.05, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim([0, 0.83])
    ax.set_xlim([0.5, 1.0])
    il = plot_label(ltr, il, ax, transl, fs_title)

    ifrs = [slice(0, 12), slice(12, 20)]
    for i, ifr in enumerate(ifrs):
        ax = plt.subplot(grid11[:, i + 1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] + 0.04 - i*0.01, pos[1] - 0.04, pos[2] * 0.8, pos[3]*0.9])
        nifr = ifr.stop - ifr.start
        for k in range(len(aps)):
            means = np.array([aps[k][nl][ifr, 0].mean(axis=0) for nl in range(len(aps[k]))])
            sems = np.array([aps[k][nl][ifr, 0].std(axis=0) / (nifr**0.5) for nl in range(len(aps[k]))])
            ax.errorbar(np.arange(0, len(means)), means, sems, color=cols[kk[k]])

        ax.set_xticks(np.arange(0, len(navgs), 2))
        ax.set_xticklabels([f"{navg}" for navg in navgs[::2]])
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim([0, 0.83])
        ax.set_xlabel("# of frames averaged")
        ax.set_title("dense expression" if i==0 else "sparse expression", fontsize="medium")


    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/fig2.pdf", dpi=100)


def fig3(folder, folder2="/media/carsen/ssd4/denoising/Denoising_Tribolium/test_data",
         save_fig=False):
    thresholds = np.arange(0.5, 1.05, 0.05)

    fig = plt.figure(figsize=(14, 8), dpi=100)
    yratio = 14 / 8
    grid = plt.GridSpec(5, 10, figure=fig, left=0.02, right=0.98, top=0.95, bottom=0.1,
                        wspace=0.05, hspace=0.25)
    il = 0

    out = load_benchmarks(folder, ctype="nuclei", thresholds=thresholds)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
     flows_perseg, flows_true) = out

    if 1:
        transl = mtransforms.ScaledTranslation(-18 / 72, 10 / 72, fig.dpi_scale_trans)
        il = denoising_ex(folder, grid, il, transl, ctype="nuclei",
                          noise_type="poisson", kk=[0, 1, 5], seg=[0, 0,
                                                                   0], dy=0.008, j0=0)

    transl = mtransforms.ScaledTranslation(-45 / 72, 10 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid[:3, -2:])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.045, pos[1] + 0.05, pos[2] * 0.75,
                     pos[3] * 0.9])  #+pos[3]*0.15-0.03, pos[2], pos[3]*0.7])
    il = plot_label(ltr, il, ax, transl, fs_title)
    #theight = [0, 1, 2, 3, 5, 6, 7, 8, 4]
    theight = [0, 1, 2, 3, 5, 6, 6, 7, 4]
    for k in [1, 2, 3, 4, 6, 7, 8]:
        ax.plot(thresholds, aps[k, :, :].mean(axis=0), color=cols[k], ls=lss[k])
        ax.text(0.55 - 0.035 * theight[k], 0.55 + 0.06 * theight[k], legstr[k],
                color=cols[k], transform=ax.transAxes)
    ax.set_ylim([0, 0.72])
    ax.set_ylabel("average precision (AP)")
    ax.set_xlabel("IoU threshold")
    ax.text(-0.18, 1.04, "Segmentation performance", fontstyle="italic",
            transform=ax.transAxes, fontsize="large")
    ax.set_xticks(np.arange(0.5, 1.05, 0.1))
    ax.set_xlim([0.5, 1.0])

    dat = np.load(f"{folder2}/cp_masks.npy", allow_pickle=True).item()
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=grid[-2:, :],
                                                        wspace=0.05, hspace=0.1)

    iex = 0
    nl = 1
    ylim = [250, 500]
    xlim = [480, 920]
    ylim = [650, 850]
    xlim = [170, 570]
    transl = mtransforms.ScaledTranslation(-18 / 72, 26 / 72, fig.dpi_scale_trans)
    outlines_gt = utils.outlines_list(dat["masks_clean"][iex].copy().T,
                                      multiprocessing=False)
    titlesd = [
        "high laser power (20mW)", "low laser power (0.3mW)", "denoised (Cellpose3)"
    ]
    for k in range(3):
        if k == 0:
            img = dat["clean"][iex].copy().T
        elif k == 1:
            img = dat["noisy"][nl][iex].copy().T
            maskk = dat["masks_noisy"][nl][iex].copy().T
            ap = dat["ap_noisy"][nl][iex, 0]
        else:
            img = dat["denoised"][nl][iex].copy().T
            maskk = dat["masks_denoised"][nl][iex].copy().T
            ap = dat["ap_denoised"][nl][iex, 0]
        img = transforms.normalize99(img)

        ax = plt.subplot(grid1[0, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - 0.05, pos[2], pos[3]])
        ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        ax.set_title(titlesd[k], color="k" if k < 2 else cols[-2], fontsize="medium")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        if k == 0:
            ax.text(0, 1.25, "Denoising tribolium nuclei", fontsize="large",
                    fontstyle="italic", transform=ax.transAxes)
            il = plot_label(ltr, il, ax, transl, fs_title)

        ax = plt.subplot(grid1[1, k])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - 0.05, pos[2], pos[3]])
        ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        #ax.set_title("segmentation")
        if k == 0:
            for o in outlines_gt:
                ax.plot(o[:, 0], o[:, 1], color=[1, 0, 1], lw=1, ls="--")
            ax.text(1, -0.15, "ground-truth", ha="right", transform=ax.transAxes)
        else:
            outlines = utils.outlines_list(maskk, multiprocessing=False)
            for o in outlines:
                ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1, ls="--")
            ax.text(1, -0.15, f"AP@0.5 = {ap:.2f}", ha="right", transform=ax.transAxes)

    aps = []
    aps.append(dat["ap_noisy"])
    dat2 = np.load(f"{folder2}/n2v_masks.npy", allow_pickle=True).item()
    aps.append(dat2["ap_n2v"])
    dat2 = np.load(f"{folder2}/n2s_masks.npy", allow_pickle=True).item()
    aps.append(dat2["ap_n2s"])
    aps.append(dat["ap_denoised"])

    transl = mtransforms.ScaledTranslation(-40 / 72, 15 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid1[:, 3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.05, pos[1] - 0.03, pos[2] * 0.7, pos[3]])
    titlesd = titles.copy()
    titlesd[1] = "noisy \n(0.3mW laser \npower)"
    titlesd[7] = "Cellpose3"
    kk = [1, 2, 3, 7]
    theight = [-2, 2, 1, 3]
    for k in range(len(aps)):
        means = aps[k][nl].mean(axis=0)
        ax.plot(thresholds, means, color=cols[kk[k]])
        ax.text(1.1, 0.7 + theight[k] * 0.08, titlesd[kk[k]], color=cols[kk[k]],
                transform=ax.transAxes, ha="right")
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.text(-0.18, 1.07, "Segmentation performance", fontstyle="italic",
            transform=ax.transAxes, fontsize="large")
    ax.set_ylabel("average precision (AP)")
    ax.set_xlabel("IoU threshold")
    ax.set_xticks(np.arange(0.5, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 0.7, 0.2))
    ax.set_ylim([0, 0.64])
    ax.set_xlim([0.5, 1.0])

    ax = plt.subplot(grid1[:, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.05, pos[1] - 0.03, pos[2] * 0.7, pos[3]])
    kk = [1, 2, 3, 7]
    for k in range(len(aps)):
        means = np.array([aps[k][nl][:, 0].mean(axis=0) for nl in [0, 1, 2]])
        sems = np.array([aps[k][nl][:, 0].std(axis=0) / (25**0.5) for nl in [0, 1, 2]])
        ax.errorbar(np.arange(0, 3), means, sems, color=cols[kk[k]])

    ax.set_ylabel("AP @ 0.5 IoU threshold")
    ax.set_yticks(np.arange(0, 0.7, 0.2))
    ax.set_ylim([0, 0.64])
    ax.set_xlim([-0.1, 2.1])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0.2", "0.3", "0.5"])
    ax.set_xlabel("laser power (mW)")

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/fig3.pdf", dpi=100)


def fig4(folder, ctype="cyto2", save_fig=False):
    titlesd = np.array(titles)[[0, 1, 4, 5, 6, 7]]
    colsd = np.array(cols)[[0, 1, 4, 5, 6, 7]]
    titlesd[0] = "clean"

    thresholds = np.arange(0.5, 1.05, 0.05)

    fig = plt.figure(figsize=(14, 6), dpi=100)
    yratio = 14 / 6
    grid = plt.GridSpec(4, 10, figure=fig, left=0.02, right=0.98, top=0.95, bottom=0.04,
                        wspace=0.05, hspace=0.1 if ctype == "cyto2" else 0.4)
    transl = mtransforms.ScaledTranslation(-15 / 72, 8 / 72, fig.dpi_scale_trans)
    transl1 = mtransforms.ScaledTranslation(-35 / 72, 3 / 72, fig.dpi_scale_trans)

    il = 0

    out = load_benchmarks(folder, ctype=ctype, thresholds=thresholds, noise_type="blur",
                          others=False)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
     flows_perseg, flows_true) = out
    diams = diam_test.copy()

    il = 0
    il = denoising_ex(folder, grid, il, transl, ctype=ctype, noise_type="blur",
                      kk=[1, 5], seg=[0, 0, 0], dy=0.03)
    il += 1
    il = denoising_ex(folder, grid, il, transl, ctype=ctype, noise_type="downsample",
                      kk=[1, 5], seg=[0, 0, 0], j0=2, diams=diams, dy=0.03)
    il -= 2

    for d in range(2):
        if d == 1:
            out = load_benchmarks(folder, ctype=ctype, thresholds=thresholds,
                                  noise_type="downsample", others=False)
            (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true,
             diam_test, flows_perseg, flows_true) = out

        ax = plt.subplot(grid[2 * d:2 * d + 2, -1:])
        pos = ax.get_position().bounds
        ax.set_position(
            [pos[0] - 0.05, pos[1] + pos[3] * 0.18, pos[2] * 1.2,
             pos[3] * 0.78])  #+pos[3]*0.15-0.03, pos[2], pos[3]*0.7])
        il = plot_label(ltr, il, ax, transl1, fs_title)
        theight = [0, 0, 1, 2, 2, 3]
        titlesd[1] = "blurry image" if d == 0 else "bilinearly\nupsampled "
        for k in [1, 2, 4, 5]:  #range(1, len(aps)):
            ax.plot(thresholds, aps[k, :, :].mean(axis=0), color=colsd[k])
            ax.text(
                1.01,
                0.74 + 0.08 * theight[k] + 0.06 * (ctype == "nuclei") * (d == 0),
                titlesd[k],
                color=colsd[k],  #- 0.08*(d==1)*(k==1)
                transform=ax.transAxes,
                ha="right",
                fontsize="small",
                va="top")
        ax.set_ylim([0, 0.7])
        ax.set_ylabel("average precision (AP)")
        ax.set_xlabel("IoU threshold")
        ax.set_xticks(np.arange(0.5, 1.05, 0.1))
        ax.set_xlim([0.5, 1.0])
        il += 1

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        if ctype == "cyto2":
            fig.savefig("figs/fig4.pdf", dpi=100)
        else:
            fig.savefig("figs/suppfig_blur.pdf", dpi=100)


def load_benchmarks_specialist(folder, thresholds=np.arange(0.5, 1.05, 0.05)):
    noise_type = "poisson"
    ctype = "cyto2"
    nimg_test = 11
    folder_name = ctype
    root = Path(f"{folder}/images_{folder_name}/")

    masks_all = []
    imgs_all = []
    dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                  allow_pickle=True).item()
    test_data = dat["test_data"][:nimg_test]
    test_noisy = dat["test_noisy"][:nimg_test]
    masks_noisy = dat["masks_noisy"][:nimg_test]
    masks_true = dat["masks_true"][:nimg_test]
    masks_data = dat["masks_orig"][:nimg_test]
    diam_test = dat["diam_test"][:nimg_test]
    noise_levels = dat["noise_levels"][:nimg_test]

    masks_all.append(masks_data)
    masks_all.append(masks_noisy)
    imgs_all.append(test_data)
    imgs_all.append(test_noisy)

    dat = np.load(root / "noisy_test" / f"test_{noise_type}_n2v_specialist.npy",
                  allow_pickle=True).item()
    test_n2v = dat["test_n2v"][:nimg_test]
    masks_n2v = dat["masks_n2v"][:nimg_test]
    imgs_all.append(test_n2v)
    masks_all.append(masks_n2v)

    dat = np.load(root / "noisy_test" / f"test_{noise_type}_n2s_specialist.npy",
                  allow_pickle=True).item()
    test_n2s = dat["test_n2s"][:nimg_test]
    masks_n2s = dat["masks_n2s"][:nimg_test]
    imgs_all.append(test_n2s)
    masks_all.append(masks_n2s)

    dat = np.load(root / "noisy_test" / f"test_{noise_type}_care_specialist.npy",
                  allow_pickle=True).item()
    test_care = dat["test_care"][:nimg_test]
    masks_care = dat["masks_care"][:nimg_test]
    imgs_all.append(test_care)
    masks_all.append(masks_care)

    dat = np.load(root / "noisy_test" / f"test_{noise_type}_denoiseg_specialist.npy",
                  allow_pickle=True).item()
    test_dns = dat["test_denoiseg"][:nimg_test]
    masks_dns = dat["masks_denoiseg"][:nimg_test]
    imgs_all.append(test_dns)
    masks_all.append(masks_dns)
    masks_dns = dat["masks_denoiseg_seg"][:nimg_test]
    imgs_all.append(test_dns)
    masks_all.append(masks_dns)

    dat = np.load(root / "noisy_test" / f"test_{noise_type}_cp3_all.npy",
                  allow_pickle=True).item()
    istr = ["rec", "per", "seg", "perseg"]
    for k in range(len(istr)):
        if f"test_{istr[k]}" in dat:
            test_dn = dat[f"test_{istr[k]}"][:nimg_test]
            masks_dn = dat[f"masks_{istr[k]}"][:nimg_test]
            imgs_all.append(test_dn)
            masks_all.append(masks_dn)

    # benchmarking
    aps = []
    tps = []
    fps = []
    fns = []
    for k in range(len(masks_all)):
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_all[k],
                                                   threshold=thresholds)
        aps.append(ap)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    aps, tps, fps, fns = np.array(aps), np.array(tps), np.array(fps), np.array(fns)

    return (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true,
            diam_test)


def suppfig_specialist(folder, save_fig=True):
    n_train = 100
    im_train = [
        io.imread(
            Path(folder) / "images_cyto2" / "noisy_test" / "care" / "source" /
            f"{i:03d}.tif") for i in range(n_train)
    ]
    im_gt = [
        io.imread(
            Path(folder) / "images_cyto2" / "noisy_test" / "care" / "GT" /
            f"{i:03d}.tif") for i in range(n_train)
    ]

    thresholds = np.arange(0.5, 1.05, 0.05)
    out = load_benchmarks_specialist(folder, thresholds=thresholds)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test) = out

    legstr0 = []
    for ls in legstr[:-1]:
        legstr0.append(" ".join(ls.split(" ")[1:]))
        legstr0[-1] = u"\u2013 " + legstr0[-1]
    legstr0.insert(4, u"\u2013 CARE")
    legstr0.insert(5, u"\u2013 denoiseg")
    legstr0.insert(6, "-- denoiseg\n(segmentation)")
    cols0 = list(cols[:-1].copy())
    cols0.insert(4, [1, 0.5, 1])
    cols0.insert(5, 0.4*np.ones(3))
    cols0.insert(6, 0.4*np.ones(3))
    print(len(cols0))
    legstr0[-1] = u"\u2013 Cellpose3\n(per. + seg.)"

    il = 0

    fig = plt.figure(figsize=(9, 5), dpi=100)
    yratio = 9 / 5
    grid = plt.GridSpec(2, 4, figure=fig, left=0.02, right=0.96, top=0.96, bottom=0.1,
                        wspace=0.15, hspace=0.2)

    titles = ["train - clean", "train - noisy", "test - noisy"]

    transl = mtransforms.ScaledTranslation(-12 / 72, 22 / 72, fig.dpi_scale_trans)
    for j in range(3):
        if j == 0:
            imset = im_gt.copy()
        elif j == 1:
            imset = im_train.copy()
        else:
            imset = imgs_all[1].copy()
        ax = plt.subplot(grid[0, j])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] - 0.02 * j, pos[1] - 0.04, pos[2], pos[3]])
        ly, lx = 128, 128
        dy, dx = 20, 30
        ni = 5
        img0 = np.ones((ly + (ni - 1) * dy, lx + (ni - 1) * dx))
        ii = np.arange(0, 5)[::-1] if j == 2 else np.arange(1, 20 * ni, 20)[::-1]
        if j < 2:
            x0, y0 = 20, 20
        else:
            x0, y0 = 350, 100
        for k, i in enumerate(ii):
            img0[k * dy:k * dy + ly, (ni - k - 1) * dx:(ni - k - 1) * dx +
                 lx] = imset[i].squeeze()[y0:y0 + ly, x0:x0 + lx]
        img0 *= 1.1 if j < 2 else 1
        img0 = np.clip(img0, 0, 1)

        ax.imshow(img0, cmap="gray", vmin=0, vmax=1)
        ax.text(0.5, 1.05, titles[j], transform=ax.transAxes, ha="center")
        ax.axis("off")

        if j == 0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.text(0.02, 1.2, "Specialist dataset", fontsize="large",
                    fontstyle="italic", transform=ax.transAxes)

    transl = mtransforms.ScaledTranslation(-45 / 72, 8 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid[0, -1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.01, pos[1] - 0.03, pos[2] * 0.8,
                     pos[3] * 1])  #+pos[3]*0.15-0.03, pos[2], pos[3]*0.7])
    il = plot_label(ltr, il, ax, transl, fs_title)
    theight = [0, 0, 4, 3, 6, 5, 1, 5, 7, 8, 7.1]
    for k in [1, 2, 3, 4, 5, 6, 10]:
        ax.plot(thresholds, aps[k, :, :].mean(axis=0), color=cols0[k],
                lw=3 if k==4 else 1, ls="--" if k==6 else "-")
        #ax.errorbar(thresholds, aps[k,:,:].mean(axis=0), aps[k,:,:].std(axis=0) / 10**0.5, color=cols0[k])
        ax.text(0.7, 0.3 + 0.09 * theight[k], legstr0[k], color=cols0[k],
                transform=ax.transAxes)
    ax.set_ylim([0, 0.8])
    ax.set_ylabel("average precision (AP)")
    ax.set_xlabel("IoU threshold")
    ax.set_xticks(np.arange(0.5, 1.05, 0.1))
    ax.set_xlim([0.5, 1.0])

    transl = mtransforms.ScaledTranslation(-10 / 72, 20 / 72, fig.dpi_scale_trans)

    kk = [2, 3, 4, 10]
    iex = 8
    ylim = [10, 310]
    xlim = [100, 500]
    legstr0[-1] = u"\u2013 Cellpose3 (per. + seg.)"
    for j, k in enumerate(kk):
        ax = plt.subplot(grid[1, j])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - 0.07, pos[2], pos[3]])
        img0 = imgs_all[k][iex].squeeze()
        img0 *= 1.1
        img0 = np.clip(img0, 0, 1)

        ax.imshow(img0, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_title(legstr0[k][2:], color=cols0[k], fontsize="medium")
        ax.text(1, -0.04, f"AP@0.5 = {aps[k,iex,0] : 0.2f}", va="top", ha="right",
                transform=ax.transAxes)
        if j == 0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.text(0.02, 1.2, "Denoised test image", fontsize="large",
                    fontstyle="italic", transform=ax.transAxes)

    print(aps.mean(axis=1)[:, [0, 5, 8]])

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/suppfig_specialist.pdf", dpi=100)

def suppfig_impr(folder, save_fig=True):
    aps_all = [[], []]
    imgs_all, masks_all = [[], []], [[], []]
    inds_all = [[], []]
    diams = [[], []]
    noise_types = ["poisson", "blur", "downsample"]
    for noise_type in noise_types:
        for j, ctype in enumerate(["cyto2", "nuclei"]):
            nimg_test = 68 if ctype == "cyto2" else 111
            folder_name = ctype
            root = Path(f"{folder}/images_{folder_name}/")

            dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                            allow_pickle=True).item()
            test_data = dat["test_data"][:nimg_test]
            test_noisy = dat["test_noisy"][:nimg_test]
            masks_noisy = dat["masks_noisy"][:nimg_test]
            masks_true = dat["masks_true"][:nimg_test]
            masks_data = dat["masks_orig"][:nimg_test]
            diam_test = dat["diam_test"][:nimg_test]
            noise_levels = dat["noise_levels"][:nimg_test]

            dat = np.load(root / "noisy_test" / f"test_{noise_type}_cp3_all.npy",
                            allow_pickle=True).item()

            masks_denoised = dat["masks_perseg"][:nimg_test]
            test_denoised = dat["test_perseg"][:nimg_test]
            thresholds=np.arange(0.5, 1.05, 0.05)
            ap_c, tp_d, fp_d, fn_d = metrics.average_precision(masks_true, masks_data,
                                                            threshold=thresholds)
            ap_d, tp_d, fp_d, fn_d = metrics.average_precision(masks_true, masks_denoised,
                                                            threshold=thresholds)
            ap_n, tp_n, fp_n, fn_n = metrics.average_precision(masks_true, masks_noisy,
                                                            threshold=thresholds)

            aps_all[j].append([ap_c, ap_n, ap_d])
            igood = np.nonzero(ap_d[:,0] > 0)[0]
            impr = (ap_d[igood,0] - ap_n[igood,0]) / ap_n[igood,0]
            ii = np.hstack((impr.argsort()[-2:][::-1], impr.argsort()[:2]))
            ii = igood[ii]
            imgs_all[j].append([np.array([test_data[i].squeeze(), test_noisy[i].squeeze(), test_denoised[i].squeeze()])
                            for i in ii])
            masks_all[j].append([np.array([masks_data[i].squeeze(), masks_noisy[i].squeeze(), masks_denoised[i].squeeze()])
                            for i in ii])
            diams[j].append(dat["diam_test"][ii])
            inds_all[j].append(ii)    

    colors = [["darkblue", "royalblue", [0.46, 1, 0], "cyan", "orange", "maroon"],
            ["darkblue", [0.46, 1, 0], "dodgerblue"]]

    titles = [["CellImageLibrary", "Cells : fluorescent", "Cells : nonfluorescent", 
                        "Cell membranes", "Microscopy : other", "Non-microscopy"],
                ["DSB 2018 / kaggle", "MoNuSeg (H&E)", "ISBI 2009 (fluorescent)"]]

    cinds = [[np.arange(0, 11), np.arange(11,28,1,int), np.arange(28,33,1,int), 
                        np.arange(33,42,1,int), np.arange(42,55,1,int),
                        np.arange(55,68,1,int)],
            [np.arange(0, 75), np.arange(75, 103), np.arange(103, 111)]]

    ddeg = ["noisy", "blurry", "downsampled"]
    dcorr = ["denoised", "deblurred", "upsampled"]
    dtitle = ["Denoising", "Deblurring", "Upsampling"]

    fig = plt.figure(figsize=(14,8))
    yratio = 14/10
    grid = plt.GridSpec(2, 5, hspace=0.3, wspace=0.5, 
                        left=0.05, right=0.97, top=0.95, bottom=0.05)
    il = 0
    transl = mtransforms.ScaledTranslation(-45 / 72, 5 / 72, fig.dpi_scale_trans)

    for c, ctype in enumerate(["cyto2", "nuclei"]):
        for d in range(3):
            imgs = imgs_all[c][d]
            masks = masks_all[c][d]
            inds = inds_all[c][d]
            aps = aps_all[c][d]

            ax = plt.subplot(grid[c, d + 2*(d>0)])
            pos = ax.get_position().bounds 
            ax.set_position([pos[0], pos[1]+(pos[3]-pos[2]*yratio), pos[2], pos[2]*yratio])
            for k in range(len(cinds[c])):
                ax.scatter(aps[1][cinds[c][k],0], aps[2][cinds[c][k],0], marker="x", 
                                label=titles[c][k], color=colors[c][k])
            ax.plot([0, 1], [0, 1], color="k", lw=1, ls="--")
            ax.set_xlabel(f"{ddeg[d]}, AP@0.5")
            ax.set_ylabel(f"{dcorr[d]}, AP@0.5", color=[0, 0.5, 0])
            ax.text(-0.2, 1.05, dtitle[d], fontsize="large", transform=ax.transAxes, 
                    fontstyle="italic")
            il = plot_label(ltr, il, ax, transl, fs_title)
            if d==0:
                ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1.3+c*0.4), fontsize="small")

                dstr = ["clean", "noisy", "denoised"]
                diam_mean = 30 if ctype=="cyto2" else 17
                grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=grid[c, 1:3],
                                                                        wspace=0.15, hspace=0.05)
                for j in range(4):
                    Ly, Lx = imgs[j][0].shape
                    yinds, xinds = plot.interesting_patch(masks[j][0], 
                                        bsize=min(Ly, Lx, int(300 * diams[0][0][j] / diam_mean)))
                    for k in range(3):
                        ax = plt.subplot(grid1[k, j])
                        pos = ax.get_position().bounds
                        ax.set_position([pos[0]-0.01, pos[1] - 0.015*k, *pos[2:]])
                        ax.imshow(imgs[j][k], vmin=0, vmax=1, cmap="gray")
                        ax.axis("off")
                        #outlines = utils.outlines_list(masks[j][k], multiprocessing=False)
                        #for o in outlines:
                        #    ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5, ls="--")
                        ax.set_ylim([yinds[0], yinds[-1]+1])
                        ax.set_xlim([xinds[0], xinds[-1]+1])
                        ax.text(1, -0.01, f"AP@0.5 = {aps[k][inds[j],0]:.2f}", ha="right", 
                                va="top", fontsize="small", transform=ax.transAxes)
                        if j%2==0 and k==0:
                            istr = ["most improved", "least improved"]
                            ax.set_title(f"{istr[j//2]}", fontsize="medium", fontstyle="italic")
                        if j==0:
                            ax.text(-0.05, 0.5, dstr[k], ha="right", va="center", 
                                    rotation=90, transform=ax.transAxes, 
                                    color="k" if k<2 else [0., 0.5, 0])
    fig.savefig("figs/suppfig_impr.pdf", dpi=300)


def load_benchmarks_generalist(folder, noise_type="poisson", ctype="cyto2",
                               thresholds=np.arange(0.5, 1.05, 0.05)):
    nimg_test = 68 if ctype == "cyto2" else 111
    folder_name = ctype
    root = Path(f"{folder}/images_{folder_name}/")

    masks_all = []
    imgs_all = []
    dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                  allow_pickle=True).item()
    test_data = dat["test_data"][:nimg_test]
    test_noisy = dat["test_noisy"][:nimg_test]
    masks_noisy = dat["masks_noisy"][:nimg_test]
    masks_true = dat["masks_true"][:nimg_test]
    masks_data = dat["masks_orig"][:nimg_test]
    diam_test = dat["diam_test"][:nimg_test]
    noise_levels = dat["noise_levels"][:nimg_test]
    masks_all.append(masks_data)
    masks_all.append(masks_noisy)
    imgs_all.append(test_data)
    imgs_all.append(test_noisy)

    dat = np.load(root / "noisy_test" / f"test_{noise_type}_cp3_all.npy",
                allow_pickle=True).item()
    istrs = ["perseg", "noise_spec", "data_spec", "gen"]
    for istr in istrs:
        test_dn = dat[f"test_{istr}"][:nimg_test]
        masks_dn = dat[f"masks_{istr}"][:nimg_test]
        imgs_all.append(test_dn)
        masks_all.append(masks_dn)
            
    # benchmarking
    aps = []
    tps = []
    fps = []
    fns = []
    for k in range(len(masks_all)):
        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_all[k],
                                                   threshold=thresholds)
        aps.append(ap)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    aps, tps, fps, fns = np.array(aps), np.array(tps), np.array(fps), np.array(fns)

    return (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true,
            diam_test)


def fig6(folder, save_fig=True):
    folders = [
        "cyto2", "nuclei", "tissuenet", "livecell", "yeast_BF", "yeast_PhC",
        "bact_phase", "bact_fluor", "deepbacs"
    ]
    diam_mean = 30.

    #iexs = [340, 50, 10, 5, 70, 2, 33]
    iexs = [305, 1071, 0, 3, 70, 9, 31]
    imgs, lbls = [[], [], []], []
    masks = [[], [], []]
    for f, iex in zip(folders[2:], iexs):
        dat = np.load(Path(folder) / f"{f}_generalist_masks.npy",
                      allow_pickle=True).item()
        img = dat["imgs"][iex].copy()
        img = img[:1] if img.ndim > 2 else img
        img = np.maximum(0, transforms.normalize99(img))
        imgs[0].append(img)
        masks[0].append(dat["masks_pred"][iex])
        lbls.append(dat["masks"][iex].astype("uint16"))

    diams = [utils.diameters(lbl)[0] for lbl in lbls]

    gen_model = "/home/carsen/dm11_string/datasets_cellpose/models/per_1.00_seg_1.50_rec_0.00_poisson_blur_downsample_2024_08_20_11_46_25.557039"
    model = denoise.DenoiseModel(gpu=True, nchan=1, diam_mean=diam_mean,
                                 pretrained_model=gen_model)
    seg_model = models.CellposeModel(gpu=True, model_type="cyto3")
    pscales = [1.5, 20., 1.5, 1., 5., 40., 3.]
    denoise.deterministic()
    for i, img in tqdm(enumerate(imgs[0])):
        img0 = torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0)
        img0 = img0.float()
        noisy0 = denoise.add_noise(img0, poisson=1., downsample=0., blur=0.,
                                   pscale=pscales[i]).numpy().squeeze()
        denoised0 = model.eval(noisy0, diameter=diams[i])
        imgs[1].append(noisy0)
        imgs[2].append(denoised0)
        for j in range(1, 3):
            masks[j].append(
                seg_model.eval(
                    imgs[j][i], diameter=diams[i], channels=[0, 0], tile_overlap=0.5,
                    flow_threshold=0.4, augment=True, bsize=224,
                    niter=2000 if folders[i - 2] == "bact_phase" else None)[0])
    api = np.array(
        [metrics.average_precision(lbls, masks[i])[0][:, 0] for i in range(3)])

    thresholds = np.arange(0.5, 1.05, 0.05)
    aps = []
    for i in range(6):
        ctype = "cyto2" if i < 3 else "nuclei"
        noise_type = ["poisson", "blur", "downsample"][i % 3]
        out = load_benchmarks_generalist(folder, ctype=ctype, noise_type=noise_type,
                                         thresholds=thresholds)
        (aps0, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true,
         diam_test) = out
        print(ctype, noise_type, aps0[1:, :, 0].mean(axis=1))
        aps.append(aps0)

    fig = plt.figure(figsize=(14, 7), dpi=100)
    yratio = 14 / 7
    grid = plt.GridSpec(3, 14, figure=fig, left=0.02, right=0.97, top=0.97, bottom=0.1,
                        wspace=0.05, hspace=0.2)

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=grid[0, :],
                                                        wspace=0.4, hspace=0.15)

    transl = mtransforms.ScaledTranslation(-0 / 72, 3 / 72, fig.dpi_scale_trans)
    il = 0
    noise_type = ["poisson", "blur", "downsample"][i % 3]

    ax = plt.subplot(grid1[0:2])
    pos = ax.get_position().bounds
    im = plt.imread("figs/cellpose3_models.png")
    yr = im.shape[0] / im.shape[1]
    w = 0.22
    ax.set_position([0.0, pos[1]-0.08, w, w*yratio*yr])
    plt.imshow(im)
    ax.axis("off")
    ax.text(0.08, 1.02, "General restoration models", transform=ax.transAxes,
                    fontstyle="italic", fontsize="large")
    il = plot_label(ltr, il, ax, transl, fs_title)
    
    transl = mtransforms.ScaledTranslation(-40 / 72, 20 / 72, fig.dpi_scale_trans)
    thresholds = np.arange(0.5, 1.05, 0.05)
    cols0 = np.array([[0, 0, 0], [0, 0, 0], [0, 128, 0], [180, 229, 162], 
                    [246, 198, 173], [192, 71, 29], ])
    cols0 = cols0 / 255
    lss0 = ["-", "-", "-","-", "-", "-"]
    legstr0 = ["", u"\u2013 noisy image", u"\u2013 original", 
                u"\u2013 noise-specific", "\u2013 data-specific", u"-- one-click"]
    theight = [0, 0,4,3,2,1]
    for i in range(6):
        ctype = "cellpose test set" if i < 3 else "nuclei test set"
        noise_type = ["denoising", "deblurring", "upsampling"][i % 3]

        ax = plt.subplot(grid1[i+2])
        pos = ax.get_position().bounds
        ax.set_position([
            pos[0] + 0.025 * (i>2), pos[1] - 0.05, # (5 - i) * 0.01 - 0.02 + 0.03 * (i > 2)
            pos[2] * 0.92, pos[3]
        ])
        for k in range(1, len(aps[0])):
            ax.plot(thresholds, aps[i][k].mean(axis=0), color=cols0[k], ls=lss0[k], lw=1)
        if i == 0 or i == 3:
            ax.set_ylabel("average precision (AP)")
            ax.set_xlabel("IoU threshold")
            il = plot_label(ltr, il, ax, transl, fs_title)
        if i == 1 or i == 4:
            ax.text(0.5, 1.18, ctype, transform=ax.transAxes, ha="center",
                    fontsize="large")
        
        ax.set_ylim([0, 0.72])
        ax.set_xticks(np.arange(0.5, 1.05, 0.25))
        ax.set_xlim([0.5, 1.0])
        ax.set_title(f"{noise_type}", fontsize="medium")

    titlesj = ["clean", "noisy", "denoised (one-click)"]
    titlesi = [
        "Tissuenet", "Livecell", "Yeaz bright-field", "YeaZ phase-contrast",
        "Omnipose phase-contrast", "Omnipose fluorescent", "DeepBacs"
    ]
    colsj = cols0[[0, 1, -1]]

    ly0 = 250

    transl = mtransforms.ScaledTranslation(-15 / 72, 30 / 72, fig.dpi_scale_trans)
    for i in range(len(imgs[0])):
        ratio = diams[i] / 30.
        d = utils.diameters(lbls[i])[0]
        ly = ly0 * (d / 30.) * (1 + 0.5 * (i == 6))
        yr, xr = plot.interesting_patch(lbls[i], bsize=ly)

        mask_gt = lbls[i].copy()
        #outlines_gt = utils.outlines_list(mask_gt, multiprocessing=False)

        for j in range(1, 3):
            img = np.clip(transforms.normalize99(imgs[j][i].copy().squeeze()), 0, 1)
            for k in range(2):
                ax = plt.subplot(grid[j, 2 * i + k])
                pos = ax.get_position().bounds
                ax.set_position([
                    pos[0] + 0.003 * i - 0.00 * k, pos[1] - (2 - j) * 0.025 - 0.07,
                    pos[2], pos[3]
                ])
                if 1:
                    ax.imshow(img, cmap="gray", vmin=0,
                              vmax=0.35 if j == 1 and i == 2 else 1.0)
                    if k == 1:
                        outlines = utils.outlines_list(masks[j][i],
                                                       multiprocessing=False)
                        #for o in outlines_gt:
                        #    ax.plot(o[:,0], o[:,1], color=[0.7,0.4,1], lw=1, ls="-")
                        for o in outlines:
                            ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5,
                                    ls="--")
                        ax.text(1, -0.015, f"AP@0.5 = {api[j,i] : 0.2f}", va="top",
                                ha="right", transform=ax.transAxes, fontsize="medium")

                ax.set_ylim([yr[0], yr[-1]])
                if i == 4 or i == 6:
                    ax.set_xlim([xr[0], xr[-1] - (xr[-1] - xr[0]) / 2])
                else:
                    ax.set_xlim(
                        [xr[0] + (xr[-1] - xr[0]) / 4, xr[-1] - (xr[-1] - xr[0]) / 4])
                ax.axis("off")
                if k == 0 and i == 0:
                    ax.text(-0.22, 0.5, titlesj[j], transform=ax.transAxes, va="center",
                            rotation=90, color=colsj[j], fontsize="medium")
                    if j == 0:
                        il = plot_label(ltr, il, ax, transl, fs_title)
                        ax.text(-0.0, 1.22, "Denoising examples from other datasets",
                                fontstyle="italic", transform=ax.transAxes,
                                fontsize="large")
                if k == 0 and j == 0:
                    ax.text(0.0, 1.05, titlesi[i], transform=ax.transAxes,
                            fontsize="medium")
    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/fig6.pdf", dpi=150)

def load_seg_generalist(folder):
    folders = [
        "cyto2", "nuclei", "tissuenet", "livecell", "yeast_BF", "yeast_PhC",
        "bact_phase", "bact_fluor", "deepbacs"
    ]
    nd = len(folders)

    ap0 = []
    apc = []
    rec = []
    prec = []
    imgs = []
    masks_true = []
    masks_pred = []
    api = []
    iexs = [
        [0, 17, 20],
        [110, 10, 95],
        [2, 200, 110],
        [1071, 500, 0],  # 607, 543
        [0, 10],
        [3],
        [63, 70, 0],
        [9, 50, 26],
        [31, 17, 10]
    ]

    net_types = ["generalist", "specialist", "transformer"]
    apcs = []
    for net_type in net_types:
        apc = []
        for i, f in enumerate(folders):
            try:
                dat = np.load(
                    Path(folder) / f"{f}_{net_type}_masks.npy",
                    allow_pickle=True).item()
            except:
                apc.append(np.zeros(ap.shape[-1]))
                continue
            ap, tp, fp, fn = dat["performance"]
            igood = np.ones(len(ap), "bool")
            if f == "deepbacs":
                igood[np.arange(1, 10, 2)] = False
            apc.append(ap[igood].mean(axis=0))
            if net_type == "generalist":
                if i != 5:
                    imgs.append([dat["imgs"][iex] for iex in iexs[i]])
                    masks_true.append([dat["masks"][iex] for iex in iexs[i]])
                    masks_pred.append([dat["masks_pred"][iex] for iex in iexs[i]])
                    api.append(ap[np.array(iexs[i]), 0])
                else:
                    imgs[-1].extend([dat["imgs"][iex] for iex in iexs[i]])
                    masks_true[-1].extend([dat["masks"][iex] for iex in iexs[i]])
                    masks_pred[-1].extend([dat["masks_pred"][iex] for iex in iexs[i]])
                    api[-1] = np.append(api[-1], ap[np.array(iexs[i]), 0])
                # plt.figure(figsize=(3,3))
                # plt.imshow(imgs[-1][-1][0] if imgs[-1][-1].ndim==3 else imgs[-1][-1])
                # plt.show()

        apcs.append(apc)

    apcs = np.array(apcs)

    return apcs, api, imgs, masks_true, masks_pred


def fig5(folder, save_fig=True):
    thresholds = np.arange(0.5, 1.05, 0.05)
    apcs, api, imgs, masks_true, masks_pred = load_seg_generalist(folder)
    titlesi = [
        "Cellpose", "Nuclei", "Tissuenet", "Livecell", "YeaZ\nbright-field",
        "YeaZ\nphase-contrast", "Omnipose\nphase-contrast", "Omnipose\nfluorescent",
        "DeepBacs"
    ]
    nd = len(titlesi)

    fig = plt.figure(figsize=(14, 9), dpi=100)
    grid = plt.GridSpec(5, 8, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.05,
                        wspace=0.12, hspace=0.05)
    il = 0

    from cellpose import plot
    transl = mtransforms.ScaledTranslation(-19 / 72, 22 / 72, fig.dpi_scale_trans)
    ly0 = 240
    for i in range(nd - 1):
        for j in range(3):
            ax = plt.subplot(grid[j, i])
            pos = ax.get_position().bounds
            ax.set_position([pos[0], pos[1] - 0.05 - j * 0.005, pos[2],
                             pos[3]])  #-0.12+(j)*0.02
            d = utils.diameters(masks_true[i][j])[0]
            ly = ly0 * (d / 30.) * (1 + 1. * (i == 5) * (j == 0) + 0.5 * (i == 7) *
                                    (j < 2)
                                   )  #+ 0.5*(i==3) + 0.35*(i==6) + 0.5*(i==7) + 1)
            yr, xr = plot.interesting_patch(masks_true[i][j], bsize=ly)
            img = imgs[i][j].copy()
            img = img[np.newaxis, ...] if img.ndim == 2 else img
            if img.shape[0] == 1 or img[1].sum() == 0:
                ax.imshow(
                    transforms.normalize99(img[0]) * 1., cmap="gray", vmin=0, vmax=1)
            else:
                img = np.vstack((np.zeros((1, *img.shape[-2:])), img))
                img = np.clip(img * 1., 0, 1).transpose(1, 2, 0)
                ax.imshow(img)
            ax.text(1, -0.015, f"AP@0.5 = {api[i][j] : 0.2f}", va="top", ha="right",
                    transform=ax.transAxes, fontsize="small")
            ax.set_ylim([yr[0], yr[-1]])
            ax.set_xlim([xr[0], xr[-1]])
            ax.axis("off")

            maskk = masks_true[i][j].copy()
            outlines = utils.outlines_list(maskk, multiprocessing=False)
            for o in outlines:
                ax.plot(o[:, 0], o[:, 1], color=[0.7, 0.4, 1], lw=0.75, ls="-")

            maskk = masks_pred[i][j].copy()
            outlines = utils.outlines_list(maskk, multiprocessing=False)
            for o in outlines:
                ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1., ls="--")

            if i == 0 and j == 0:
                il = plot_label(ltr, il, ax, transl, fs_title)
                ax.text(-0.02, 1.2, 'Super-generalist "cyto3" model segmentations',
                        fontstyle="italic", fontsize="large", transform=ax.transAxes)

            if j == 0:
                ax.text(0.5, 1.05, titlesi[i + (i > 4)] if i != 4 else "YeaZ",
                        ha="center", transform=ax.transAxes, fontsize="medium")

    lss = ["-", "--", "-"]
    cols = ["k", 0.35 * np.ones(3), [0, 0, 1]]
    lws = [1.5, 1.5, 1, 2]
    net_types = [
        "super-\n   generalist", "dataset-\n   specific", "adaptive-net", "transformer"
    ]
    legstrn = [u"\u2013 %s" % net_types[k] for k in range(len(net_types))]
    legstrn[1] = f"-- {net_types[1]}"
    #legstrn[2] += "-net"
    theight = [1, 3, 0]
    zorder = [10, 20, 0]

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=grid[-2:, :],
                                                        wspace=0.35, hspace=0.5)
    transl = mtransforms.ScaledTranslation(-45 / 72, 27 / 72, fig.dpi_scale_trans)
    for i in range(9):
        ax = plt.subplot(grid1[0, i])
        pos = ax.get_position().bounds
        ax.set_position([
            pos[0] + 0.035 * (i > 0) + 0.02 - (i - 1) * 0.005, pos[1] + 0.03,
            pos[2] - 0.015, pos[3] * 0.54
        ])
        for j in [0, 1, 2]:
            ax.plot(np.arange(0.5, 1.05, 0.05), apcs[j, i].T, lw=lws[j], color=cols[j],
                    ls=lss[j], alpha=1, zorder=zorder[j])
            if i == 0:
                ax.text(0.63, 0.5 + theight[j] * 0.11, legstrn[j],
                        transform=ax.transAxes, color=cols[j])
        ax.set_title(titlesi[i], fontsize="medium", loc="center")
        ax.set_xlim([0.5, 1.0])
        ax.set_xticks([0.5, 0.75, 1.0])
        if i > 3:
            ax.set_ylim([0, 1])
            ax.set_yticks(np.arange(0, 1.1, 0.2))
        else:
            ax.set_ylim([0, 0.85])
            ax.set_yticks(np.arange(0, 0.9, 0.2))
        ax.set_xticklabels(["0.5", "0.75", "1.0"])
        if i == 0:
            ax.set_ylabel("average precision (AP)")
            ax.set_xlabel("IoU threshold")
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.text(-0.4, 1.2, "Segmentation performance", fontstyle="italic",
                    fontsize="large", transform=ax.transAxes)

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/fig5.pdf", dpi=100)


if __name__ == "__main__":
    # folder with folders images_cyto2 and images_nuclei for those datasets
    folder = "/media/carsen/ssd4/datasets_cellpose/"

    # https://publications.mpi-cbg.de/publications-sites/7207/
    folder2 = "/media/carsen/ssd4/denoising/"  # download CARE datasets here

    # denoising cells
    fig1(folder, save_fig=0)
    plt.show()
    fig2(folder, folder2=folder2 + "Projection_Flywing/test_data", save_fig=0)
    plt.show()
    suppfig_seg(folder, save_fig=0)
    plt.show()

    # denoising specialist
    suppfig_specialist(folder, save_fig=0)
    plt.show()

    # denoising nuclei
    fig3(folder, folder2=folder2 + "Denoising_Tribolium/test_data", save_fig=0)
    plt.show()
    suppfig_nuclei(folder, save_fig=0)
    plt.show()

    # deblurring / upsampling
    fig4(folder, save_fig=0, ctype="cyto2")
    plt.show()
    fig4(folder, save_fig=0, ctype="nuclei")
    plt.show()

    # ex images
    suppfig_impr(folder, save_fig=0)
    plt.show()

    # one-click + supergeneralist
    fig5(folder, save_fig=0)
    plt.show()
    fig6(folder, save_fig=0)
    plt.show()


def old_nuclei_ex(folder, save_fig=False):
    out = load_benchmarks(folder, ctype="nuclei", thresholds=thresholds)
    (aps, tps, fps, fns, imgs_all, noise_levels, masks_all, masks_true, diam_test,
     flows_perseg, flows_true) = out
    iex = 107  #5
    print(iex)
    ylim = [0, 400]  #[0, 500]
    xlim = [0, 300]  #[200, 500]

    xi = [0, 1, 1, 0, 1]
    dx = 0.0
    dy = 0.06

    for j, k in enumerate([0, 1]):
        imgk = transforms.normalize99(imgs_all[k][iex].squeeze().copy()) * 1.1
        ax = plt.subplot(grid[-2:, j])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - dy, pos[2], pos[3]])
        il = plot_label(ltr, il, ax, transl, fs_title)
        ax.imshow(imgk.T, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_title(titles[k], color=cols[k], fontsize="medium")
        if j == 0:
            ax.text(-0., 1.2, "Nucleus denoising", fontsize="large",
                    transform=ax.transAxes, fontstyle="italic")
        ax.text(1, -0.04, f"AP@0.5 = {aps[k,iex,0] : 0.2f}", va="top", ha="right",
                transform=ax.transAxes)

    mask_gt = masks_true[iex].copy().T
    outlines_gt = utils.outlines_list(mask_gt, multiprocessing=False)

    for j, k in enumerate([4, 7]):
        imgk = transforms.normalize99(imgs_all[k][iex].squeeze().copy()) * 1.
        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=grid[-2:, 2 * j + 2:2 * j + 4], wspace=0.05, hspace=0.1)
        ax = plt.subplot(grid1[0, 0])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - dy, pos[2], pos[3]])
        il = plot_label(ltr, il, ax, transl, fs_title)
        ax.imshow(imgk.T, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_title(titles[k], color=cols[k], fontsize="medium")

        maskk = masks_all[k][iex].copy().T
        outlines = utils.outlines_list(maskk, multiprocessing=False)
        ax = plt.subplot(grid1[0, 1])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1] - dy, pos[2], pos[3]])
        ax.imshow(imgk.T, cmap="gray", vmin=0, vmax=1)
        for o in outlines_gt:
            ax.plot(o[:, 0], o[:, 1], color=[0.7, 0.4, 1], lw=1, ls="-")
        for o in outlines:
            ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5, ls="--")
        ax.axis("off")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_title("segmentation", fontsize="medium")
        ax.text(0, -0.04, f"AP@0.5 = {aps[k,iex,0] : 0.2f}", va="top", ha="center",
                transform=ax.transAxes)

    transl = mtransforms.ScaledTranslation(-45 / 72, -2.5 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid[-2:, -2:-1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + 0.05, pos[1] - 0.03, pos[2],
                     pos[3]])  #+pos[3]*0.15-0.03, pos[2], pos[3]*0.7])
    il = plot_label(ltr, il, ax, transl, fs_title)
    #theight = [0, 1, 2, 3, 5, 6, 7, 8, 4]
    theight = [0, 1, 2, 3, 5, 6, 6, 7, 4]
    for k in [1, 2, 3, 4, 6, 7, 8]:
        ax.plot(thresholds, aps[k, :, :].mean(axis=0), color=cols[k], ls=lss[k])
        ax.text(0.7, 0.4 + 0.06 * theight[k], legstr[k], color=cols[k],
                transform=ax.transAxes)
    ax.set_ylim([0, 0.72])
    ax.set_ylabel("average precision (AP)")
    ax.set_xlabel("IoU threshold")
    ax.set_xticks(np.arange(0.5, 1.05, 0.1))
    ax.set_xlim([0.5, 1.0])

    if save_fig:
        os.makedirs("figs/", exist_ok=True)
        fig.savefig("figs/fig2.pdf", dpi=100)
