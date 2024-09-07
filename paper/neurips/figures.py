
from fig_utils import * 
import matplotlib.patheffects as pe

def fig1(imgs_norm, masks_true, dat, timings, save_fig=False):
    fig = plt.figure(figsize=(14, 5.5))
    thresholds = dat["thresholds"]
    grid = plt.GridSpec(2, 5, figure=fig, left=0.05, right=0.98, top=0.94, bottom=0.09,
                            wspace=0.4, hspace=0.6)
    il = 0
    transl = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    transl1 = mtransforms.ScaledTranslation(-38 / 72, 7 / 72, fig.dpi_scale_trans)

    iex = 54
    img0 = np.clip(imgs_norm[iex].copy().transpose(1,2,0), 0, 1)
    xlim = [300, 660]
    ylim = [350, 650]

    cols = {"grayscale": [0.5, 0.5, 1], 
            "maetal": "b", 
            "default": "g", 
            "mediar": "r",
            "transformer": [0,1,0]}
    titles = {"grayscale": "Cellpose (impaired)", 
            "maetal": "Cellpose (Ma et al)", 
            "default": "Cellpose (default)", 
            "mediar": "Mediar",
            "transformer": "Cellpose (transformer)"}

    ax = plt.subplot(grid[0,0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0] - 0.03, pos[1], pos[2] + 0.035, pos[3]])
    ax.imshow(img0)#, aspect="auto")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.set_title("Example validation image")

    maskk = masks_true[iex].copy()
    outlines_gt = utils.outlines_list(maskk, multiprocessing=False)

    pltmasks = [(0, 1, "maetal"), 
                (0, 2, "default"),
                (1, 0, "mediar"),
                (1, 1, "transformer"),
                ]

    for k,pltmask in enumerate(pltmasks):
        ax = plt.subplot(grid[pltmask[0], pltmask[1]])
        pos = ax.get_position().bounds
        ax.set_position([pos[0] - 0.03, pos[1], pos[2] + 0.035, pos[3]])
        il = plot_label(ltr, il, ax, transl, fs_title)
        ax.imshow(img0)
        maskk = dat[pltmask[2]][iex].copy()
        outlines = utils.outlines_list(maskk, multiprocessing=False)
        for o in outlines_gt:
            ax.plot(o[:, 0], o[:, 1], color=[0.7,0.4,1], lw=1, ls="-")
        for o in outlines:
            ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5, ls="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")
        if k==0:
            ax.set_title("Cellpose (Ma et al, 2024)", color=cols[pltmask[2]])
        else:
            ax.set_title(titles[pltmask[2]], color=cols[pltmask[2]])
        if k==0:
            ax.text(-0.1, -0.1, "ground-truth", color=[0.7, 0.4, 1], transform=ax.transAxes, 
                    ha="left", fontweight="normal", fontsize="large",
                    path_effects=[pe.withStroke(linewidth=1, foreground="k")])
            ax.text(-0.1, -0.22, "model", color=[1, 1, 0.3], transform=ax.transAxes, 
                    ha="left", fontweight="normal", fontsize="large",
                    path_effects=[pe.withStroke(linewidth=1, foreground="k")])
        f1 = dat[pltmask[2]+"_f1"][iex, 0]
        ax.text(1, -0.1, f"F1@0.5 = {f1:.2f}", transform=ax.transAxes, ha="right")
        
    ax = plt.subplot(grid[1,2])
    il = plot_label(ltr, il, ax, transl1, fs_title)
    mtypes = ["default", "transformer", "mediar"]
    dx = 0.4 
    for k, mtype in enumerate(mtypes):
        tsec = timings[:,k]
        vp = ax.violinplot(tsec, positions=k*np.ones(1), bw_method=0.1,
                        showextrema=False, showmedians=False)#, quantiles=[[0.25, 0.5, 0.75]])
        ax.plot(dx*np.arange(-1, 2, 2) + k, 
                np.median(tsec) * np.ones(2), 
                color=cols[mtype])
        vp["bodies"][0].set_facecolor(cols[mtype])
        ax.text(k+0.2 if k>0 else k-0.1, -1, titles[mtype].replace(" (", "\n("), 
                color=cols[mtype], rotation=0, 
                va="top", ha="center")
    ax.set_xticklabels([])
    ax.text(-0.1, 1.05, "Test set runtimes", 
                fontsize="large", transform=ax.transAxes)
    ax.set_ylabel("runtime per image (sec.)")

    ax = plt.subplot(grid[:2, 3])
    il = plot_label(ltr, il, ax, transl1, fs_title)
    f1s = np.array([[0.8612, 0.8346, 0.7976, 0.7013, 0.4116],
                [0.8484, 0.8190, 0.7761, 0.6744, 0.3907],
                [0.8263, 0.7903, 0.7371, 0.6063, 0.2911],
                ])
    mtypes = ["default", "transformer", "mediar"]
    for k, mtype in enumerate(mtypes):
        ax.plot(np.arange(0.5, 1, 0.1), f1s[k], color=cols[mtype], lw=3)
        ax.text(0.1, 0.5-k*0.13 if k<2 else 0.5-k*0.13+0.05, titles[mtype].replace(" (", "\n("), 
                color=cols[mtype], fontsize="large", transform=ax.transAxes)
    ax.set_ylim([0, 0.9])
    ax.set_xlim([0.49, 0.91])
    ax.set_xticks([0.5, 0.7, 0.9])
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("Test set results")
    
    mtypes = ["default", "transformer", "mediar", "maetal", "grayscale"]
    dx = 0.3
    stype = "f1"
    ax = plt.subplot(grid[0,4])
    for k, mtype in enumerate(mtypes): #enumerate(model_types):
        score = dat[f"{mtype}_{stype}"][:,0]
        vp = ax.violinplot(score, positions=k*np.ones(1), bw_method=0.1,
                        showextrema=False, showmedians=False)#, quantiles=[[0.25, 0.5, 0.75]])
        ax.plot(dx*np.arange(-1, 2, 2) + k, 
                np.median(score) * np.ones(2), 
                color=cols[mtype])
        vp["bodies"][0].set_facecolor(cols[mtype])
        ax.text(k+0.2, -0.06, titles[mtype].replace(" (", "\n("), 
                color=cols[mtype], rotation=90, 
                va="top", ha="center")
    ax.text(-0.1, 1.05, "Validation set scores", 
                fontsize="large", transform=ax.transAxes)  
    il = plot_label(ltr, il, ax, transl1, fs_title)
    ax.set_ylabel("F1 score @ 0.5 IoU")
    ax.set_xticks(np.arange(len(mtypes)))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim([-0.01, 1.01])
    ax.set_xticklabels([])

    ax = plt.subplot(grid[1,4])
    for k, mtype in enumerate(mtypes):
        ax.errorbar(thresholds, np.median(dat[f"{mtype}_f1"], axis=0), 
                    dat[f"{mtype}_f1"].std(axis=0) / ((dat[f"{mtype}_f1"].shape[0]-1)**0.5), 
                    color=cols[mtype], lw=2, #if mtype=="grayscale" else 1,
                    ls="--" if mtype=="transformer" else "-", zorder=30 if mtype=="maetal" else 0)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("IoU threshold")
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([0.49, 1.01])
    ax.set_xticks([0.5, 0.75, 1.0])
    
    if save_fig:
        fig.savefig("figs/fig1_neurips.pdf", dpi=100)

def fig2(imgs_norm, masks_true, dat, type_names, types, emb, emb_test, save_fig=False):
    ids = [0, 3, 56, 55, 81, 75]

    outlines_gt = [utils.outlines_list(masks_true[iex], multiprocessing=False) for iex in ids]
    outlines_cp = [utils.outlines_list(dat["default"][iex], multiprocessing=False)    for iex in ids]
    outlines_m = [utils.outlines_list(dat["mediar"][iex], multiprocessing=False) for iex in ids]

    fig = plt.figure(figsize=(14,10))
    grid = plt.GridSpec(4, 6, figure=fig, left=0.025, right=0.98, top=0.97, bottom=0.04,
                        wspace=0.1, hspace=0.2)
    il = 0
    transl = mtransforms.ScaledTranslation(-20 / 72, 14 / 72, fig.dpi_scale_trans)
    transl1 = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)

    ylims = [[0, 500], [1550, 1950], [450, 700], [250, 500], [300, 700], [400, 800]]
    xlims = [[0, 600], [500, 900], [300, 550], [100, 350], [0, 400], [200, 600]]

    for j in range(len(ids)):
        iex = ids[j]
        img0 = np.clip(imgs_norm[iex].transpose(1,2,0).copy(), 0, 1)
        
        ax = plt.subplot(grid[0,j])
        maskk = dat["default"][iex].copy()
        ax.imshow(img0)
        for o in outlines_gt[j]:
            ax.plot(o[:, 0], o[:, 1], color=[0.7,0.4,1], lw=2, ls="-", rasterized=True)
        for o in outlines_cp[j]:
            ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5, ls="--", rasterized=True)
        ax.set_ylim(ylims[j])
        ax.set_xlim(xlims[j])
        ax.axis("off")
        f1 = dat["default_f1"][iex,0]
        ax.text(1, -0.1, f"F1@0.5 = {f1:.2f}", transform=ax.transAxes, ha="right")
        if j==0:
            ax.text(-0.1, 0.5, "Cellpose (default)", rotation=90, va="center", transform=ax.transAxes)    
            ax.set_title("Example validation images", y=1.07)
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.text(-0.1, -0.18, "ground-truth", color=[0.7, 0.4, 1], transform=ax.transAxes, 
                    ha="left", fontweight="normal", fontsize="large",
                    path_effects=[pe.withStroke(linewidth=1, foreground="k")])
            ax.text(-0.1, -0.3, "model", color=[1, 1, 0.3], transform=ax.transAxes, 
                    ha="left", fontweight="normal", fontsize="large",
                    path_effects=[pe.withStroke(linewidth=1, foreground="k")])

        ax = plt.subplot(grid[1,j])
        maskk = dat["mediar"][iex].copy()
        ax.imshow(img0)
        for o in outlines_gt[j]:
            ax.plot(o[:, 0], o[:, 1], color=[0.7,0.4,1], lw=2, ls="-", rasterized=True)
        for o in outlines_m[j]:
            ax.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=1.5, ls="--", rasterized=True)
        ax.set_ylim(ylims[j])
        ax.set_xlim(xlims[j])
        f1 = dat["mediar_f1"][iex,0]
        ax.text(1, -0.1, f"F1@0.5 = {f1:.2f}", transform=ax.transAxes, ha="right")
        ax.axis("off")
        if j==0:
            ax.text(-0.1, 0.5, "Mediar", rotation=90, va="center", transform=ax.transAxes)    
            
    ax = plt.subplot(grid[2:, :])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], pos[2]-0.02, pos[3]-0.05])
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=ax,
                                                        wspace=0.15, hspace=0.15)
    ax.remove()
    cols = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    
    ax = plt.subplot(grid1[0,0])    
    cols0 = plt.get_cmap("Paired")(np.linspace(0, 1, 12))
    cols = np.zeros((len(type_names), 4))
    cols[:,-1] = 1
    cols[:2] = cols0[:2]
    cols[2] = cols0[3]
    cols[4] = cols0[-3]
    cols[-2:] = cols0[4:6]
    cols[3] = np.array([0,1.,1.,1.])
    cols[6] = cols0[6]
    cols[7] = cols0[-1]
    irand = np.random.permutation(len(emb)-100)
    ax.scatter(emb[:-100,1][irand], emb[:-100,0][irand], color=cols[types[:-100]][irand],#, cmap="tab10",
                    s=1, alpha=0.5, marker="o", rasterized=True, zorder=-10)
    new_names = ["Omnipose (fluor.)", "Omnipose (phase)", "Cellpose", "DeepBacs", "Livecell", "Ma et al, 2024", "Nuclei", "Tissuenet", "YeaZ (BF)", "YeaZ (phase)"]
    ax.set_title("t-SNE of image style vectors\n(training set)", va="top", y=1.05)
    torder = np.array([5, 2, 6, 4, 7, 0, 1, 3, 8, 9])
    for k in range(len(type_names)):
        th = (torder==k).nonzero()[0][0]
        ax.text(0.9, 0.93-0.045*th, new_names[k], color=cols[k], 
                transform=ax.transAxes, fontsize="small")
    ax.axis("off")
    il = plot_label(ltr, il, ax, transl1, fs_title)

    dx = 0.03
    ax = plt.subplot(grid1[0,1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx, pos[1], pos[2], pos[3]])
    ax.scatter(emb[:-100,1], emb[:-100,0], color=0.8*np.ones(3), s=1, rasterized=True)
    s1 = ax.scatter(emb[-100:,1], emb[-100:,0], color="k",
                s=50, marker="x", alpha=1, lw=0.5, rasterized=True)
    s2 = ax.scatter(emb_test[:,1], emb_test[:,0], color="k", facecolors='none',
                s=50, marker="o", alpha=1, lw=0.5, rasterized=True)
    ax.axis("off")
    ax.set_title("Validation and test set\n(Ma et al, 2024)", va="top", y=1.05)
    ax.legend([s1, s2], ["validation", "test"], frameon=False, loc="upper left")
    il = plot_label(ltr, il, ax, transl1, fs_title)
    
    ax = plt.subplot(grid1[0,2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx, pos[1], pos[2], pos[3]])
    pos = ax.get_position().bounds
    ax.scatter(emb[:-100,1], emb[:-100,0], color=0.8*np.ones(3), s=1, rasterized=True)
    im = ax.scatter(emb[-100:,1], emb[-100:,0], c=dat["default_f1"][:,0], lw=2,
                    s=60, marker="x", alpha=1, cmap="plasma", vmin=0, vmax=1, rasterized=True)
    ax.axis("off")
    cax = fig.add_axes([pos[0]+pos[2]-0.02, pos[1]+pos[3]-0.12, 0.005, 0.11])
    plt.colorbar(im, cax=cax)
    ax.set_title("F1 score for Cellpose (default)")
    il = plot_label(ltr, il, ax, transl1, fs_title)

    ax = plt.subplot(grid1[0,3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx, pos[1], pos[2], pos[3]])
    pos = ax.get_position().bounds
    ax.scatter(emb[:-100,1], emb[:-100,0], color=0.8*np.ones(3), s=1, rasterized=True)
    im = ax.scatter(emb[-100:,1], emb[-100:,0], c=dat["default_f1"][:,0] - dat["mediar_f1"][:,0], lw=2,
                    s=60, marker="x", alpha=1, cmap="coolwarm", vmin=-0.3, vmax=0.3, rasterized=True)
    ax.axis("off")
    cax = fig.add_axes([pos[0]+pos[2]-0.02, pos[1]+pos[3]-0.12, 0.005, 0.11])
    plt.colorbar(im, cax=cax)
    ax.set_title("$\Delta$F1, Cellpose (default) - Mediar")
    il = plot_label(ltr, il, ax, transl1, fs_title)

    if save_fig:
        fig.savefig("figs/fig2_neurips.pdf", dpi=200)
