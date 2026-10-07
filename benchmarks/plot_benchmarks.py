"""Render inference latency, speedup and SAM mask-stage figures from JSON reports.

Requires NumPy and Matplotlib. Figures use measured samples only: latency
whiskers show the repeat range, and speedup whiskers show the paired repeat range.
These ranges are descriptive, not confidence intervals.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REFERENCE = "#64748b"
OPTIMIZED = "#0f766e"
INK = "#172b3a"
IMAGE_NAMES = {"gray_2D.png": "Grayscale PNG", "rgb_2D.png": "RGB PNG",
               "rgb_2D_tif.tif": "RGB TIFF", "gray_3D.tif": "Grayscale volume"}


def load_report(path):
    report = json.loads(Path(path).read_text())
    if not report.get("all_exact"):
        raise ValueError(f"Report has no passing exact equivalence result: {path}")
    return report


def hardware_label(report):
    return (report.get("gpu") or report.get("device", "CPU")).replace(
        "NVIDIA ", "").replace(" Max-Q Workstation Edition", "")


def samples(row, stage=None):
    result = row if stage is None else row[stage]
    ref = np.asarray(result["reference_seconds"], dtype=float)
    opt = np.asarray(result["candidate_seconds"], dtype=float)
    if ref.shape != opt.shape or ref.size == 0 or np.any(ref <= 0) or np.any(opt <= 0):
        raise ValueError("Expected positive timings with one reference per candidate repeat")
    return ref, opt


def row_label(row, kind):
    name = IMAGE_NAMES.get(row["image"], row["image"])
    if kind == "cyto3":
        return f"{name}\nCell diameter: {row['diameter']:g} pixels"
    if "3D" in row["image"]:
        dimensions = " x ".join(map(str, row["shape"][:3]))
        return f"{name}: {dimensions} voxels\nCheckpoint: {row['model']}"
    return f"{name}\nCheckpoint: {row['model']}"


def positions(rows, kind):
    groups = [row["diameter"] if kind == "cyto3" else row["model"] for row in rows]
    pos, gap = [], 0.
    for i, group in enumerate(groups):
        if i and group != groups[i - 1]:
            gap += .5
        pos.append(i + gap)
    return np.asarray(pos)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(colors=INK, labelsize=10)


def range_error(center, values):
    return [[center - values.min()], [values.max() - center]]


def export(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg", "pdf"]:
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=300,
                    bbox_inches="tight", facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def overview(report, kind, directory):
    rows = report["results"]
    y = positions(rows, kind)
    stage = None if kind == "cyto3" else "eval"
    times = [samples(row, stage) for row in rows]
    is_volume = report["parameters"].get("do_3D", False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8 if is_volume else 7.8), sharey=True,
                             gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=.28 if is_volume else .24, right=.97,
                        top=.72 if is_volume else .80,
                        bottom=.28 if is_volume else .19, wspace=.18)
    title = "Cellpose 3 (cyto3)" if kind == "cyto3" else "Cellpose-SAM"
    fig.text(.035, .95, f"{title}: total segmentation time for " + ("3D volumes" if is_volume else "2D images"),
             fontsize=19, weight="bold", color=INK)
    revision = report["reference_revision"][:7]
    precision = "bfloat16" if report["parameters"].get("use_bfloat16") else "FP32"
    subtitle = (f"Reference: Cellpose 3.1.1.3 ({revision}) ; FP32 network ; channels [0, 0]"
                if kind == "cyto3" else
                f"Reference: Cellpose main {revision} ; {precision} network ; native scale (no resizing)")
    fig.text(.035, .905, subtitle, fontsize=11, color=REFERENCE)
    fig.legend([Line2D([0], [0], color=REFERENCE, linewidth=8),
                Line2D([0], [0], color=OPTIMIZED, linewidth=8)],
               ["Upstream reference", "Exact optimization"], loc="upper left",
               bbox_to_anchor=(.235, .865), frameon=False, ncol=2, fontsize=11)
    max_time = max(max(ref.max(), opt.max()) * 1000 for ref, opt in times)
    ratios = [ref / opt for ref, opt in times]
    ratio_min = min(v.min() for v in ratios)
    ratio_max = max(v.max() for v in ratios)
    axes[0].set_xlim(0, max_time * 1.55)
    axes[1].set_xlim(min(.9, ratio_min - .05), ratio_max + .32)
    axes[1].axvline(1, color=REFERENCE, linestyle="--", linewidth=1.2)
    summary = []
    for i, (row, (ref, opt)) in enumerate(zip(rows, times)):
        med_ref, med_opt = np.median(ref) * 1000, np.median(opt) * 1000
        delta = med_opt - med_ref
        for values, center, offset, color in [(ref * 1000, med_ref, -.18, REFERENCE),
                                              (opt * 1000, med_opt, .18, OPTIMIZED)]:
            axes[0].barh(y[i] + offset, center, height=.30, color=color,
                         xerr=range_error(center, values),
                         error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK})
            label = f"{center:.1f}"
            if offset > 0:
                label += f" ({abs(delta):.1f} ms {'saved' if delta < 0 else 'longer'})"
            axes[0].text(max(center, values.max()) + max_time * .018,
                         y[i] + offset, label, va="center", fontsize=9, color=INK)
        speedup = med_ref / med_opt
        axes[1].scatter(ratios[i], y[i] + np.linspace(-.07, .07, len(ref)),
                        s=12, color=OPTIMIZED, alpha=.3, zorder=3)
        axes[1].errorbar(speedup, y[i], xerr=range_error(speedup, ratios[i]),
                         fmt="o", color=OPTIMIZED, markersize=7, capsize=3,
                         linewidth=1.2, zorder=4)
        axes[1].text(.98, y[i], f"{speedup:.2f}×", ha="right", va="center",
                     transform=axes[1].get_yaxis_transform(), fontsize=11, color=INK)
        summary.append([row_label(row, kind), med_ref, med_opt, delta, speedup, len(ref)])
    axes[0].set_yticks(y, [row_label(row, kind) for row in rows], fontsize=11)
    axes[0].set_ylim(y[-1] + .6, -.6)
    axes[1].tick_params(labelleft=False)
    axes[0].set_xlabel("Total inference time (ms)", fontsize=12, color=INK)
    axes[1].set_xlabel("Speedup (reference / optimized)", fontsize=12, color=INK)
    for ax in axes:
        style_axis(ax)
    fig.text(.035, .105, "Bars/large markers: medians. Latency whiskers: repeat min–max. "
             "Speedup whiskers/dots: paired repeat range/samples.", fontsize=9, color=REFERENCE)
    counts = "/".join(str(n) for n in sorted({len(ref) for ref, _ in times}))
    fig.text(.035, .070, "Time differences compare medians. "
             f"{counts} synchronized repeats; batch size {report['parameters']['batch_size']}; "
             f"{hardware_label(report)}.",
             fontsize=9, color=REFERENCE)
    fig.text(.035, .035, "Public Cellpose test fixtures ; All candidate mask bytes match the reference ; "
             "Model loading and image decoding excluded", fontsize=9, color=REFERENCE)
    filename = "cyto3_inference" if kind == "cyto3" else ("sam_3d_inference" if is_volume else "sam_inference")
    export(fig, directory, filename)
    with (directory / f"{filename}_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["configuration", "reference_ms", "optimized_ms", "delta_ms", "speedup", "repeats"])
        writer.writerows(summary)


def mask_effect(report, volume_report, directory):
    rows = list(report["results"])
    is_volume = report["parameters"].get("do_3D", False)
    y = positions(rows, "sam")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8 if is_volume else 8), sharey=True)
    fig.subplots_adjust(left=.28 if is_volume else .25, right=.97,
                        top=.70 if is_volume else .79,
                        bottom=.27 if is_volume else .18, wspace=.24)
    fig.text(.035, .95, "Cellpose-SAM: " + ("3D volume" if is_volume else "2D images") + " timing breakdown",
             fontsize=19, weight="bold", color=INK)
    fig.text(.035, .905, "Separate timings show the effect of the shared dynamics optimization",
             fontsize=11, color=REFERENCE)
    fig.legend([Line2D([0], [0], color=REFERENCE, marker="o"),
                Line2D([0], [0], color=OPTIMIZED, marker="s")],
               ["Total inference", "Mask construction from cached flows"],
               loc="upper left", bbox_to_anchor=(.24, .865), frameon=False, ncol=2, fontsize=10)
    summary, all_ratios, all_savings = [], [], []
    for i, row in enumerate(rows):
        for stage, offset, color, marker in [("eval", -.12, REFERENCE, "o"),
                                              ("masks", .12, OPTIMIZED, "s")]:
            ref, opt = samples(row, stage)
            med_ref, med_opt = np.median(ref), np.median(opt)
            speedup = med_ref / med_opt
            ratios = ref / opt
            saved = (med_ref - med_opt) * 1000
            savings = (ref - opt) * 1000
            all_ratios.extend(ratios)
            all_savings.extend(savings)
            axes[0].errorbar(speedup, y[i] + offset, xerr=range_error(speedup, ratios),
                             fmt=marker, color=color, capsize=3, markersize=7, linewidth=1.1)
            axes[1].errorbar(saved, y[i] + offset, xerr=range_error(saved, savings),
                             fmt=marker, color=color, capsize=3, markersize=7, linewidth=1.1)
            summary.append([row_label(row, "sam"), stage, med_ref * 1000,
                            med_opt * 1000, saved, speedup, len(ref)])
    axes[0].axvline(1, linestyle="--", color=REFERENCE, linewidth=1.2)
    axes[1].axvline(0, linestyle="--", color=REFERENCE, linewidth=1.2)
    axes[0].set_xlim(min(.9, min(all_ratios) - .04), max(all_ratios) + .06)
    span = max(max(all_savings) - min(all_savings), 1.)
    axes[1].set_xlim(min(all_savings) - .1 * span, max(all_savings) + .1 * span)
    labels = [row_label(row, "sam") + f" (n={len(samples(row, 'eval')[0])})"
              for row in rows]
    axes[0].set_yticks(y, labels, fontsize=11)
    axes[0].set_ylim(y[-1] + .6, -.6)
    axes[1].tick_params(labelleft=False)
    axes[0].set_xlabel("Speedup (reference / optimized)", fontsize=12, color=INK)
    axes[1].set_xlabel("Time saved: reference − optimized (ms)", fontsize=12, color=INK)
    for ax in axes:
        style_axis(ax)
    fig.text(.035, .09, "Markers: ratios/differences of median times. "
             "Whiskers: observed paired repeat ranges, not confidence intervals.", fontsize=9, color=REFERENCE)
    precision = "bfloat16" if report["parameters"].get("use_bfloat16") else "FP32"
    fig.text(.035, .055, f"{precision} network; FP32 dynamics; {hardware_label(report)}. "
             "All mask bytes identical. GPU load varies; small total-inference changes need cautious interpretation.",
             fontsize=9, color=REFERENCE)
    filename = "sam_3d_mask_effect" if is_volume else "sam_mask_effect"
    export(fig, directory, filename)
    with (directory / f"{filename}_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["configuration", "stage", "reference_ms", "optimized_ms",
                         "saved_ms", "speedup", "repeats"])
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["cyto3", "sam"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--volume-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.labelcolor": INK})
    report = load_report(args.report)
    overview(report, args.kind, args.output_dir)
    if args.kind == "sam":
        volume = load_report(args.volume_report) if args.volume_report else None
        mask_effect(report, None, args.output_dir)
        if volume:
            overview(volume, "sam", args.output_dir)
            mask_effect(volume, None, args.output_dir)


if __name__ == "__main__":
    main()
