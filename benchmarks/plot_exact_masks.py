"""Reproduce cyto3 masks and plot visual, IoU and byte-equality comparisons.

Requires the public fixture images and cyto3 weights used by benchmark_cyto3.py.
Figures compare inference outputs, rather than accuracy against manual labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, hsv_to_rgb
from matplotlib.patches import Patch
import numpy as np
import torch

from benchmark_cyto3 import (ROOT, reference_functions, implementation,
                              arrays_exact, comparison, digest, models, io, dynamics)
from plot_benchmarks import export, IMAGE_NAMES, INK, REFERENCE
from cellpose import metrics, utils


def display_image(image):
    image = image.astype(np.float32)
    lower, upper = np.percentile(image, [1, 99], axis=(0, 1))
    image = np.clip((image - lower) / np.maximum(upper - lower, 1e-6), 0, 1)
    return np.repeat(image[..., None], 3, axis=-1) if image.ndim == 2 else image[..., :3]


def overlay(image, masks):
    ids = np.arange(int(masks.max()) + 1)
    colors = hsv_to_rgb(np.stack([(ids * .61803398875) % 1,
                                  np.full(len(ids), .85), np.ones(len(ids))], axis=-1))
    result = image.copy()
    foreground = masks > 0
    result[foreground] = .75 * result[foreground] + .25 * colors[masks[foreground]]
    outlines = utils.masks_to_outlines(masks)
    result[outlines] = colors[masks[outlines]]
    return result


def visualize(examples, directory):
    fig, axes = plt.subplots(len(examples), 4, figsize=(15, 3.6 * len(examples) + 1.7),
                             squeeze=False)
    fig.subplots_adjust(left=.04, right=.98, top=.86, bottom=.10, wspace=.09, hspace=.20)
    fig.text(.035, .965, "Cellpose 3 (cyto3): optimized inference preserves mask labels",
             fontsize=20, weight="bold", color=INK)
    fig.text(.035, .925, "2D images; cell diameter setting: 30 pixels; FP32; flow QC and iteration settings retained",
             fontsize=11, color=REFERENCE)
    titles = ["Input image", "Upstream cell masks", "Optimized cell masks", "Label difference map"]
    for i, (row, image, ref, pred) in enumerate(examples):
        display = display_image(image)
        difference = ref != pred
        for ax, view in zip(axes[i, :3], [display, overlay(display, ref), overlay(display, pred)]):
            ax.imshow(view)
        axes[i, 3].imshow(difference, cmap=ListedColormap(["#f1f5f9", "#dc2626"]), vmin=0, vmax=1)
        axes[i, 3].text(.5, .5, f"{np.count_nonzero(difference)} changed label pixels\n"
                       f"{row['reference_cells']} cells in each output\nBitwise identical: yes\n\n"
                       f"{row['reference_ms']:.1f} ms → {row['candidate_ms']:.1f} ms\n"
                       f"{row['speedup']:.2f}x faster\n{row['saved_ms']:.1f} ms saved",
                       transform=axes[i, 3].transAxes, ha="center", va="center",
                       fontsize=12, color=INK)
        label = f"{IMAGE_NAMES.get(row['image'], row['image'])}\n{ref.shape[0]} x {ref.shape[1]} pixels"
        axes[i, 0].set_ylabel(label, fontsize=10, color=INK)
        for j, ax in enumerate(axes[i]):
            if i == 0:
                ax.set_title(titles[j], fontsize=12, pad=14)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.legend([Patch(facecolor="#f1f5f9"), Patch(facecolor="#dc2626")],
               ["Same label", "Different label"], loc="lower right",
               bbox_to_anchor=(.98, .045), frameon=False, ncol=2, fontsize=10)
    fig.text(.035, .055, "Matching colors denote the same cell label. Equality checks include every pixel, label ID, array shape and dtype.",
             fontsize=10, color=REFERENCE)
    fig.text(.035, .025, "Times are medians from the five-repeat speed benchmark for each image; mask reproduction is a separate untimed check.",
             fontsize=10, color=REFERENCE)
    export(fig, directory, "cyto3_identical_masks")


def agreement_plot(rows, directory):
    fig, axes = plt.subplots(1, 3, figsize=(15, max(7, .75 * len(rows) + 2.5)), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1, 1.4]})
    fig.subplots_adjust(left=.20, right=.98, top=.80, bottom=.20, wspace=.22)
    fig.text(.035, .955, "Cellpose 3 (cyto3): identical masks across benchmark cases",
             fontsize=19, weight="bold", color=INK)
    fig.text(.035, .90, "Recomputed outputs match each other and the published reference/candidate mask SHA256 hashes",
             fontsize=10, color=REFERENCE)
    labels = []
    for i, row in enumerate(rows):
        labels.append(f"{IMAGE_NAMES.get(row['image'], row['image'])}\nCell diameter: {row['diameter']:g} pixels")
        for ax, key, color in zip(axes[:2], ["object_iou", "foreground_iou"], ["#0f766e", "#2563eb"]):
            ax.scatter(row[key], i, color=color, s=55)
            ax.annotate(f"{row[key]:.4f}", (row[key], i), xytext=(-12, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=11)
        axes[2].text(.03, i, f"Identical bytes: yes\nChanged label pixels: {row['changed_pixels']}",
                     va="center", transform=axes[2].get_yaxis_transform(), fontsize=11, color=INK)
    for ax in axes[:2]:
        ax.set_xlim(0, 1.06)
        ax.set_xticks([0, .5, 1])
        ax.set_xlabel("IoU agreement with upstream masks", fontsize=10)
        ax.grid(axis="x", color="#e2e8f0")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(rows)), labels, fontsize=10)
    axes[0].set_ylim(len(rows) - .5, -.5)
    axes[0].set_title("Mean matched cell IoU", fontsize=12)
    axes[1].set_title("Foreground IoU", fontsize=12)
    axes[2].set_title("Byte equality and label differences", fontsize=12)
    axes[2].set_axis_off()
    fig.text(.035, .11, "IoU is agreement with upstream inference, not ground-truth accuracy. Matching mask hashes verify shape/dtype/label-ID-preserving equality.",
             fontsize=9, color=REFERENCE)
    fig.text(.035, .06, "One additional inference pair per case is shown here. The speed benchmark independently checks all five candidate repeats per case.",
             fontsize=9, color=REFERENCE)
    export(fig, directory, "cyto3_mask_iou")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "benchmarks/results/public-rtx-pro-6000.json")
    parser.add_argument("--additional-report", type=Path, nargs="*", default=[],
                        help="Include supplemental reports with the same weights/settings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/figures")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    benchmark = json.loads(args.report.read_text())
    reports = [benchmark] + [json.loads(path.read_text()) for path in args.additional_report]
    for report in reports:
        for key in ["reference_revision", "model_sha256", "parameters", "matmul_tf32",
                    "cudnn_tf32", "cudnn_benchmark", "cudnn_deterministic", "threads"]:
            if report[key] != benchmark[key]:
                raise ValueError(f"Reports must share {key}")
        if not report["all_exact"]:
            raise ValueError("Each report must pass all exactness checks")
    if not benchmark["all_exact"]:
        raise ValueError("The benchmark report must pass all exactness checks")
    torch.set_num_threads(benchmark["threads"])
    torch.backends.cuda.matmul.allow_tf32 = benchmark["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = benchmark["cudnn_tf32"]
    torch.backends.cudnn.benchmark = benchmark["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = benchmark["cudnn_deterministic"]
    device = torch.device(args.device)
    reference = reference_functions(benchmark["reference_revision"])
    candidate = {name: getattr(dynamics, name) for name in reference}
    model = models.CellposeModel(gpu=device.type == "cuda", model_type="cyto3",
                                pretrained_model="cyto3", device=device)
    if hashlib.sha256(Path(model.pretrained_model).read_bytes()).hexdigest() != benchmark["model_sha256"]:
        raise AssertionError("Model weights do not match the published benchmark")
    # Match by input content so supplemental image filenames need not be public.
    images = {}
    for path in args.images:
        image = io.imread(str(path))
        images[digest(image)] = image
    rows, examples = [], []
    params = benchmark["parameters"]
    for case in [case for report in reports for case in report["results"]]:
        image = images[case["input_sha256"]]
        if digest(image) != case["input_sha256"]:
            raise AssertionError("Input does not match the published benchmark")
        def evaluate(functions):
            with implementation(functions):
                return model.eval(image, channels=params["channels"], diameter=case["diameter"],
                    batch_size=params["batch_size"], augment=params["augment"],
                    flow_threshold=params["flow_threshold"], min_size=params["min_size"],
                    tile_overlap=params["tile_overlap"])
        ref, pred = evaluate(reference), evaluate(candidate)
        if not (arrays_exact(ref[0], pred[0]) and
                all(arrays_exact(a, b) for a, b in zip(ref[1], pred[1])) and arrays_exact(ref[2], pred[2])):
            raise AssertionError("Reference and candidate outputs differ")
        row = comparison(ref[0], pred[0])
        expected = case["mask_comparison"]
        if any(row[key] != expected[key] for key in ["reference_sha256", "candidate_sha256"]):
            raise AssertionError("Recomputed masks do not match the published mask hashes")
        ious = metrics.mask_ious(ref[0], pred[0])[0] if ref[0].max() else np.ones(1)
        union = np.count_nonzero((ref[0] > 0) | (pred[0] > 0))
        row.update(image=case["image"], diameter=case["diameter"], shape=list(ref[0].shape),
                   object_iou=float(ious.mean()), foreground_iou=float(
                       np.count_nonzero((ref[0] > 0) & (pred[0] > 0)) / union) if union else 1.)
        row["reference_ms"] = float(np.median(case["reference_seconds"]) * 1000)
        row["candidate_ms"] = float(np.median(case["candidate_seconds"]) * 1000)
        row["saved_ms"] = row["reference_ms"] - row["candidate_ms"]
        row["speedup"] = row["reference_ms"] / row["candidate_ms"]
        rows.append(row)
        if case["diameter"] == 30:
            examples.append((row, image, ref[0], pred[0]))
        print(f"{case['image']} diameter={case['diameter']:g}: exact bytes, IoU={row['object_iou']:.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    visualize(examples, args.output_dir)
    agreement_plot(rows, args.output_dir)
    proof = {"reference_revision": benchmark["reference_revision"], "device": str(device),
             "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
             "model_sha256": benchmark["model_sha256"], "parameters": params,
             "all_exact": True, "matches_published_hashes": True, "results": rows}
    (args.output_dir / "cyto3_mask_equality.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
