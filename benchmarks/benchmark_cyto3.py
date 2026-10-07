"""Benchmark exact cyto3 inference against an unmodified git revision.

Run from the checkout: python benchmarks/benchmark_cyto3.py --help.
Input decoding and model loading are excluded; eval (including masks and returned
flows/styles) is timed. No private images or weights are embedded in the report.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cellpose import dynamics, io, models


def reference_functions(revision):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:cellpose/dynamics.py"], cwd=ROOT, text=True)
    names = {"steps_interp", "get_masks_torch"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise ValueError("Reference must expose steps_interp and get_masks_torch")
    namespace = dict(vars(dynamics))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<git reference>", "exec"), namespace)
    return {name: namespace[name] for name in names}


@contextmanager
def implementation(functions):
    saved = {name: getattr(dynamics, name) for name in functions}
    try:
        for name, fn in functions.items():
            setattr(dynamics, name, fn)
        yield
    finally:
        for name, fn in saved.items():
            setattr(dynamics, name, fn)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def arrays_exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def comparison(ref, pred):
    same_shape = ref.shape == pred.shape
    values_equal = bool(same_shape and np.array_equal(ref, pred))
    partition_equal = False
    if same_shape:
        if values_equal:
            partition_equal = True
        else:
            # Cellpose label IDs fit uint32. Encoding pairs avoids sorting a
            # structured array of every pixel for this untimed diagnostic.
            pairs = np.unique((ref.astype(np.uint64).ravel() << np.uint64(32)) |
                              pred.astype(np.uint64).ravel())
            ref_ids, pred_ids = pairs >> np.uint64(32), pairs & np.uint64(0xffffffff)
            partition_equal = (len(pairs) == len(np.unique(ref_ids)) == len(np.unique(pred_ids))
                               and np.all((ref_ids == 0) == (pred_ids == 0)))
    return {"exact": bool(arrays_exact(ref, pred)),
            "partition_equal_ignoring_label_ids": bool(partition_equal),
            "foreground_changed_pixels": int(np.count_nonzero((ref > 0) != (pred > 0)))
            if same_shape else None,
            "reference_dtype": str(ref.dtype), "candidate_dtype": str(pred.dtype),
            "changed_pixels": int(np.count_nonzero(ref != pred)) if same_shape else None,
            "reference_sha256": digest(ref), "candidate_sha256": digest(pred),
            "reference_cells": int(ref.max()), "candidate_cells": int(pred.max())}


def timed(fn, device):
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    elapsed = time.perf_counter() - start
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return result, elapsed, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="v3.1.1.3")
    parser.add_argument("--model", default="cyto3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diameters", type=float, nargs="+", default=[30., 15.])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must both be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    reference = reference_functions(args.reference)
    candidate = {name: getattr(dynamics, name) for name in reference}
    model = models.CellposeModel(gpu=device.type == "cuda", model_type=args.model,
                                pretrained_model=args.model, device=device)
    images = [(p.name, io.imread(str(p))) for p in args.images]
    if any(image is None for _, image in images):
        raise ValueError("Could not decode an input image")
    report = {"reference_revision": subprocess.check_output(
        ["git", "rev-parse", args.reference], cwd=ROOT, text=True).strip(),
        "candidate_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate_diff_sha256": hashlib.sha256(subprocess.check_output(
        ["git", "diff", "--", "cellpose/dynamics.py"], cwd=ROOT)).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda" else None, "threads": args.threads,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader"], text=True).strip() if device.type == "cuda" else None,
        "parameters": {"batch_size": args.batch_size, "channels": [0, 0],
        "flow_threshold": .4, "min_size": 15, "augment": False,
        "tile_overlap": .1, "resample": True, "warmup": args.warmup,
        "repeats": args.repeats}, "model_sha256": hashlib.sha256(
        Path(model.pretrained_model).read_bytes()).hexdigest(), "results": []}
    all_exact = True
    for diameter in args.diameters:
        for name, image in images:
            def evaluate(functions):
                with implementation(functions):
                    return model.eval(image, channels=[0, 0], diameter=diameter,
                                      batch_size=args.batch_size, augment=False,
                                      flow_threshold=.4, min_size=15, tile_overlap=.1)
            for _ in range(args.warmup):
                evaluate(reference)
                evaluate(candidate)
            baseline = evaluate(reference)
            row = {"image": name, "shape": list(image.shape), "input_sha256": digest(image),
                   "diameter": diameter, "reference_seconds": [], "candidate_seconds": [],
                   "reference_stable": True, "masks_exact_all_repeats": True,
                   "flows_exact_all_repeats": True, "styles_exact_all_repeats": True}
            for repeat in range(args.repeats):
                order = [("reference", reference), ("candidate", candidate)]
                if repeat % 2:
                    order.reverse()
                for label, functions in order:
                    output, elapsed, memory = timed(lambda: evaluate(functions), device)
                    row[f"{label}_seconds"].append(elapsed)
                    row[f"{label}_peak_allocated_bytes"] = max(
                        row.get(f"{label}_peak_allocated_bytes", 0) or 0, memory or 0)
                    cmp = comparison(baseline[0], output[0])
                    if label == "reference":
                        row["reference_stable"] &= cmp["exact"]
                    else:
                        row["masks_exact_all_repeats"] &= cmp["exact"]
                        row["mask_comparison"] = cmp
                        row["flows_exact_all_repeats"] &= all(
                            arrays_exact(a, b) for a, b in zip(baseline[1], output[1]))
                        row["styles_exact_all_repeats"] &= arrays_exact(baseline[2], output[2])
            row["speedup"] = statistics.median(row["reference_seconds"]) / statistics.median(row["candidate_seconds"])
            all_exact &= (row["reference_stable"] and row["masks_exact_all_repeats"] and
                          row["flows_exact_all_repeats"] and row["styles_exact_all_repeats"])
            report["results"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name} diameter={diameter:g}: {row['speedup']:.2f}x, masks exact={row['masks_exact_all_repeats']}", flush=True)
    report["all_exact"] = bool(all_exact)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all_exact:
        raise SystemExit("Exact equivalence check failed; see report")


if __name__ == "__main__":
    main()
