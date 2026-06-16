# %%
"""Cellpose vs TensorRT benchmarking

Compare full pipeline (masks/flows/styles) between Torch and TRT
using the same inputs, report IoU and percent error (sMAPE),
and time both implementations.

Example usage:
python 'trt_benchmark.py' \
    --image=/data/registered/reg-0076.tif \
    --pretrained cpsam \
    --engine cpsam.plan \
    --batch-size=4

Example output:
    Loaded tile: (2, 512, 512) uint16
    Engine path: /home/chaichontat/cellpose/scripts/builds/cpsam_b4_sm120_bf16.plan
    Using CUDA device: cuda:0 | NVIDIA GeForce RTX 5090

    [TEST] Full pipeline parity
    masks: torch=(512, 512) trt=(512, 512)  IoU=0.9986
    flow[0]: shape=(512, 512, 3) | sMAPE=2.257%  MAE=0.176858
    flow[1]: shape=(2, 512, 512) | sMAPE=27.623%  MAE=0.0060048
    flow[2]: shape=(512, 512) | sMAPE=0.816%  MAE=0.0170394

    [TIMING] Full pipeline eval(tile3)
    Torch eval: 222.155 ms/iter (avg over 5, warmup=1)
    TRT eval: 138.330 ms/iter (avg over 5, warmup=1)
    Speedup vs Torch: x1.61

    [TIMING] Net-only forward (Nx3x256x256)
    Torch net: 15.930 ms/iter (CUDA events, iters=50, warmup=10)
    TRT net  : 7.110 ms/iter (CUDA events, iters=50, warmup=10)
    Speedup (net-only): x2.24

    [TEST] IoU parity on first 20 images from: /data/registered
    1/20 processed... IoU=0.9994
               ⋮
    20/20 processed... IoU=0.9991
    IoU range: min=0.9987  median=0.9991  max=0.9996  (N=20)
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import torch

from cellpose import models
from cellpose.contrib.cellposetrt import CellposeModelTRT

TILE_SLICE = np.s_[5, :, :512, :512]


def parse_args():
    ap = argparse.ArgumentParser(description="Cellpose vs TensorRT benchmarking")
    ap.add_argument(
        "--image", type=Path, required=True, help="Path to a test image (TIF)"
    )
    ap.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path/name of pretrained Cellpose model",
    )
    ap.add_argument(
        "--engine", type=Path, required=True, help="TensorRT engine (.plan) path"
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of folder images to test IoU on",
    )
    ap.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Folder of images for IoU test; defaults to image's parent",
    )
    ap.add_argument("--batch-size", type=int, default=4, help="Eval/engine batch size")
    ap.add_argument(
        "--save-masks",
        type=Path,
        default=None,
        help="Optional output path (directory or .tif file) to save stacked masks from the IoU parity test",
    )
    return ap.parse_args()


def print_smape(name: str, ref, tst) -> None:
    r = torch.as_tensor(ref).float().flatten()
    t = torch.as_tensor(tst).float().flatten()
    diff = (t - r).abs()
    mae = float(diff.mean())
    smape = float((2.0 * diff / (r.abs() + t.abs() + 1e-12)).mean() * 100.0)
    print(
        f"{name}: shape={tuple(torch.as_tensor(ref).shape)} | sMAPE={smape:.3f}%  MAE={mae:.6g}"
    )


def time_op(
    name: str,
    fn: Callable,
    *,
    warmup: int = 1,
    iters: int = 5,
) -> float:
    # Warmup
    for _ in range(warmup):
        _ = fn()

    # Run
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()

    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    ms = dt * 1000.0
    print(f"{name}: {ms:.3f} ms/iter (avg over {iters}, warmup={warmup})")
    return ms


def time_op_cuda(
    name: str,
    fn,
    *,
    warmup: int = 10,
    iters: int = 50,
) -> float:
    """GPU kernel timing using CUDA events (net-only).

    Records elapsed time on the current CUDA stream across `iters` calls. Does
    not include Python/host sync beyond the final event synchronize.
    """
    # Warmup to stabilize autotuning/caches
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"{name}: {ms:.3f} ms/iter (CUDA events, iters={iters}, warmup={warmup})")
    return ms


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(1, float(union))


args = parse_args()

save_masks_target: Path | None = None
if args.save_masks is not None:
    save_masks_target = Path(args.save_masks)
    save_masks_target.parent.mkdir(parents=True, exist_ok=True)

eval_kwargs = dict(
    batch_size=args.batch_size,
    flow_threshold=0,
    compute_masks=True,
)

tile = tifffile.imread(args.image)[TILE_SLICE]
print("Loaded tile:", tile.shape, tile.dtype)
print(f"Engine path: {args.engine}")

# ---- Build models ----
device = torch.device("cuda:0")
print(f"Using CUDA device: {device} | {torch.cuda.get_device_name(device)}")

base = models.CellposeModel(gpu=True, device=device, pretrained_model=args.pretrained)
trt_model = CellposeModelTRT(
    gpu=True,
    device=device,
    pretrained_model=args.pretrained,
    engine_path=str(args.engine),
)

with torch.inference_mode():
    base_out = base.eval(tile, **eval_kwargs)
    trt_out = trt_model.eval(tile, **eval_kwargs)

print("\n[TEST] Full pipeline parity")
masks_pt, masks_trt = base_out[0], trt_out[0]
print(
    f"  masks: torch={masks_pt.shape} trt={masks_trt.shape}  IoU={iou_binary(masks_pt != 0, masks_trt != 0):.4f}"
)

flows_pt = base_out[1]
flows_trt = trt_out[1]
for k, (fpt, ftrt) in enumerate(zip(flows_pt, flows_trt)):
    print_smape(f"  flow[{k}]", fpt, ftrt)

# Timing (full pipeline):
with torch.inference_mode():
    print("\n[TIMING] Full pipeline eval(tile3)")
    ms_base = time_op("  Torch eval", lambda: base.eval(tile, **eval_kwargs))
    ms_trt = time_op(
        "  TRT eval",
        lambda: models.CellposeModel.eval(trt_model, tile, **eval_kwargs),
    )

spd = ms_base / ms_trt
print(f"  Speedup vs Torch: x{spd:.2f}")

# Net-only timing on representative Nx3x256x256 batch (CUDA events)
with torch.inference_mode():
    print(f"\n[TIMING] Net-only forward ({args.batch_size}x3x256x256)")
    Xb = torch.randn(args.batch_size, 3, 256, 256, device=device, dtype=torch.bfloat16)
    ms_torch_net = time_op_cuda("  Torch net", lambda: base.net(Xb))
    ms_trt_net = time_op_cuda("  TRT net  ", lambda: trt_model.net(Xb))
    if ms_trt_net > 0:
        print(f"  Speedup (net-only): x{ms_torch_net / ms_trt_net:.2f}")

# ---- TEST: Folder IoU on first N images (Torch vs TRT masks) ----
folder = args.folder or args.image.parent
files = [p for p in sorted(folder.glob("*.tif"))]
sub = files[: args.n_samples]

print(f"\n[TEST] IoU parity on first {len(sub)} images from: {folder}")
ious: list[float] = []
saved_masks: list[np.ndarray] = []
for idx, f in enumerate(sub):
    try:
        arr = tifffile.imread(f)[TILE_SLICE]
        with torch.inference_mode():
            out_t = base.eval(arr, **eval_kwargs)
            out_r = trt_model.eval(arr, **eval_kwargs)
            if not np.any(out_t[0]) or not np.any(out_r[0]):
                print(
                    f"  [warn] skipping {f.name}: empty masks from at least one of the models"
                )
                continue

            m_t = out_t[0]
            m_r = out_r[0]
        iou = iou_binary(m_t != 0, m_r != 0)
        ious.append(iou)
        if save_masks_target is not None:
            saved_masks.append(np.stack((m_t, m_r), axis=0))

        print(f"  {idx + 1}/{len(sub)} processed... IoU={iou:.4f}")
    except Exception as e:
        print(f"  [warn] skipping {f.name}: {e}")

a = np.array(ious, dtype=float)
print(
    f"  IoU range: min={a.min():.4f}  median={np.median(a):.4f}  max={a.max():.4f}  (N={len(a)})"
)

if save_masks_target is not None:
    stacked = np.stack(saved_masks, axis=0)
    tifffile.imwrite(
        save_masks_target,
        stacked,
        metadata={"axes": "TCYX"},
        compression="zstd",
    )
    print(f"Saved masks to {save_masks_target} with shape {stacked.shape}")
