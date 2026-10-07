"""Differential mask construction audit against a git reference (no images needed)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_cyto3 import dynamics, reference_functions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="v3.1.1.3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = reference_functions(args.reference)["get_masks_torch"]
    rng = np.random.default_rng(87)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    cases = 0
    for device in devices:
        for shape in [(64, 80), (20, 24, 28)]:
            for case in range(20):
                locations = np.stack([rng.integers(0, size, 25) for size in shape])
                counts = rng.integers(3, 30, 25)
                if case % 2 == 0:
                    counts[:] = 12  # tied seeds and overlapping windows
                locations[:, 0] = 0
                locations[:, 1] = np.array(shape) - 1
                points = np.repeat(locations, counts, axis=1)
                inds = np.unravel_index(np.arange(points.shape[1]), shape)
                for fraction in [0., .01, .4, 1.]:
                    pt = torch.as_tensor(points, device=device).int()
                    expected = reference(pt.clone(), inds, shape, max_size_fraction=fraction)
                    actual = dynamics.get_masks_torch(pt.clone(), inds, shape,
                                                      max_size_fraction=fraction)
                    if expected.dtype != actual.dtype or expected.tobytes() != actual.tobytes():
                        raise AssertionError((str(device), shape, case, fraction))
                    cases += 1
    report = {"reference": args.reference, "random_seed": 87, "cases": cases,
              "devices": [str(device) for device in devices], "all_exact": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{cases} exact mask comparisons passed", flush=True)


if __name__ == "__main__":
    main()
