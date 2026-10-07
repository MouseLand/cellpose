"""Regression checks for coordinate arithmetic and seed overwrite behavior.

These also run without pytest: python -m unittest discover -s tests
    -p test_exact_dynamics.py
"""
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from cellpose import dynamics


def coordinate_reference(dP, inds, niter, device):
    """Original scalar-coordinate Euler update, including FP32 operation order."""
    ndim = len(inds)
    shape = np.array(dP.shape[1:])[::-1].astype(float) - 1
    pt = torch.zeros((*[1]*ndim, len(inds[0]), ndim), device=device)
    im = torch.zeros((1, ndim, *dP.shape[1:]), device=device)
    for n in range(ndim):
        pt[..., ndim - n - 1] = torch.as_tensor(inds[n], device=device).float()
        im[0, ndim - n - 1] = torch.as_tensor(dP[n], device=device)
    for k in range(ndim):
        im[:, k] *= 2. / shape[k]
        pt[..., k] /= shape[k]
    pt *= 2
    pt -= 1
    for _ in range(niter):
        delta = F.grid_sample(im, pt, align_corners=False)
        for k in range(ndim):
            pt[..., k] = torch.clamp(pt[..., k] + delta[:, k], -1., 1.)
    pt += 1
    pt *= .5
    for k in range(ndim):
        pt[..., k] *= shape[k]
    return pt.reshape(-1, ndim)[:, list(reversed(range(ndim)))].T


class ExactDynamicsTests(unittest.TestCase):
    def devices(self):
        return [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])

    def test_coordinates(self):
        rng = np.random.default_rng(61)
        for device in self.devices():
            for shape in ((17, 23), (7, 13, 19)):
                # 2D CPU uses the unchanged numba implementation.
                if device.type == "cpu" and len(shape) == 2:
                    continue
                for npoints in (1, 151):
                    for niter in (0, 1, 200):
                        for scale in (0., 1., 50.):
                            with self.subTest(device=device, shape=shape, points=npoints,
                                              iterations=niter, scale=scale):
                                flow = (rng.standard_normal((len(shape), *shape)) * scale).astype(np.float32)
                                inds = tuple(rng.integers(0, size, npoints) for size in shape)
                                expected = coordinate_reference(flow, inds, niter, device)
                                actual = dynamics.steps_interp(flow, inds, niter, device)
                                self.assertEqual(expected.shape, actual.shape)
                                self.assertEqual(expected.dtype, actual.dtype)
                                self.assertEqual(expected.cpu().numpy().tobytes(),
                                                 actual.cpu().numpy().tobytes())

    def mask(self, endpoints, counts, shape, device, fraction=.4):
        points = np.concatenate([np.repeat(np.array(p)[:, None], n, axis=1)
                                 for p, n in zip(endpoints, counts)], axis=1)
        inds = np.unravel_index(np.arange(points.shape[1]), shape)
        mask = dynamics.get_masks_torch(torch.as_tensor(points, device=device).int(),
                                       inds, shape, max_size_fraction=fraction)
        return mask, inds

    def test_seeds_at_image_boundaries(self):
        for device in self.devices():
            for shape in ((32, 40), (9, 12, 15)):
                with self.subTest(device=device, shape=shape):
                    endpoints = [tuple(0 for _ in shape), tuple(size-1 for size in shape)]
                    mask, inds = self.mask(endpoints, [11, 15], shape, device)
                    expected = np.zeros(shape, dtype=np.uint16)
                    expected[inds] = np.r_[np.ones(11), np.full(15, 2)].astype(np.uint16)
                    np.testing.assert_array_equal(mask, expected)
                    self.assertEqual(mask.dtype, expected.dtype)

    def test_equal_height_overlapping_seeds(self):
        for device in self.devices():
            for shape in ((32, 40), (9, 12, 15)):
                with self.subTest(device=device, shape=shape):
                    first = tuple(size//2 for size in shape)
                    second = (*first[:-1], first[-1] + 1)
                    mask, inds = self.mask([first, second], [12, 12], shape, device)
                    expected = np.zeros(shape, dtype=np.uint16)
                    expected[inds] = 1  # later seed overwrites the connected plateau
                    np.testing.assert_array_equal(mask, expected)

    def test_seed_threshold_and_oversized_removal(self):
        for device in self.devices():
            for counts, fraction in (([10], .4), ([20], .001)):
                with self.subTest(device=device, counts=counts, fraction=fraction):
                    mask, _ = self.mask([(8, 8)], counts, (32, 40), device, fraction)
                    self.assertFalse(mask.any())
                    self.assertEqual(mask.dtype, np.uint16)

    def test_without_scatter_reduce(self):
        with patch.object(torch.Tensor, "scatter_reduce_", None):
            self.test_equal_height_overlapping_seeds()
            self.test_seeds_at_image_boundaries()


if __name__ == "__main__":
    unittest.main()
