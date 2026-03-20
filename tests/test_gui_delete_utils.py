import numpy as np
import importlib.util
from pathlib import Path


_DELETE_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "cellpose/gui/delete_utils.py"
)
_DELETE_UTILS_SPEC = importlib.util.spec_from_file_location(
    "cellpose_gui_delete_utils", _DELETE_UTILS_PATH
)
delete_utils = importlib.util.module_from_spec(_DELETE_UTILS_SPEC)
_DELETE_UTILS_SPEC.loader.exec_module(delete_utils)


def _legacy_remove_state(cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids):
    cellpix = cellpix.copy()
    outpix = outpix.copy()
    ismanual = ismanual.copy()
    cellcolors = cellcolors.copy()
    zdraw = list(zdraw)

    for idx in remove_ids:
        cp = cellpix == idx
        op = outpix == idx
        cellpix[cp] = 0
        outpix[op] = 0
        cellpix[cellpix > idx] -= 1
        outpix[outpix > idx] -= 1
        ismanual = np.delete(ismanual, idx - 1)
        cellcolors = np.delete(cellcolors, [idx], axis=0)
        del zdraw[idx - 1]

    return cellpix, outpix, ismanual, cellcolors, zdraw


def _random_state(seed, nz=1, ly=64, lx=64, ncells=40):
    rng = np.random.default_rng(seed)
    dtype = np.uint16 if ncells < 2**16 - 1 else np.uint32

    cellpix = rng.integers(0, ncells + 1, size=(nz, ly, lx), dtype=dtype)
    force_idx = rng.choice(cellpix.size, size=ncells, replace=False)
    cellpix.flat[force_idx] = np.arange(1, ncells + 1, dtype=dtype)

    outline_mask = rng.random(cellpix.shape) < 0.2
    outpix = np.where(outline_mask, cellpix, 0).astype(dtype, copy=False)

    ismanual = rng.integers(0, 2, size=ncells, dtype=np.uint8).astype(bool)
    cellcolors = rng.integers(0, 256, size=(ncells + 1, 3), dtype=np.uint8)
    cellcolors[0] = np.array([255, 255, 255], dtype=np.uint8)

    zdraw = []
    for _ in range(ncells):
        nplanes = int(rng.integers(1, max(2, nz + 1)))
        zdraw.append(list(rng.integers(0, max(1, nz), size=nplanes)))

    return cellpix, outpix, ismanual, cellcolors, zdraw


def _assert_state_equal(expected, actual):
    exp_cellpix, exp_outpix, exp_ismanual, exp_cellcolors, exp_zdraw = expected
    got_cellpix, got_outpix, got_ismanual, got_cellcolors, got_zdraw = actual

    assert np.array_equal(exp_cellpix, got_cellpix)
    assert np.array_equal(exp_outpix, got_outpix)
    assert np.array_equal(exp_ismanual, got_ismanual)
    assert np.array_equal(exp_cellcolors, got_cellcolors)
    assert len(exp_zdraw) == len(got_zdraw)
    for z0, z1 in zip(exp_zdraw, got_zdraw):
        assert np.array_equal(np.asarray(z0), np.asarray(z1))


def test_batch_delete_reindex_matches_legacy_small_example():
    cellpix = np.array(
        [[[1, 1, 2, 2], [1, 3, 3, 2], [4, 4, 5, 5], [4, 0, 5, 5]]], dtype=np.uint16
    )
    outpix = np.array(
        [[[1, 0, 2, 0], [0, 3, 0, 2], [4, 0, 5, 0], [0, 0, 0, 5]]], dtype=np.uint16
    )
    ismanual = np.array([True, False, True, False, True])
    cellcolors = np.array(
        [[255, 255, 255], [10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0], [50, 0, 0]],
        dtype=np.uint8,
    )
    zdraw = [[0], [0], [0], [0], [0]]

    remove_ids = np.array([5, 3, 2], dtype=np.int64)

    expected = _legacy_remove_state(
        cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids
    )
    got = delete_utils.batch_delete_reindex(
        cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids
    )[:5]
    _assert_state_equal(expected, got)


def test_batch_delete_reindex_matches_legacy_random_2d():
    for seed in range(20):
        cellpix, outpix, ismanual, cellcolors, zdraw = _random_state(seed, nz=1)
        rng = np.random.default_rng(seed + 1000)
        ncells = len(cellcolors) - 1
        remove_n = int(rng.integers(1, ncells + 1))
        remove_ids = rng.choice(np.arange(1, ncells + 1), size=remove_n, replace=False)
        remove_ids = delete_utils.normalize_remove_ids(remove_ids, ncells)

        expected = _legacy_remove_state(
            cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids
        )
        got = delete_utils.batch_delete_reindex(
            cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids
        )[:5]
        _assert_state_equal(expected, got)


def test_batch_delete_reindex_matches_legacy_random_3d():
    for seed in range(12):
        cellpix, outpix, ismanual, cellcolors, zdraw = _random_state(seed + 100, nz=4)
        rng = np.random.default_rng(seed + 2000)
        ncells = len(cellcolors) - 1
        remove_n = int(rng.integers(1, ncells + 1))
        remove_ids = rng.choice(np.arange(1, ncells + 1), size=remove_n, replace=False)
        remove_ids = delete_utils.normalize_remove_ids(remove_ids, ncells)

        expected = _legacy_remove_state(
            cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids
        )
        got = delete_utils.batch_delete_reindex(
            cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids
        )[:5]
        _assert_state_equal(expected, got)


def test_batch_delete_reindex_noop_invalid_ids():
    cellpix, outpix, ismanual, cellcolors, zdraw = _random_state(999, nz=1)
    ncells = len(cellcolors) - 1
    remove_ids = np.array([0, -1, ncells + 10], dtype=np.int64)

    got = delete_utils.batch_delete_reindex(
        cellpix, outpix, ismanual, cellcolors, zdraw, remove_ids
    )
    got_state = got[:5]
    out_ids = got[5]
    remove_mask = got[6]

    _assert_state_equal((cellpix, outpix, ismanual, cellcolors, zdraw), got_state)
    assert out_ids.size == 0
    assert not remove_mask.any()
