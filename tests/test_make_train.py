import builtins
import sys

import numpy as np
import pytest
import tifffile

from cellpose.gui import make_train


@pytest.fixture
def volume(tmp_path):
    # Each channel records one voxel coordinate, independently of crop sampling.
    image = np.moveaxis(np.indices((7, 8, 9), dtype=np.uint16), 0, -1)
    image_path = tmp_path / "volume.tif"
    tifffile.imwrite(image_path, image, photometric="rgb")
    masks = np.zeros(image.shape[:3], dtype=np.uint16)
    # Same-valued voxels are disconnected; unlike labels can touch diagonally.
    masks[::2, ::2, ::2] = 7
    masks[1::4, 1::4, 1::4] = 42
    return image_path, masks


def run_make_train(monkeypatch, image_path, seg_masks=True):
    argv = [
        "make_train", "--image_path",
        str(image_path), "--channel_axis", "3", "--z_axis", "0", "--nimg_per_tif", "3",
        "--crop_size", "5"
    ]
    if seg_masks:
        argv.append("--seg_masks")
    monkeypatch.setattr(sys, "argv", argv)
    # main seeds sampling with 0; don't leak that change into other tests.
    random_state = np.random.get_state()
    try:
        make_train.main()
    finally:
        np.random.set_state(random_state)


def read_crops(image_path):
    train_dir = image_path.parent / "train"
    assert len(list(train_dir.glob("*.tif"))) == 9
    for plane, slice_axis, row_axis, col_axis in [("YX", 0, 1, 2), ("ZY", 2, 0, 1),
                                                  ("ZX", 1, 0, 2)]:
        paths = sorted(train_dir.glob(f"volume_{plane}_*.tif"))
        assert len(paths) == 3
        for path in paths:
            image = tifffile.imread(path)
            assert image.shape == (5, 5, 3)
            origin = image[0, 0]
            np.testing.assert_array_equal(image[..., slice_axis], origin[slice_axis])
            np.testing.assert_array_equal(
                image[..., row_axis],
                np.broadcast_to(origin[row_axis] + np.arange(5)[:, None], (5, 5)))
            np.testing.assert_array_equal(
                image[..., col_axis],
                np.broadcast_to(origin[col_axis] + np.arange(5), (5, 5)))
            yield path, tuple(np.moveaxis(image, -1, 0))


@pytest.mark.parametrize("empty", [False, True], ids=["disconnected", "empty"])
@pytest.mark.parametrize("with_skimage", [True, False], ids=["skimage", "no-skimage"])
def test_make_train_saves_paired_masks(volume, monkeypatch, capsys, empty,
                                       with_skimage):
    image_path, masks = volume
    if empty:
        masks.fill(0)
    np.save(image_path.with_name("volume_seg.npy"), {"masks": masks})
    if with_skimage:
        pytest.importorskip("skimage.measure")
    else:
        original_import = builtins.__import__

        def without_label(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "skimage.measure" and "label" in fromlist:
                raise ImportError("scikit-image unavailable for this test")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", without_label)

    run_make_train(monkeypatch, image_path)
    crops = list(read_crops(image_path))
    assert len(list((image_path.parent / "train").glob("*_seg.npy"))) == len(crops)
    foreground_count = 0
    for image_file, coordinates in crops:
        # Only load trusted files generated inside this test's temporary directory.
        saved = np.load(image_file.with_name(image_file.stem + "_seg.npy"),
                        allow_pickle=True).item()
        actual = saved["masks"]
        expected = masks[coordinates]
        assert saved["filename"] == str(image_file)
        assert actual.shape == (5, 5)
        np.testing.assert_array_equal(actual > 0, expected > 0)
        foreground_count += np.count_nonzero(expected)
        if with_skimage:
            # Every foreground voxel is a separate component in these slices.
            labels = np.sort(actual[actual > 0])
            np.testing.assert_array_equal(labels, np.arange(1, len(labels) + 1))
        else:
            np.testing.assert_array_equal(actual, expected)
    assert (foreground_count == 0) == empty
    assert ("skimage not found" in capsys.readouterr().out) == (not with_skimage)


@pytest.mark.parametrize("seg_masks", [False, True], ids=["no-flag", "missing-mask"])
def test_make_train_without_masks_still_saves_images(volume, monkeypatch, capsys,
                                                     seg_masks):
    image_path, masks = volume
    if not seg_masks:
        np.save(image_path.with_name("volume_seg.npy"), {"masks": masks})
    run_make_train(monkeypatch, image_path, seg_masks=seg_masks)
    assert len(list(read_crops(image_path))) == 9
    assert not list((image_path.parent / "train").glob("*_seg.npy"))
    assert ("no segmentation masks found" in capsys.readouterr().out) == seg_masks
