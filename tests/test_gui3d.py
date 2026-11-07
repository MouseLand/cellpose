import logging
import sys
import traceback

import numpy as np
import pytest

pytest.importorskip("qtpy", reason="requires Cellpose GUI extras", exc_type=ImportError)
pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("superqt")

from qtpy import QtCore
from qtpy.QtWidgets import QApplication

from cellpose import models, utils
from cellpose.gui import gui3d, io


@pytest.fixture
def ortho_window(qt_app, tmp_path, monkeypatch):
    rng_state = np.random.get_state()
    config = pg.CONFIG_OPTIONS.copy()
    app = qt_app
    previous_widgets = set(app.topLevelWidgets())
    callback_errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: callback_errors.append(
        "".join(traceback.format_exception(*exc))))
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / "models"
    for module in (models, io):
        monkeypatch.setattr(module, "MODEL_DIR", model_dir)
        monkeypatch.setattr(module, "MODEL_LIST_PATH", str(model_dir / "gui_models.txt"))
    window = None
    try:
        window = gui3d.MainW_3d(logger=logging.getLogger(__name__))
        window.NZ, window.Ly, window.Lx = 17, 11, 13
        window.Ly0, window.Lx0 = window.Ly, window.Lx
        window.nchan = 1
        window.stack = (np.arange(17 * 11 * 13).reshape(17, 11, 13, 1)
                        % 251).astype(np.uint8)
        window.saturation = [[[0, 255] for _ in range(window.NZ)] for _ in range(3)]
        window.cellpix = np.zeros((17, 11, 13), np.uint16)
        window.cellpix[2:15, 2:9, 3:11] = 1
        window.outpix = utils.masks_to_outlines(window.cellpix) * window.cellpix
        window.cellcolors = np.array([[255, 255, 255], [30, 90, 150],
                                     [180, 70, 20]], np.uint8)
        window.currentZ = 8
        window.dzedit.setText("10")
        window.zaspectedit.setText("1.7")
        window.scroll.setMaximum(window.NZ - 1)
        window.scroll.setValue(window.currentZ)
        window.draw_layer()
        window.loaded = True
        window.orthobtn.setChecked(True)
        app.processEvents()
        yield window
    finally:
        try:
            for widget in set(app.topLevelWidgets()) - previous_widgets:
                widget.close()
            if window is not None:
                window.deleteLater()
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            app.processEvents()
            # Destroy the window and its children before its unparented widgets.
            for widget in set(app.topLevelWidgets()) - previous_widgets:
                widget.deleteLater()
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            app.processEvents()
        finally:
            np.random.set_state(rng_state)
            pg.setConfigOptions(**config)
        assert not callback_errors, "\n".join(callback_errors)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def rectangle_outlines(plane):
    """Independent oracle for the rectangular outer contours in these fixtures."""
    outlines = np.zeros(plane.shape, bool)
    for label in np.unique(plane):
        if label:
            rows, cols = np.nonzero(plane == label)
            r0, r1 = rows.min(), rows.max()
            c0, c1 = cols.min(), cols.max()
            assert (np.all(plane[r0:r1 + 1, c0] == label) and
                    np.all(plane[r0:r1 + 1, c1] == label) and
                    np.all(plane[r0, c0:c1 + 1] == label) and
                    np.all(plane[r1, c0:c1 + 1] == label)), \
                "The fixture must have a rectangular outer boundary"
            outlines[r0:r1 + 1, [c0, c1]] = True
            outlines[[r0, r1], c0:c1 + 1] = True
    return outlines


def test_rectangle_outline_oracle_rejects_invalid_perimeter():
    labels = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
    with pytest.raises(AssertionError, match="rectangular outer boundary"):
        rectangle_outlines(labels)


def overlay(window, labels, outlines):
    expected = np.zeros((*labels.shape, 4), np.uint8)
    if window.masksOn:
        expected[..., :3] = window.cellcolors[labels]
        expected[..., 3] = (labels > 0) * window.opacity
        if window.selected:
            expected[labels == window.selected] = [255, 255, 255, window.opacity]
    if window.outlinesOn:
        expected[outlines] = window.outcolor
    return expected


def assert_display(window, zmin=0, zmax=17):
    sources = [window.stack, window.cellpix, window.outpix]
    originals = [array.copy() for array in sources]
    window.update_plot()
    window.draw_layer()
    window.update_layer()
    QApplication.processEvents()
    x, y = window.xortho, window.yortho
    planes = [window.cellpix[:, :, x].T, window.cellpix[:, y, :]]
    crops = [(slice(None), slice(zmin, zmax)),
             (slice(zmin, zmax), slice(None))]
    raw = [window.stack[zmin:zmax, :, x, 0].T,
           window.stack[zmin:zmax, y, :, 0]]
    for j, (plane, crop) in enumerate(zip(planes, crops)):
        expected = overlay(window, plane[crop], rectangle_outlines(plane)[crop])
        np.testing.assert_array_equal(window.layerOrtho[j].image, expected)
        np.testing.assert_array_equal(window.imgOrtho[j].image, raw[j])
        assert window.layerOrtho[j].axisOrder == "row-major"
    np.testing.assert_array_equal(window.img.image, window.stack[window.currentZ])
    if window.masksOn or window.outlinesOn:
        expected_xy = overlay(window, window.cellpix[window.currentZ],
                              window.outpix[window.currentZ] > 0)
        np.testing.assert_array_equal(window.layer.image, expected_xy)
    for actual, source, original in zip(
            [window.stack, window.cellpix, window.outpix], sources, originals):
        assert actual is source
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("masks, outlines, selected", [
    (False, True, 0), (True, True, 1), (True, False, 1), (False, False, 0)
])
def test_ortho_box_display(ortho_window, masks, outlines, selected, monkeypatch):
    window = ortho_window
    window.MCheckBox.setChecked(masks)
    window.OCheckBox.setChecked(outlines)
    window.select_cell(selected)
    calls = []
    real_outlines = gui3d.masks_to_outlines

    def trace_outlines(plane):
        # Keep the real geometry, and reject volume scans or copied label volumes.
        assert plane.ndim == 2
        assert np.shares_memory(plane, window.cellpix)
        calls.append(plane.shape)
        return real_outlines(plane)

    monkeypatch.setattr(gui3d, "masks_to_outlines", trace_outlines)
    assert_display(window)
    assert calls == ([(11, 17), (17, 13)] if outlines else [])


@pytest.mark.parametrize("z, zmin, zmax", [(0, 0, 6), (8, 5, 11), (16, 11, 17)])
def test_ortho_z_crop(ortho_window, z, zmin, zmax):
    window = ortho_window
    window.MCheckBox.setChecked(False)
    window.OCheckBox.setChecked(True)
    window.dzedit.setText("3")
    window.scroll.setValue(z)
    assert_display(window, zmin, zmax)
    assert window.zc == z - zmin


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["z", "y", "x"])
def test_ortho_touching_labels(ortho_window, axis):
    window = ortho_window
    region = [slice(2, 15), slice(2, 9), slice(3, 11)]
    region[axis] = slice([8, 6, 7][axis], region[axis].stop)
    window.cellpix[tuple(region)] = 2
    window.outpix = utils.masks_to_outlines(window.cellpix) * window.cellpix
    window.MCheckBox.setChecked(False)
    window.OCheckBox.setChecked(True)
    assert_display(window)


def test_ortho_crosshair_and_label_changes(ortho_window):
    window = ortho_window
    window.MCheckBox.setChecked(False)
    window.OCheckBox.setChecked(True)
    for y, x in [(5, 6), (0, 6), (5, 0), (0, 0), (3, 8)]:
        window.yortho, window.xortho = y, x
        assert_display(window)
        assert window.hLine.value() == y
        assert window.vLine.value() == x
    # A label edit must appear on the next refresh, without an outline cache.
    window.cellpix[:4] = 0
    window.outpix = utils.masks_to_outlines(window.cellpix) * window.cellpix
    assert_display(window)


def test_ortho_crop_preserves_external_contours(ortho_window):
    window = ortho_window
    # A cavity is enclosed in both full orthogonal planes, but opens onto the
    # cropped Z edges. Cropping before contouring would introduce false edges.
    window.cellpix[4:13, 4:7, 5:9] = 0
    window.outpix = utils.masks_to_outlines(window.cellpix) * window.cellpix
    window.MCheckBox.setChecked(False)
    window.OCheckBox.setChecked(True)
    window.dzedit.setText("3")
    assert_display(window, 5, 11)


def test_ortho_refresh_preserves_exports(ortho_window, tmp_path):
    window = ortho_window
    window.filename = str(tmp_path / "volume.tif")
    masks = window.cellpix.copy()
    outlines = window.outpix.copy()
    raw = window.stack.copy()
    window.OCheckBox.setChecked(True)
    window.update_ortho()
    io._save_sets(window)
    io._save_png(window)
    saved = np.load(tmp_path / "volume_seg.npy", allow_pickle=True).item()
    np.testing.assert_array_equal(saved["masks"], masks)
    np.testing.assert_array_equal(saved["outlines"], outlines)
    np.testing.assert_array_equal(io.imread(str(tmp_path / "volume_cp_masks.tif")), masks)
    np.testing.assert_array_equal(window.stack, raw)
