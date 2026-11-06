import logging

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")
pytest.importorskip("superqt")

from qtpy.QtWidgets import QApplication

from cellpose.gui import gui, io


@pytest.fixture
def filter_window(tmp_path, monkeypatch):
    # Keep the GUI's model list isolated; no model or image downloads are needed.
    for module in (gui.models, io):
        monkeypatch.setattr(module, "MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr(module, "MODEL_LIST_PATH", str(tmp_path / "models.txt"))
    monkeypatch.setattr(gui.core, "use_gpu", lambda **kwargs: False)
    app = QApplication.instance() or QApplication([])
    random_state = np.random.get_state()
    image_axis_order = gui.pg.getConfigOption("imageAxisOrder")
    window = None
    try:
        window = gui.MainW(logger=logging.getLogger(__name__))
        image = np.random.default_rng(42).uniform(0, 255, (64, 64, 1)).astype("float32")
        io._initialize_images(window, image)
        window.loaded = True
        window.filename = str(tmp_path / "synthetic.tif")
        window.enable_buttons()
        window.color = "rgb"
        window.update_plot()
        # A retained annotation must survive filtering and clearing the filter.
        window.cellpix[:, 20:30, 20:30] = 1
        window.outpix[:, 20, 20:30] = 1
        window.ncells.set(1)
        window.cellcolors = np.array([[255, 255, 255], [100, 200, 50]])
        window.ismanual = np.array([True])
        window.draw_layer()
        window.update_layer()
        yield window
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
            app.processEvents()
        np.random.set_state(random_state)
        gui.pg.setConfigOption("imageAxisOrder", image_axis_order)


@pytest.mark.parametrize("auto_saturation", [False, True])
def test_filter_repeated_click_refreshes_image(filter_window, auto_saturation):
    window = filter_window
    # Full-range levels stay at 0..255 after filtering, even with auto enabled.
    window.segmentation_settings.norm_percentile_low_box.setText("0")
    window.segmentation_settings.norm_percentile_high_box.setText("100")
    window.autobtn.setChecked(auto_saturation)
    raw = window.stack.copy()
    masks = window.cellpix.copy()
    outlines = window.outpix.copy()
    overlay = window.layer.image.copy()
    filter_button = next(b for b in window.FilterButtons if b.model_type == "filter")
    none_button = next(b for b in window.FilterButtons if b.model_type == "none")

    window.filt_edits[1].setText("1")
    filter_button.click()
    assert window.view == "restored"
    first = window.stack_filtered.copy()
    np.testing.assert_array_equal(window.img.image, first[0])

    window.filt_edits[1].setText("4")
    filter_button.click()
    second = window.stack_filtered.copy()
    assert not np.allclose(first, second)
    np.testing.assert_array_equal(window.img.image, second[0])

    # The same settings must be idempotent, not filter the previous output again.
    filter_button.click()
    np.testing.assert_array_equal(window.stack_filtered, second)
    np.testing.assert_array_equal(window.img.image, second[0])
    # Returning to earlier settings must also start from the raw stack.
    window.filt_edits[1].setText("1")
    filter_button.click()
    np.testing.assert_array_equal(window.stack_filtered, first)
    np.testing.assert_array_equal(window.img.image, first[0])

    none_button.click()
    assert window.restore is None
    assert window.view == "image"
    assert not hasattr(window, "stack_filtered")
    assert all(float(edit.text()) == 0 for edit in window.filt_edits[:3])
    np.testing.assert_array_equal(window.img.image, raw[0])

    window.filt_edits[1].setText("4")
    filter_button.click()
    np.testing.assert_array_equal(window.stack_filtered, second)
    np.testing.assert_array_equal(window.img.image, second[0])
    np.testing.assert_array_equal(window.stack, raw)
    np.testing.assert_array_equal(window.cellpix, masks)
    np.testing.assert_array_equal(window.outpix, outlines)
    np.testing.assert_array_equal(window.layer.image, overlay)
    np.testing.assert_array_equal(window.ismanual, [True])
    assert window.ncells.get() == 1
