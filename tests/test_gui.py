import numpy as np
import pytest


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def win(qapp):
    from cellpose.gui.gui3d import MainW_3d
    return MainW_3d()


def test_saturation_invariants_with_autobtn_unchecked(win):
    """Regression: when 'auto-adjust saturation' is unchecked, loading a 3D
    image must (1) leave self.saturation sized 3 x NZ so that update_plot's
    self.saturation[c][currentZ] index is valid, and (2) preserve the user's
    per-channel values across reloads that change NZ."""
    from cellpose.gui import io as gui_io

    win.autobtn.setChecked(False)

    # First load: NZ=10, saturation must be 3 x 10 with valid [lo, hi] entries.
    win.load_3D = True
    win.filename = "test.tif"
    gui_io._initialize_images(
        win, np.zeros((10, 64, 64, 3), dtype=np.float32), load_3D=True
    )

    assert win.NZ == 10
    assert len(win.saturation) == 3
    for c in range(3):
        assert len(win.saturation[c]) == win.NZ
        for z in range(win.NZ):
            assert len(win.saturation[c][z]) == 2  # [lo, hi]

    # Trigger the original failure path explicitly: move_in_Z -> update_plot
    # -> setLevels(self.saturation[self.color - 1][self.currentZ]).
    win.loaded = True
    win.color = 1  # red channel (color > 0 and < 4 branch)
    win.nchan = 3
    win.scroll.setValue(win.NZ - 1)
    win.move_in_Z()  # would have raised IndexError pre-fix

    # User picks custom per-channel values.
    for c in range(3):
        for z in range(win.NZ):
            win.saturation[c][z] = [10 + c, 200 + c]

    # Reload with a different NZ; user values should be broadcast across new Z.
    gui_io._initialize_images(
        win, np.zeros((8, 64, 64, 3), dtype=np.float32), load_3D=True
    )

    assert win.NZ == 8
    for c in range(3):
        assert len(win.saturation[c]) == win.NZ
        for z in range(win.NZ):
            assert win.saturation[c][z] == [10 + c, 200 + c]
