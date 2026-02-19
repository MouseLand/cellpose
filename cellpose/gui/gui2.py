import os
import numpy as np
from enum import IntEnum

from qtpy.QtWidgets import QMainWindow, QWidget, QGridLayout, QScrollArea, QVBoxLayout, QApplication
from qtpy import QtGui
from qtpy import QtCore
import pyqtgraph as pg

from pathlib import Path
import sys

from cellpose import transforms
from cellpose.gui import menus
from cellpose.gui import io
import guiparts


class ColorMode(IntEnum):
    RGB = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    GRAY = 4
    SPECTRAL = 5


class ViewMode(IntEnum):
    IMAGE = 0
    GRAD_XY = 1
    CELLPROB = 2
    RESTORED = 3


def _make_single_channel_lut(channel_index):
    """Create a 256x3 uint8 LUT for a single RGB channel."""
    r = np.arange(256)
    color = np.zeros((256, 3), dtype=np.uint8)
    color[:, channel_index] = r
    cmap = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return cmap.getLookupTable(start=0.0, stop=255.0, alpha=False)


def _make_spectral_lut():
    # make spectral colormap
    r = np.array([
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
        84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 120, 112, 104, 96, 88,
        80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 11, 15, 19, 23,
        27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103,
        107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167,
        171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231,
        235, 239, 243, 247, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255
    ])
    g = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 3, 3,
        2, 2, 1, 1, 0, 0, 0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111,
        119, 127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
        247, 255, 247, 239, 231, 223, 215, 207, 199, 191, 183, 175, 167, 159, 151, 143,
        135, 128, 129, 131, 132, 134, 135, 137, 139, 140, 142, 143, 145, 147, 148, 150,
        151, 153, 154, 156, 158, 159, 161, 162, 164, 166, 167, 169, 170, 172, 174, 175,
        177, 178, 180, 181, 183, 185, 186, 188, 189, 191, 193, 194, 196, 197, 199, 201,
        202, 204, 205, 207, 208, 210, 212, 213, 215, 216, 218, 220, 221, 223, 224, 226,
        228, 229, 231, 232, 234, 235, 237, 239, 240, 242, 243, 245, 247, 248, 250, 251,
        253, 255, 251, 247, 243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199,
        195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135,
        131, 127, 123, 119, 115, 111, 107, 103, 99, 95, 91, 87, 83, 79, 75, 71, 67, 63,
        59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3, 0, 8, 16, 24, 32, 41,
        49, 57, 65, 74, 82, 90, 98, 106, 115, 123, 131, 139, 148, 156, 164, 172, 180,
        189, 197, 205, 213, 222, 230, 238, 246, 254
    ])
    b = np.array([
        0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143,
        151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 247,
        243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199, 195, 191, 187, 183,
        179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135, 131, 128, 126, 124,
        122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90,
        88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50,
        48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,
        8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 16, 24, 32, 41, 49, 57, 65, 74,
        82, 90, 98, 106, 115, 123, 131, 139, 148, 156, 164, 172, 180, 189, 197, 205,
        213, 222, 230, 238, 246, 254
    ])
    color = (np.vstack((r, g, b)).T).astype(np.uint8)
    spectral = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return spectral.getLookupTable(start=0.0, stop=255.0, alpha=False)


def run(image=None):
    app = QApplication(sys.argv)
    icon_path = Path.home().joinpath(".cellpose", "logo.png")
    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    app.setPalette(guiparts.DarkPalette())
    main_window = MainW(image=image)
    main_window.show()
    ret = app.exec()
    sys.exit(ret)


class MainW(QMainWindow):

    loadedChanged = QtCore.Signal(bool)

    def __init__(self, image=None):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setStyleSheet(guiparts.stylesheet())

        self._loaded = False
        self.color = ColorMode.RGB
        self.view = ViewMode.IMAGE
        self.saturation = [[0, 255.], [0, 255.], [0, 255.]]

        # LUTs keyed by ColorMode
        self.luts = {
            ColorMode.RED: _make_single_channel_lut(0),
            ColorMode.GREEN: _make_single_channel_lut(1),
            ColorMode.BLUE: _make_single_channel_lut(2),
            ColorMode.SPECTRAL: _make_spectral_lut(),
        }

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.main_layout = QGridLayout()
        self.cwidget.setLayout(self.main_layout)
        self.setCentralWidget(self.cwidget)
        self.main_layout.setVerticalSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 10)

        # ---- SIDEBAR (scrollable, left) ---- #
        self.scrollarea = QScrollArea()
        self.scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollarea.setStyleSheet("QScrollArea { border: none }")
        self.scrollarea.setWidgetResizable(True)
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_widget.setLayout(sidebar_layout)
        self.scrollarea.setWidget(sidebar_widget)
        self.main_layout.addWidget(self.scrollarea, 0, 0, 39, 9)

        # -- Views panel --
        self.views_panel = guiparts.ViewsPanel()
        sidebar_layout.addWidget(self.views_panel)

        # -- Segmentation panel --
        self.seg_panel = guiparts.SegmentationPanel()
        sidebar_layout.addWidget(self.seg_panel)

        sidebar_layout.addStretch()

        # ---- DRAWING AREA (right) ---- #
        self.win = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.win, 0, 9, 40, 30)
        self.main_layout.setColumnStretch(10, 1)

        self.__make_viewbox()
        self.setAcceptDrops(True)


        # ---- CONNECT SIDEBAR SIGNALS ---- #
        self.views_panel.colorChanged.connect(self._on_color_changed)
        self.views_panel.viewChanged.connect(self._on_view_changed)
        self.views_panel.saturationChanged.connect(self._on_saturation_changed)
        self.views_panel.autoSaturationToggled.connect(self._on_auto_saturation)
        self.seg_panel.runSegmentation.connect(self._on_run_segmentation)
        self.loadedChanged.connect(self.seg_panel.run_button.setEnabled)

        self.win.show()

    @property
    def loaded(self):
        return self._loaded

    @loaded.setter
    def loaded(self, value):
        if value != self._loaded:
            self._loaded = value
            self.loadedChanged.emit(value)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self._load_image(filename=files[0])

    def __make_viewbox(self):
        """Create a viewbox for the image, only used at startup."""
        self.viewbox = guiparts.ViewBoxNoRightDrag(
            parent=self, lockAspect=True, name="plot1",
            border=[100, 100, 100], invertY=True,
        )
        self.viewbox.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size = 3
        self.win.addItem(self.viewbox, 0, 0, rowspan=1, colspan=1)
        self.viewbox.setMenuEnabled(False)
        self.viewbox.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.viewbox, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.viewbox, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.viewbox, parent=self)
        self.scale.setLevels([0, 255])
        self.viewbox.scene().contextMenuItem = self.viewbox
        self.Ly, self.Lx = 512, 512
        self.viewbox.addItem(self.img)
        self.viewbox.addItem(self.layer)
        self.viewbox.addItem(self.scale)

    def _load_image(self, filename=None):
        """Load image, normalize to float32 0-255, compute saturation, update display."""
        image = io.imread_2D(filename)
        image = transforms.convert_image(image)
        if image.ndim < 4:
            image = image[np.newaxis, ...]
        self.imageContainer = guiparts.ImageDataContainer(image)

        self.loaded = True
        if self.views_panel.autobtn.isChecked():
            self._compute_saturation()
        self.update_plot()

    def _compute_saturation(self):
        """Compute per-channel saturation from 1st/99th percentiles and update sliders."""
        self.saturation = self.imageContainer.get_saturation_percentiles()

        # Push computed values to sliders
        for i, (low, high) in enumerate(self.saturation):
            self.views_panel.set_slider_values(i, low, high)

    def update_plot(self):
        if not self.loaded:
            return

        current_z = 0
        image = self.imageContainer.image[current_z]

        if self.color == ColorMode.RGB:
            self.img.setImage(image, autoLevels=False, lut=None)
            levels = np.array([self.saturation[0], self.saturation[1], self.saturation[2]])
            self.img.setLevels(levels)
        elif self.color in (ColorMode.RED, ColorMode.GREEN, ColorMode.BLUE):
            ch = self.color - ColorMode.RED
            self.img.setImage(image[:, :, ch], autoLevels=False, lut=self.luts[self.color])
            self.img.setLevels(self.saturation[ch])
        elif self.color == ColorMode.GRAY:
            gray = image.mean(axis=-1)
            self.img.setImage(gray, autoLevels=False, lut=None)
            self.img.setLevels(self.saturation[0])
        elif self.color == ColorMode.SPECTRAL:
            gray = image.mean(axis=-1)
            self.img.setImage(gray, autoLevels=False, lut=self.luts[ColorMode.SPECTRAL])
            self.img.setLevels(self.saturation[0])

    # ---- Sidebar signal handlers ---- #

    def _on_color_changed(self, index):
        self.color = ColorMode(index)
        self.view = ViewMode.IMAGE
        self.views_panel.ViewDropDown.setCurrentIndex(ViewMode.IMAGE)
        if self.loaded:
            self.update_plot()

    def _on_view_changed(self, index):
        self.view = ViewMode(index)
        if self.loaded:
            self.update_plot()

    def _on_saturation_changed(self, channel_name):
        idx = guiparts.ViewsPanel.CHANNEL_NAMES.index(channel_name)
        lo, hi = self.views_panel.slider_values(idx)
        self.saturation[idx] = [lo, hi]
        if self.loaded:
            self.update_plot()

    def _on_auto_saturation(self, checked):
        if checked and self.loaded:
            self._compute_saturation()
            self.update_plot()

    def _on_run_segmentation(self, model_name):
        print(f"GUI_INFO: run segmentation with {model_name}")


if __name__ == "__main__":
    run('')
