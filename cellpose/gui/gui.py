"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, pathlib, warnings, datetime, time, copy

from qtpy import QtGui, QtCore
from superqt import QRangeSlider, QCollapsible
from qtpy.QtWidgets import QScrollArea, QMainWindow, QApplication, QWidget, QScrollBar, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox
import pyqtgraph as pg

import numpy as np
from scipy.stats import mode
import cv2

from . import guiparts, menus, io
from .. import models, core, dynamics, version, denoise, train
from ..utils import download_url_to_file, masks_to_outlines, diameters
from ..io import get_image_files, imsave, imread
from ..transforms import resize_image, normalize99, normalize99_tile, smooth_sharpen_img
from ..models import normalize_default
from ..plot import disk

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

try:
    from google.cloud import storage
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "key/cellpose-data-writer.json")
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

Horizontal = QtCore.Qt.Orientation.Horizontal


class Slider(QRangeSlider):

    def __init__(self, parent, name, color):
        super().__init__(Horizontal)
        self.setEnabled(False)
        self.valueChanged.connect(lambda: self.levelChanged(parent))
        self.name = name

        self.setStyleSheet(""" QSlider{
                             background-color: transparent;
                             }
        """)
        self.show()

    def levelChanged(self, parent):
        parent.level_change(self.name)


class QHLine(QFrame):

    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        #self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(8)


def make_bwr():
    # make a bwr colormap
    b = np.append(255 * np.ones(128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255 * np.ones(128))[:, np.newaxis]
    g = np.append(np.linspace(0, 255, 128),
                  np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    color = np.concatenate((r, g, b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return bwr


def make_spectral():
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
    return spectral


def make_cmap(cm=0):
    # make a single channel colormap
    r = np.arange(0, 256)
    color = np.zeros((256, 3))
    color[:, cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return cmap


def run(image=None):
    from ..io import logger_setup
    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path, progress=True)
    if not guip_path.is_file():
        print("downloading help window image")
        download_url_to_file("https://www.cellpose.org/static/images/cellpose_gui.png",
                             guip_path, progress=True)
    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    app.setPalette(guiparts.DarkPalette())
    #app.setStyleSheet("QLineEdit { color: yellow }")

    # models.download_model_weights() # does not exist
    MainW(image=image, logger=logger)
    ret = app.exec_()
    sys.exit(ret)


class MainW(QMainWindow):

    def __init__(self, image=None, logger=None):
        super(MainW, self).__init__()

        self.logger = logger
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle(f"cellpose v{version}")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        # rgb(150,255,150)
        self.setStyleSheet(guiparts.stylesheet())

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)
        menus.helpmenu(self)

        self.stylePressed = """QPushButton {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.styleUnpressed = """QPushButton {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.lmain = QGridLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setVerticalSpacing(0)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.imask = 0
        self.scrollarea = QScrollArea()
        self.scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollarea.setStyleSheet("""QScrollArea { border: none }""")
        self.scrollarea.setWidgetResizable(True)
        self.swidget = QWidget(self)
        self.scrollarea.setWidget(self.swidget)
        self.l0 = QGridLayout()
        self.swidget.setLayout(self.l0)
        b = self.make_buttons()
        self.lmain.addWidget(self.scrollarea, 0, 0, 39, 9)

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()

        self.lmain.addWidget(self.win, 0, 9, 40, 30)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.lmain.setColumnStretch(10, 1)
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        # spectral colormap
        self.cmap.append(make_spectral().getLookupTable(start=0.0, stop=255.0,
                                                        alpha=False))
        # single channel colormaps
        for i in range(3):
            self.cmap.append(
                make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False))

        if MATPLOTLIB:
            self.colormap = (plt.get_cmap("gist_ncar")(np.linspace(0.0, .9, 1000000)) *
                             255).astype(np.uint8)
            np.random.seed(42)  # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42)  # make colors stable
            self.colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(
                np.uint8)
        self.NZ = 1
        self.restore = None
        self.ratio = 1.
        self.reset()

        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename)

        # training settings
        d = datetime.datetime.now()
        self.training_params = {
            "model_index": 0,
            "learning_rate": 0.1,
            "weight_decay": 0.0001,
            "n_epochs": 100,
            "model_name": "CP" + d.strftime("_%Y%m%d_%H%M%S"),
        }

        self.load_3D = False
        self.stitch_threshold = 0.

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    def help_window(self):
        HW = guiparts.HelpWindow(self)
        HW.show()

    def train_help_window(self):
        THW = guiparts.TrainHelpWindow(self)
        THW.show()

    def gui_window(self):
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        self.boldfont = QtGui.QFont("Arial", 11, QtGui.QFont.Bold)
        self.boldmedfont = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 9)
        self.smallfont = QtGui.QFont("Arial", 8)

        b = 0
        self.satBox = QGroupBox("Views")
        self.satBox.setFont(self.boldfont)
        self.satBoxG = QGridLayout()
        self.satBox.setLayout(self.satBoxG)
        self.l0.addWidget(self.satBox, b, 0, 1, 9)

        b0 = 0
        self.view = 0  # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0  # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBDropDown = QComboBox()
        self.RGBDropDown.addItems(
            ["RGB", "red=R", "green=G", "blue=B", "gray", "spectral"])
        self.RGBDropDown.setFont(self.medfont)
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.satBoxG.addWidget(self.RGBDropDown, b0, 0, 1, 3)

        label = QLabel("<p>[&uarr; / &darr; or W/S]</p>")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 3, 1, 3)
        label = QLabel("[R / G / B \n toggles color ]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 6, 1, 3)

        b0 += 1
        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.setFont(self.medfont)
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.ViewDropDown.currentIndexChanged.connect(self.update_plot)
        self.satBoxG.addWidget(self.ViewDropDown, b0, 0, 2, 3)

        label = QLabel("[pageup / pagedown]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 3, 1, 5)

        b0 += 2
        label = QLabel("")
        label.setToolTip(
            "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
        )
        self.satBoxG.addWidget(label, b0, 0, 1, 5)

        self.autobtn = QCheckBox("auto-adjust saturation")
        self.autobtn.setToolTip("sets scale-bars as normalized for segmentation")
        self.autobtn.setFont(self.medfont)
        self.autobtn.setChecked(True)
        self.satBoxG.addWidget(self.autobtn, b0, 1, 1, 8)

        b0 += 1
        self.sliders = []
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100]]
        colornames = ["red", "Chartreuse", "DodgerBlue"]
        names = ["red", "green", "blue"]
        for r in range(3):
            b0 += 1
            if r == 0:
                label = QLabel('<font color="gray">gray/</font><br>red')
            else:
                label = QLabel(names[r] + ":")
            label.setStyleSheet(f"color: {colornames[r]}")
            label.setFont(self.boldmedfont)
            self.satBoxG.addWidget(label, b0, 0, 1, 2)
            self.sliders.append(Slider(self, names[r], colors[r]))
            self.sliders[-1].setMinimum(-.1)
            self.sliders[-1].setMaximum(255.1)
            self.sliders[-1].setValue([0, 255])
            self.sliders[-1].setToolTip(
                "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
            )
            #self.sliders[-1].setTickPosition(QSlider.TicksRight)
            self.satBoxG.addWidget(self.sliders[-1], b0, 2, 1, 7)

        b += 1
        self.drawBox = QGroupBox("Drawing")
        self.drawBox.setFont(self.boldfont)
        self.drawBoxG = QGridLayout()
        self.drawBox.setLayout(self.drawBoxG)
        self.l0.addWidget(self.drawBox, b, 0, 1, 9)
        self.autosave = True

        b0 = 0
        self.brush_size = 3
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["1", "3", "5", "7", "9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(40)
        self.BrushChoose.setFont(self.medfont)
        self.drawBoxG.addWidget(self.BrushChoose, b0, 3, 1, 2)
        label = QLabel("brush size:")
        label.setFont(self.medfont)
        self.drawBoxG.addWidget(label, b0, 0, 1, 3)

        b0 += 1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox("MASKS ON [X]")
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.drawBoxG.addWidget(self.MCheckBox, b0, 0, 1, 5)

        b0 += 1
        # turn off outlines
        self.outlinesOn = False  # turn off by default
        self.OCheckBox = QCheckBox("outlines on [Z]")
        self.OCheckBox.setFont(self.medfont)
        self.drawBoxG.addWidget(self.OCheckBox, b0, 0, 1, 5)
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks)

        b0 += 1
        self.SCheckBox = QCheckBox("single stroke")
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.setEnabled(True)
        self.drawBoxG.addWidget(self.SCheckBox, b0, 0, 1, 5)

        # buttons for deleting multiple cells
        self.deleteBox = QGroupBox("delete multiple ROIs")
        self.deleteBox.setStyleSheet("color: rgb(200, 200, 200)")
        self.deleteBox.setFont(self.medfont)
        self.deleteBoxG = QGridLayout()
        self.deleteBox.setLayout(self.deleteBoxG)
        self.drawBoxG.addWidget(self.deleteBox, 0, 5, 4, 4)
        self.MakeDeletionRegionButton = QPushButton("region-select")
        self.MakeDeletionRegionButton.clicked.connect(self.remove_region_cells)
        self.deleteBoxG.addWidget(self.MakeDeletionRegionButton, 0, 0, 1, 4)
        self.MakeDeletionRegionButton.setFont(self.smallfont)
        self.MakeDeletionRegionButton.setFixedWidth(70)
        self.DeleteMultipleROIButton = QPushButton("click-select")
        self.DeleteMultipleROIButton.clicked.connect(self.delete_multiple_cells)
        self.deleteBoxG.addWidget(self.DeleteMultipleROIButton, 1, 0, 1, 4)
        self.DeleteMultipleROIButton.setFont(self.smallfont)
        self.DeleteMultipleROIButton.setFixedWidth(70)
        self.DoneDeleteMultipleROIButton = QPushButton("done")
        self.DoneDeleteMultipleROIButton.clicked.connect(
            self.done_remove_multiple_cells)
        self.deleteBoxG.addWidget(self.DoneDeleteMultipleROIButton, 2, 0, 1, 2)
        self.DoneDeleteMultipleROIButton.setFont(self.smallfont)
        self.DoneDeleteMultipleROIButton.setFixedWidth(35)
        self.CancelDeleteMultipleROIButton = QPushButton("cancel")
        self.CancelDeleteMultipleROIButton.clicked.connect(self.cancel_remove_multiple)
        self.deleteBoxG.addWidget(self.CancelDeleteMultipleROIButton, 2, 2, 1, 2)
        self.CancelDeleteMultipleROIButton.setFont(self.smallfont)
        self.CancelDeleteMultipleROIButton.setFixedWidth(35)

        b += 1
        b0 = 0
        self.segBox = QGroupBox("Segmentation")
        self.segBoxG = QGridLayout()
        self.segBox.setLayout(self.segBoxG)
        self.l0.addWidget(self.segBox, b, 0, 1, 9)
        self.segBox.setFont(self.boldfont)

        self.diameter = 30
        label = QLabel("diameter (pixels):")
        label.setFont(self.medfont)
        label.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.segBoxG.addWidget(label, b0, 0, 1, 4)
        self.Diameter = QLineEdit()
        self.Diameter.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the "cyto3" model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.Diameter.setText(str(self.diameter))
        self.Diameter.setFont(self.medfont)
        self.Diameter.returnPressed.connect(self.update_scale)
        self.Diameter.setFixedWidth(50)
        self.segBoxG.addWidget(self.Diameter, b0, 4, 1, 2)

        # compute diameter
        self.SizeButton = QPushButton("calibrate")
        self.SizeButton.setFont(self.medfont)
        self.SizeButton.clicked.connect(self.calibrate_size)
        self.segBoxG.addWidget(self.SizeButton, b0, 6, 1, 3)
        #self.SizeButton.setFixedWidth(65)
        self.SizeButton.setEnabled(False)
        self.SizeButton.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the cyto3 model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )

        b0 += 1
        # choose channel
        self.ChannelChoose = [QComboBox(), QComboBox()]
        self.ChannelChoose[0].addItems(["0: gray", "1: red", "2: green", "3: blue"])
        self.ChannelChoose[1].addItems(["0: none", "1: red", "2: green", "3: blue"])
        cstr = ["chan to segment:", "chan2 (optional): "]
        for i in range(2):
            self.ChannelChoose[i].setFont(self.medfont)
            label = QLabel(cstr[i])
            label.setFont(self.medfont)
            if i == 0:
                label.setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
                self.ChannelChoose[i].setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
            else:
                label.setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
                self.ChannelChoose[i].setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
            self.segBoxG.addWidget(label, b0 + i, 0, 1, 4)
            self.segBoxG.addWidget(self.ChannelChoose[i], b0 + i, 4, 1, 5)

        b0 += 2

        # use GPU
        self.useGPU = QCheckBox("use GPU")
        self.useGPU.setToolTip(
            "if you have specially installed the <i>cuda</i> version of torch, then you can activate this"
        )
        self.useGPU.setFont(self.medfont)
        self.check_gpu()
        self.segBoxG.addWidget(self.useGPU, b0, 0, 1, 3)

        # compute segmentation with general models
        self.net_text = ["run cyto3"]
        nett = ["cellpose super-generalist model"]

        #label = QLabel("Run:")
        #label.setFont(self.boldfont)
        #label.setFont(self.medfont)
        #self.segBoxG.addWidget(label, b0, 0, 1, 2)
        self.StyleButtons = []
        jj = 4
        for j in range(len(self.net_text)):
            self.StyleButtons.append(
                guiparts.ModelButton(self, self.net_text[j], self.net_text[j]))
            w = 5
            self.segBoxG.addWidget(self.StyleButtons[-1], b0, jj, 1, w)
            jj += w
            #self.StyleButtons[-1].setFixedWidth(140)
            self.StyleButtons[-1].setToolTip(nett[j])

        b0 += 1
        self.roi_count = QLabel("0 ROIs")
        self.roi_count.setFont(self.boldfont)
        self.roi_count.setAlignment(QtCore.Qt.AlignLeft)
        self.segBoxG.addWidget(self.roi_count, b0, 0, 1, 4)

        self.progress = QProgressBar(self)
        self.segBoxG.addWidget(self.progress, b0, 4, 1, 5)

        b0 += 1
        self.segaBox = QCollapsible("additional settings")
        self.segaBox.setFont(self.medfont)
        self.segaBox._toggle_btn.setFont(self.medfont)
        self.segaBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.segaBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        #_content.layout().setContentsMargins(QtCore.QMargins(0, -20, -20, -20))
        self.segaBox.setContent(_content)
        self.segBoxG.addWidget(self.segaBox, b0, 0, 1, 9)

        b0 = 0
        # post-hoc paramater tuning
        label = QLabel("flow\nthreshold:")
        label.setToolTip(
            "threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 0, 1, 2)
        self.flow_threshold = QLineEdit()
        self.flow_threshold.setText("0.4")
        self.flow_threshold.returnPressed.connect(self.compute_cprob)
        self.flow_threshold.setFixedWidth(40)
        self.flow_threshold.setFont(self.medfont)
        self.segaBoxG.addWidget(self.flow_threshold, b0, 2, 1, 2)
        self.flow_threshold.setToolTip(
            "threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run"
        )

        label = QLabel("cellprob\nthreshold:")
        label.setToolTip(
            "threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 4, 1, 2)
        self.cellprob_threshold = QLineEdit()
        self.cellprob_threshold.setText("0.0")
        self.cellprob_threshold.returnPressed.connect(self.compute_cprob)
        self.cellprob_threshold.setFixedWidth(40)
        self.cellprob_threshold.setFont(self.medfont)
        self.cellprob_threshold.setToolTip(
            "threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run"
        )
        self.segaBoxG.addWidget(self.cellprob_threshold, b0, 6, 1, 2)

        b0 += 1
        label = QLabel("norm percentiles:")
        label.setToolTip(
            "sets normalization percentiles for segmentation and denoising\n(pixels at lower percentile set to 0.0 and at upper set to 1.0 for network)"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 0, 1, 8)

        b0 += 1
        self.norm_vals = [1., 99.]
        self.norm_edits = []
        labels = ["lower", "upper"]
        tooltips = [
            "pixels at this percentile set to 0 (default 1.0)",
            "pixels at this percentile set to 1  (default 99.0)"
        ]
        for p in range(2):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.segaBoxG.addWidget(label, b0, 4 * (p % 2), 1, 2)
            self.norm_edits.append(QLineEdit())
            self.norm_edits[p].setText(str(self.norm_vals[p]))
            self.norm_edits[p].setFixedWidth(40)
            self.norm_edits[p].setFont(self.medfont)
            self.segaBoxG.addWidget(self.norm_edits[p], b0, 4 * (p % 2) + 2, 1, 2)
            self.norm_edits[p].setToolTip(tooltips[p])

        b0 += 1
        label = QLabel("niter dynamics:")
        label.setFont(self.medfont)
        label.setToolTip(
            "number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria"
        )
        self.segaBoxG.addWidget(label, b0, 0, 1, 4)
        self.niter = QLineEdit()
        self.niter.setText("0")
        self.niter.setFixedWidth(40)
        self.niter.setFont(self.medfont)
        self.niter.setToolTip(
            "number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria"
        )
        self.segaBoxG.addWidget(self.niter, b0, 4, 1, 2)

        b += 1
        b0 = 0
        self.modelBox = QGroupBox("Other models")
        self.modelBoxG = QGridLayout()
        self.modelBox.setLayout(self.modelBoxG)
        self.l0.addWidget(self.modelBox, b, 0, 1, 9)
        self.modelBox.setFont(self.boldfont)
        # choose models
        self.ModelChooseC = QComboBox()
        self.ModelChooseC.setFont(self.medfont)
        current_index = 0
        self.ModelChooseC.addItems(["custom models"])
        if len(self.model_strings) > 0:
            self.ModelChooseC.addItems(self.model_strings)
        self.ModelChooseC.setFixedWidth(175)
        self.ModelChooseC.setCurrentIndex(current_index)
        tipstr = 'add or train your own models in the "Models" file menu and choose model here'
        self.ModelChooseC.setToolTip(tipstr)
        self.ModelChooseC.activated.connect(lambda: self.model_choose(custom=True))
        self.modelBoxG.addWidget(self.ModelChooseC, b0, 0, 1, 8)

        # compute segmentation w/ custom model
        self.ModelButtonC = QPushButton(u"run")
        self.ModelButtonC.setFont(self.medfont)
        self.ModelButtonC.setFixedWidth(35)
        self.ModelButtonC.clicked.connect(
            lambda: self.compute_segmentation(custom=True))
        self.modelBoxG.addWidget(self.ModelButtonC, b0, 8, 1, 1)
        self.ModelButtonC.setEnabled(False)

        self.net_names = [
            "nuclei", "cyto2_cp3", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3",
            "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3",
            "cyto", "cyto2", "CPx"]

        nett = [
            "nuclei", "cellpose (cyto2_cp3)", "tissuenet_cp3", "livecell_cp3",
            "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3",
            "deepbacs_cp3", "cyto", "cyto2",
            "CPx (from Cellpose2)"
        ]
        b0 += 1
        self.ModelChooseB = QComboBox()
        self.ModelChooseB.setFont(self.medfont)
        self.ModelChooseB.addItems(["dataset-specific models"])
        self.ModelChooseB.addItems(nett)
        self.ModelChooseB.setFixedWidth(175)
        tipstr = "dataset-specific models"
        self.ModelChooseB.setToolTip(tipstr)
        self.ModelChooseB.activated.connect(lambda: self.model_choose(custom=False))
        self.modelBoxG.addWidget(self.ModelChooseB, b0, 0, 1, 8)

        # compute segmentation w/ cp model
        self.ModelButtonB = QPushButton(u"run")
        self.ModelButtonB.setFont(self.medfont)
        self.ModelButtonB.setFixedWidth(35)
        self.ModelButtonB.clicked.connect(
            lambda: self.compute_segmentation(custom=False))
        self.modelBoxG.addWidget(self.ModelButtonB, b0, 8, 1, 1)
        self.ModelButtonB.setEnabled(False)

        b += 1
        self.denoiseBox = QGroupBox("Image restoration")
        self.denoiseBox.setFont(self.boldfont)
        self.denoiseBoxG = QGridLayout()
        self.denoiseBox.setLayout(self.denoiseBoxG)
        self.l0.addWidget(self.denoiseBox, b, 0, 1, 9)

        b0 = 0
        
        # DENOISING
        self.DenoiseButtons = []
        nett = [
            "clear restore/filter",
            "filter image (settings below)",
            "denoise (please set cell diameter first)",
            "deblur (please set cell diameter first)",
            "upsample to 30. diameter (cyto3) or 17. diameter (nuclei) (please set cell diameter first) (disabled in 3D)",
            "one-click model trained to denoise+deblur+upsample (please set cell diameter first)"
        ]
        self.denoise_text = ["none", "filter", "denoise", "deblur", "upsample", "one-click"]
        self.restore = None
        self.ratio = 1.
        jj = 0
        w = 3
        for j in range(len(self.denoise_text)):
            self.DenoiseButtons.append(
                guiparts.DenoiseButton(self, self.denoise_text[j]))
            self.denoiseBoxG.addWidget(self.DenoiseButtons[-1], b0, jj, 1, w)
            self.DenoiseButtons[-1].setFixedWidth(75)
            self.DenoiseButtons[-1].setToolTip(nett[j])
            self.DenoiseButtons[-1].setFont(self.medfont)
            b0 += 1 if j%2==1 else 0
            jj = 0 if j%2==1 else jj + w

        # b0+=1
        self.save_norm = QCheckBox("save restored/filtered image")
        self.save_norm.setFont(self.medfont)
        self.save_norm.setToolTip("save restored/filtered image in _seg.npy file")
        self.save_norm.setChecked(True)
        # self.denoiseBoxG.addWidget(self.save_norm, b0, 0, 1, 8)

        b0 -= 3
        label = QLabel("restore-dataset:")
        label.setToolTip(
            "choose dataset and click [denoise], [deblur], [upsample], or [one-click]")
        label.setFont(self.medfont)
        self.denoiseBoxG.addWidget(label, b0, 6, 1, 3)

        b0 += 1
        self.DenoiseChoose = QComboBox()
        self.DenoiseChoose.setFont(self.medfont)
        self.DenoiseChoose.addItems(["cyto3", "cyto2", "nuclei"])
        self.DenoiseChoose.setFixedWidth(85)
        tipstr = "choose model type and click [denoise], [deblur], or [upsample]"
        self.DenoiseChoose.setToolTip(tipstr)
        self.denoiseBoxG.addWidget(self.DenoiseChoose, b0, 6, 1, 3)

        b0 += 2
        # FILTERING
        self.filtBox = QCollapsible("custom filter settings")
        self.filtBox._toggle_btn.setFont(self.medfont)
        self.filtBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.filtBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        #_content.layout().setContentsMargins(QtCore.QMargins(0, -20, -20, -20))
        self.filtBox.setContent(_content)
        self.denoiseBoxG.addWidget(self.filtBox, b0, 0, 1, 9)

        self.filt_vals = [0., 0., 0., 0.]
        self.filt_edits = []
        labels = [
            "sharpen\nradius", "smooth\nradius", "tile_norm\nblocksize",
            "tile_norm\nsmooth3D"
        ]
        tooltips = [
            "set size of surround-subtraction filter for sharpening image",
            "set size of gaussian filter for smoothing image",
            "set size of tiles to use to normalize image",
            "set amount of smoothing of normalization values across planes"
        ]

        for p in range(4):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.filtBoxG.addWidget(label, b0 + p // 2, 4 * (p % 2), 1, 2)
            self.filt_edits.append(QLineEdit())
            self.filt_edits[p].setText(str(self.filt_vals[p]))
            self.filt_edits[p].setFixedWidth(40)
            self.filt_edits[p].setFont(self.medfont)
            self.filtBoxG.addWidget(self.filt_edits[p], b0 + p // 2, 4 * (p % 2) + 2, 1,
                                    2)
            self.filt_edits[p].setToolTip(tooltips[p])

        b0 += 3
        self.norm3D_cb = QCheckBox("norm3D")
        self.norm3D_cb.setFont(self.medfont)
        self.norm3D_cb.setChecked(True)
        self.norm3D_cb.setToolTip("run same normalization across planes")
        self.filtBoxG.addWidget(self.norm3D_cb, b0, 0, 1, 3)

        self.invert_cb = QCheckBox("invert")
        self.invert_cb.setFont(self.medfont)
        self.invert_cb.setToolTip("invert image")
        self.filtBoxG.addWidget(self.invert_cb, b0, 3, 1, 3)

        b += 1
        self.l0.addWidget(QLabel(""), b, 0, 1, 9)
        self.l0.setRowStretch(b, 100)

        b += 1
        # scale toggle
        self.scale_on = True
        self.ScaleOn = QCheckBox("scale disk on")
        self.ScaleOn.setFont(self.medfont)
        self.ScaleOn.setStyleSheet("color: rgb(150,50,150);")
        self.ScaleOn.setChecked(True)
        self.ScaleOn.setToolTip("see current diameter as red disk at bottom")
        self.ScaleOn.toggled.connect(self.toggle_scale)
        self.l0.addWidget(self.ScaleOn, b, 0, 1, 5)

        return b

    def level_change(self, r):
        r = ["red", "green", "blue"].index(r)
        if self.loaded:
            sval = self.sliders[r].value()
            self.saturation[r][self.currentZ] = sval
            if not self.autobtn.isChecked():
                for r in range(3):
                    for i in range(len(self.saturation[r])):
                        self.saturation[r][i] = self.saturation[r][self.currentZ]
            self.update_plot()

    def keyPressEvent(self, event):
        if self.loaded:
            if not (event.modifiers() &
                    (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier |
                     QtCore.Qt.AltModifier) or self.in_stroke):
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                else:
                    nviews = self.ViewDropDown.count() - 1
                    nviews += int(
                        self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                                       1).isEnabled())
                    if event.key() == QtCore.Qt.Key_X:
                        self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Z:
                        self.OCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Left or event.key(
                    ) == QtCore.Qt.Key_A:
                        self.get_prev_image()
                    elif event.key() == QtCore.Qt.Key_Right or event.key(
                    ) == QtCore.Qt.Key_D:
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_PageDown:
                        self.view = (self.view + 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)
                    elif event.key() == QtCore.Qt.Key_PageUp:
                        self.view = (self.view - 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                    self.color = (self.color - 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_Down or event.key(
                ) == QtCore.Qt.Key_S:
                    self.color = (self.color + 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_R:
                    if self.color != 1:
                        self.color = 1
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_G:
                    if self.color != 2:
                        self.color = 2
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_B:
                    if self.color != 3:
                        self.color = 3
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (event.key() == QtCore.Qt.Key_Comma or
                      event.key() == QtCore.Qt.Key_Period):
                    count = self.BrushChoose.count()
                    gci = self.BrushChoose.currentIndex()
                    if event.key() == QtCore.Qt.Key_Comma:
                        gci = max(0, gci - 1)
                    else:
                        gci = min(count - 1, gci + 1)
                    self.BrushChoose.setCurrentIndex(gci)
                    self.brush_choose()
                if not updated:
                    self.update_plot()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def autosave_on(self):
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def check_gpu(self, torch=True):
        # also decide whether or not to use torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)
        if core.use_gpu(use_torch=True):
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")

    def get_channels(self):
        channels = [
            self.ChannelChoose[0].currentIndex(), self.ChannelChoose[1].currentIndex()
        ]
        if hasattr(self, "current_model"):
            if self.current_model == "nuclei":
                channels[1] = 0
        if channels[0] == 0:
            channels[1] = 0
        if self.nchan == 1:
            channels = [0, 0]
        elif self.nchan == 2:
            if channels[0] == 3:
                channels[0] = 1 if channels[1] != 1 else 2
                print(
                    f"GUI_WARNING: only two channels in image, cannot use blue channel, changing channels"
                )
            if channels[1] == 3:
                channels[1] = 1 if channels[0] != 1 else 2
                print(
                    f"GUI_WARNING: only two channels in image, cannot use blue channel, changing channels"
                )
        self.ChannelChoose[0].setCurrentIndex(channels[0])
        self.ChannelChoose[1].setCurrentIndex(channels[1])
        return channels

    def model_choose(self, custom=False):
        index = self.ModelChooseC.currentIndex(
        ) if custom else self.ModelChooseB.currentIndex()
        if index > 0:
            if custom:
                model_name = self.ModelChooseC.currentText()
            else:
                model_name = self.net_names[index - 1]
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=custom)
            self.diameter = self.model.diam_labels
            self.Diameter.setText("%0.2f" % self.diameter)
            print(
                f"GUI_INFO: diameter set to {self.diameter: 0.2f} (but can be changed)")

    def calibrate_size(self):
        self.initialize_model(model_name="cyto3")
        diams, _ = self.model.sz.eval(self.stack[self.currentZ].copy(),
                                      channels=self.get_channels(),
                                      progress=self.progress)
        diams = np.maximum(5.0, diams)
        self.logger.info("estimated diameter of cells using %s model = %0.1f pixels" %
                         (self.current_model, diams))
        self.Diameter.setText("%0.1f" % diams)
        self.diameter = diams
        self.update_scale()
        self.progress.setValue(100)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def enable_buttons(self):
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(True)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(True)
        for i in range(len(self.DenoiseButtons)):
            self.DenoiseButtons[i].setEnabled(True)
        if self.load_3D:
            self.DenoiseButtons[-2].setEnabled(False)
        self.ModelButtonB.setEnabled(True)
        self.SizeButton.setEnabled(True)
        self.newmodel.setEnabled(True)
        self.loadMasks.setEnabled(True)

        for n in range(self.nchan):
            self.sliders[n].setEnabled(True)
        for n in range(self.nchan, 3):
            self.sliders[n].setEnabled(True)

        self.toggle_mask_ops()

        self.update_plot()
        self.setWindowTitle(self.filename)

    def disable_buttons_removeROIs(self):
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(False)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(False)
        self.ModelButtonB.setEnabled(False)
        self.SizeButton.setEnabled(False)
        self.newmodel.setEnabled(False)
        self.loadMasks.setEnabled(False)
        self.saveSet.setEnabled(False)
        self.savePNG.setEnabled(False)
        self.saveFlows.setEnabled(False)
        self.saveOutlines.setEnabled(False)
        self.saveROIs.setEnabled(False)

        self.MakeDeletionRegionButton.setEnabled(False)
        self.DeleteMultipleROIButton.setEnabled(False)
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)

    def toggle_mask_ops(self):
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()

    def toggle_saving(self):
        if self.ncells > 0:
            self.saveSet.setEnabled(True)
            self.savePNG.setEnabled(True)
            self.saveFlows.setEnabled(True)
            self.saveOutlines.setEnabled(True)
            self.saveROIs.setEnabled(True)
        else:
            self.saveSet.setEnabled(False)
            self.savePNG.setEnabled(False)
            self.saveFlows.setEnabled(False)
            self.saveOutlines.setEnabled(False)
            self.saveROIs.setEnabled(False)

    def toggle_removals(self):
        if self.ncells > 0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
            self.MakeDeletionRegionButton.setEnabled(True)
            self.DeleteMultipleROIButton.setEnabled(True)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)
            self.MakeDeletionRegionButton.setEnabled(False)
            self.DeleteMultipleROIButton.setEnabled(False)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)

    def remove_action(self):
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        if (len(self.strokes) > 0 and self.strokes[-1][0][0] == self.currentZ):
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells)

    def undo_remove_action(self):
        self.undo_remove_cell()

    def get_files(self):
        folder = os.path.dirname(self.filename)
        mask_filter = "_masks"
        images = get_image_files(folder, mask_filter)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]
        idx = np.nonzero(np.array(fnames) == f0)[0][0]
        return images, idx

    def get_prev_image(self):
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        io._load_image(self, filename=images[idx])

    def get_next_image(self, load_seg=True):
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        io._load_image(self, filename=images[idx], load_seg=load_seg)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == ".npy":
            io._load_seg(self, filename=files[0], load_3D=self.load_3D)
        else:
            io._load_image(self, filename=files[0], load_seg=True, load_3D=self.load_3D)

    def toggle_masks(self):
        if self.MCheckBox.isChecked():
            self.masksOn = True
        else:
            self.masksOn = False
        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False
        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()
        if self.loaded:
            self.update_plot()
            self.update_layer()

    def make_viewbox(self):
        self.p0 = guiparts.ViewBoxNoRightDrag(parent=self, lockAspect=True,
                                              name="plot1", border=[100, 100,
                                                                    100], invertY=True)
        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size = 3
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0
        #self.p0.setMouseEnabled(x=False,y=False)
        self.Ly, self.Lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.nchan = 3
        self.loaded = False
        self.channel = [0, 1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.resize = False
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]

        # -- zero out image stack -- #
        self.opacity = 128  # how opaque masks should be
        self.outcolor = [200, 200, 255, 200]
        self.NZ, self.Ly, self.Lx = 1, 224, 224
        self.saturation = []
        for r in range(3):
            self.saturation.append([[0, 255] for n in range(self.NZ)])
            self.sliders[r].setValue([0, 255])
            self.sliders[r].setEnabled(False)
            self.sliders[r].show()
        self.currentZ = 0
        self.flows = [[], [], [], [], [[]]]
        # masks matrix
        # image matrix with a scale disk
        self.stack = np.zeros((1, self.Ly, self.Lx, 3))
        self.Lyr, self.Lxr = self.Ly, self.Lx
        self.Ly0, self.Lx0 = self.Ly, self.Lx
        self.radii = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        if self.restore and "upsample" in self.restore:
            self.cellpix_resize = self.cellpix
            self.cellpix_orig = self.cellpix
            self.outpix_resize = self.cellpix
            self.outpix_orig = self.cellpix
        self.ismanual = np.zeros(0, "bool")

        # -- set menus to default -- #
        self.color = 0
        self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        self.ViewDropDown.setCurrentIndex(0)
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        self.delete_restore()

        self.clear_all()

        #self.update_plot()
        self.filename = []
        self.loaded = False
        self.recompute_masks = False

        self.deleting_multiple = False
        self.removing_cells_list = []
        self.removing_region = False
        self.remove_roi_obj = None

    def delete_restore(self):
        """ delete restored imgs but don't reset settings """
        if hasattr(self, "stack_filtered"):
            del self.stack_filtered
        if hasattr(self, "cellpix_orig"):
            self.cellpix = self.cellpix_orig.copy()
            self.outpix = self.outpix_orig.copy()
            del self.outpix_orig, self.outpix_resize
            del self.cellpix_orig, self.cellpix_resize

    def clear_restore(self):
        """ delete restored imgs and reset settings """
        print("GUI_INFO: clearing restored image")
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        if self.ViewDropDown.currentIndex() == self.ViewDropDown.count() - 1:
            self.ViewDropDown.setCurrentIndex(0)
        self.delete_restore()
        self.restore = None
        self.ratio = 1.
        self.set_normalize_params(self.get_normalize_params())

    def brush_choose(self):
        self.brush_size = self.BrushChoose.currentIndex() * 2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_layer()

    def clear_all(self):
        self.prev_selected = 0
        self.selected = 0
        if self.restore and "upsample" in self.restore:
            self.layerz = 0 * np.ones((self.Lyr, self.Lxr, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.cellpix_resize = self.cellpix.copy()
            self.outpix_resize = self.outpix.copy()
            self.cellpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
            self.outpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
        else:
            self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)

        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]
        self.ncells = 0
        self.toggle_removals()
        self.update_scale()
        self.update_layer()

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity])
            self.update_layer()

    def select_cell_multi(self, idx):
        if idx > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity])
            self.update_layer()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < self.ncells + 1:
                z = self.currentZ
                self.layerz[self.cellpix[z] == idx] = np.append(
                    self.cellcolors[idx], self.opacity)
                if self.outlinesOn:
                    self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                        np.uint8)
                    #[0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def unselect_cell_multi(self, idx):
        z = self.currentZ
        self.layerz[self.cellpix[z] == idx] = np.append(self.cellcolors[idx],
                                                        self.opacity)
        if self.outlinesOn:
            self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                np.uint8)
            # [0,0,0,self.opacity])
        self.update_layer()

    def remove_cell(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        # because the function remove_single_cell updates the state of the cellpix and outpix arrays
        # by reindexing cells to avoid gaps in the indices, we need to remove the cells in reverse order
        # so that the indices are correct
        idx.sort(reverse=True)
        for i in idx:
            self.remove_single_cell(i)
        self.ncells -= len(idx)  # _save_sets uses ncells

        if self.ncells == 0:
            self.ClearButton.setEnabled(False)
        if self.NZ == 1:
            io._save_sets_with_check(self)

        self.update_layer()

    def remove_single_cell(self, idx):
        # remove from manual array
        self.selected = 0
        if self.NZ > 1:
            zextent = ((self.cellpix == idx).sum(axis=(1, 2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z] == idx
            op = self.outpix[z] == idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            if z == self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0, 0, 0, 0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix > idx] -= 1
        self.outpix[self.outpix > idx] -= 1

        if self.NZ == 1:
            self.removed_cell = [
                self.ismanual[idx - 1], self.cellcolors[idx],
                np.nonzero(cp),
                np.nonzero(op)
            ]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()
            self.track_changes.append(
                [d.strftime("%m/%d/%Y, %H:%M:%S"), "removed mask", [ar, ac]])
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx - 1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        del self.zdraw[idx - 1]
        print("GUI_INFO: removed cell %d" % (idx - 1))

    def remove_region_cells(self):
        if self.removing_cells_list:
            for idx in self.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.removing_cells_list.clear()
        self.disable_buttons_removeROIs()
        self.removing_region = True

        self.clear_multi_selected_cells()

        # make roi region here in center of view, making ROI half the size of the view
        roi_width = self.p0.viewRect().width() / 2
        x_loc = self.p0.viewRect().x() + (roi_width / 2)
        roi_height = self.p0.viewRect().height() / 2
        y_loc = self.p0.viewRect().y() + (roi_height / 2)

        pos = [x_loc, y_loc]
        roi = pg.RectROI(pos, [roi_width, roi_height], pen=pg.mkPen("y", width=2),
                         removable=True)
        roi.sigRemoveRequested.connect(self.remove_roi)
        roi.sigRegionChangeFinished.connect(self.roi_changed)
        self.p0.addItem(roi)
        self.remove_roi_obj = roi
        self.roi_changed(roi)

    def delete_multiple_cells(self):
        self.unselect_cell()
        self.disable_buttons_removeROIs()
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.MakeDeletionRegionButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)
        self.deleting_multiple = True

    def done_remove_multiple_cells(self):
        self.deleting_multiple = False
        self.removing_region = False
        self.DoneDeleteMultipleROIButton.setEnabled(False)
        self.MakeDeletionRegionButton.setEnabled(False)
        self.CancelDeleteMultipleROIButton.setEnabled(False)

        if self.removing_cells_list:
            self.removing_cells_list = list(set(self.removing_cells_list))
            display_remove_list = [i - 1 for i in self.removing_cells_list]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.removing_cells_list)
            self.removing_cells_list.clear()
            self.unselect_cell()
        self.enable_buttons()

        if self.remove_roi_obj is not None:
            self.remove_roi(self.remove_roi_obj)

    def merge_cells(self, idx):
        if self.selected == 0:
            return
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z] == self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z] == self.selected)
                touching = np.logical_and((ar0[:, np.newaxis] - ar1) < 3,
                                          (ac0[:, np.newaxis] - ac1) < 3).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z] == self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z] == self.selected)
                self.outpix[z, vr0, vc0] = 0
                self.outpix[z, vr1, vc1] = 0
                if touching > 0:
                    mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                    mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                    pvc, pvr = contours[-2][0].squeeze().T
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2

                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print("GUI_INFO: merged two cells")
            self.update_layer()
            io._save_sets_with_check(self)
            self.undo.setEnabled(False)
            self.redo.setEnabled(False)

    def undo_remove_cell(self):
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :], axis=0)
            self.ncells += 1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print(">>> added back removed cell")
            self.update_layer()
            io._save_sets_with_check(self)
            self.removed_cell = []
            self.redo.setEnabled(False)

    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        stroke = np.array(self.strokes[stroke_ind])
        cZ = self.currentZ
        inZ = stroke[0, 0] == cZ
        if inZ:
            outpix = self.outpix[cZ, stroke[:, 1], stroke[:, 2]] > 0
            self.layerz[stroke[~outpix, 1], stroke[~outpix, 2]] = np.array([0, 0, 0, 0])
            cellpix = self.cellpix[cZ, stroke[:, 1], stroke[:, 2]]
            ccol = self.cellcolors.copy()
            if self.selected > 0:
                ccol[self.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            if self.masksOn:
                col2mask = np.concatenate(
                    (col2mask, self.opacity * (cellpix[:, np.newaxis] > 0)), axis=-1)
            else:
                col2mask = np.concatenate((col2mask, 0 * (cellpix[:, np.newaxis] > 0)),
                                          axis=-1)
            self.layerz[stroke[:, 1], stroke[:, 2], :] = col2mask
            if self.outlinesOn:
                self.layerz[stroke[outpix, 1], stroke[outpix,
                                                      2]] = np.array(self.outcolor)
            if delete_points:
                # self.current_point_set = self.current_point_set[:-1*(stroke[:,-1]==1).sum()]
                del self.current_point_set[stroke_ind]
            self.update_layer()

        del self.strokes[stroke_ind]

    def plot_clicked(self, event):
        if event.button()==QtCore.Qt.LeftButton \
                and not event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)\
                and not self.removing_region:
            if event.double():
                try:
                    self.p0.setYRange(0, self.Ly + self.pr)
                except:
                    self.p0.setYRange(0, self.Ly)
                self.p0.setXRange(0, self.Lx)

    def cancel_remove_multiple(self):
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        # unselect all previously selected cells:
        for idx in self.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.removing_cells_list.clear()

    def add_roi(self, roi):
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        self.clear_multi_selected_cells()
        assert roi == self.remove_roi_obj
        self.remove_roi_obj = None
        self.p0.removeItem(roi)
        self.removing_region = False

    def roi_changed(self, roi):
        # find the overlapping cells and make them selected
        pos = roi.pos()
        size = roi.size()
        x0 = int(pos.x())
        y0 = int(pos.y())
        x1 = int(pos.x() + size.x())
        y1 = int(pos.y() + size.y())
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > self.Lx:
            x1 = self.Lx
        if y1 > self.Ly:
            y1 = self.Ly

        # find cells in that region
        cell_idxs = np.unique(self.cellpix[self.currentZ, y0:y1, x0:x1])
        cell_idxs = np.trim_zeros(cell_idxs)
        # deselect cells not in region by deselecting all and then selecting the ones in the region
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.removing_cells_list.append(idx)

        self.update_layer()

    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)

    def color_choose(self):
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.ViewDropDown.setCurrentIndex(self.view)
        self.update_plot()

    def update_plot(self):
        self.view = self.ViewDropDown.currentIndex()
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape

        if self.restore and "upsample" in self.restore:
            if self.view != 0:
                if self.view == 3:
                    self.resize = True
                elif len(self.flows[0]) > 0 and self.flows[0].shape[1] == self.Lyr:
                    self.resize = True
                else:
                    self.resize = False
            else:
                self.resize = False
            self.draw_layer()
            self.update_scale()
            self.update_layer()

        if self.view == 0 or self.view == self.ViewDropDown.count() - 1:
            image = self.stack[
                self.currentZ] if self.view == 0 else self.stack_filtered[self.currentZ]
            if self.nchan == 1:
                # show single channel
                image = image[..., 0]
            if self.color == 0:
                self.img.setImage(image, autoLevels=False, lut=None)
                if self.nchan > 1:
                    levels = np.array([
                        self.saturation[0][self.currentZ],
                        self.saturation[1][self.currentZ],
                        self.saturation[2][self.currentZ]
                    ])
                    self.img.setLevels(levels)
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color > 0 and self.color < 4:
                if self.nchan > 1:
                    image = image[:, :, self.color - 1]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
                if self.nchan > 1:
                    self.img.setLevels(self.saturation[self.color - 1][self.currentZ])
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 4:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=None)
                self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 5:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
                self.img.setLevels(self.saturation[0][self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])

        for r in range(3):
            self.sliders[r].setValue([
                self.saturation[r][self.currentZ][0],
                self.saturation[r][self.currentZ][1]
            ])
        self.win.show()
        self.show()

    def update_layer(self):
        if self.masksOn or self.outlinesOn:
            #self.draw_layer()
            self.layer.setImage(self.layerz, autoLevels=False)
        self.update_roi_count()
        self.win.show()
        self.show()

    def update_roi_count(self):
        self.roi_count.setText(f"{self.ncells} ROIs")

    def add_set(self):
        if len(self.current_point_set) > 0:
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set[0]) > 8:
                color = self.colormap[self.ncells, :3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :],
                                                axis=0)
                    self.ncells += 1
                    self.ismanual = np.append(self.ismanual, True)
                    if self.NZ == 1:
                        # only save after each cell if single image
                        io._save_sets_with_check(self)
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_layer()

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        # points is list of strokes
        points_all = np.concatenate(points, axis=0)

        # loop over z values
        median = []
        zdraw = np.unique(points_all[:, 0])
        z = 0
        ars, acs, vrs, vcs = np.zeros(0, "int"), np.zeros(0, "int"), np.zeros(
            0, "int"), np.zeros(0, "int")
        for stroke in points:
            stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
            vr = stroke[:, 1]
            vc = stroke[:, 2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2),
                           axis=-1)[:, np.newaxis, :]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0].squeeze().T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 8:
                print("ERROR: cell too small without overlaps, not drawn")
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0].squeeze().T
                vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
            ars = np.concatenate((ars, ar), axis=0)
            acs = np.concatenate((acs, ac), axis=0)
            vrs = np.concatenate((vrs, vr), axis=0)
            vcs = np.concatenate((vcs, vc), axis=0)

        self.draw_mask(z, ars, acs, vrs, vcs, color)
        median.append(np.array([np.median(ars), np.median(acs)]))

        self.zdraw.append(zdraw)
        d = datetime.datetime.now()
        self.track_changes.append(
            [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ar, ac]])
        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        """ draw single mask using outlines and area """
        if idx is None:
            idx = self.ncells + 1
        self.cellpix[z, vr, vc] = idx
        self.cellpix[z, ar, ac] = idx
        self.outpix[z, vr, vc] = idx
        if self.restore and "upsample" in self.restore:
            if self.resize:
                self.cellpix_resize[z, vr, vc] = idx
                self.cellpix_resize[z, ar, ac] = idx
                self.outpix_resize[z, vr, vc] = idx
                self.cellpix_orig[z, (vr / self.ratio).astype(int),
                                  (vc / self.ratio).astype(int)] = idx
                self.cellpix_orig[z, (ar / self.ratio).astype(int),
                                  (ac / self.ratio).astype(int)] = idx
                self.outpix_orig[z, (vr / self.ratio).astype(int),
                                 (vc / self.ratio).astype(int)] = idx
            else:
                self.cellpix_orig[z, vr, vc] = idx
                self.cellpix_orig[z, ar, ac] = idx
                self.outpix_orig[z, vr, vc] = idx

                # get upsampled mask
                vrr = (vr.copy() * self.ratio).astype(int)
                vcr = (vc.copy() * self.ratio).astype(int)
                mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
                pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2),
                               axis=-1)[:, np.newaxis, :]
                mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
                arr, acr = np.nonzero(mask)
                arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
                # get dense outline
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0].squeeze().T
                vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
                # concatenate all points
                arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
                self.cellpix_resize[z, vrr, vcr] = idx
                self.cellpix_resize[z, arr, acr] = idx
                self.outpix_resize[z, vrr, vcr] = idx

        if z == self.currentZ:
            self.layerz[ar, ac, :3] = color
            if self.masksOn:
                self.layerz[ar, ac, -1] = self.opacity
            if self.outlinesOn:
                self.layerz[vr, vc] = np.array(self.outcolor)

    def compute_scale(self):
        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))
        self.radii_padding = int(self.pr * 1.25)
        self.radii = np.zeros((self.Ly + self.radii_padding, self.Lx, 4), np.uint8)
        yy, xx = disk([self.Ly + self.radii_padding / 2 - 1, self.pr / 2 + 1],
                      self.pr / 2, self.Ly + self.radii_padding, self.Lx)
        # rgb(150,50,150)
        self.radii[yy, xx, 0] = 150
        self.radii[yy, xx, 1] = 50
        self.radii[yy, xx, 2] = 150
        self.radii[yy, xx, 3] = 255
        self.p0.setYRange(0, self.Ly + self.radii_padding)
        self.p0.setXRange(0, self.Lx)

    def update_scale(self):
        self.compute_scale()
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        self.win.show()
        self.show()

    def redraw_masks(self, masks=True, outlines=True, draw=True):
        self.draw_layer()

    def draw_masks(self):
        self.draw_layer()

    def draw_layer(self):
        if self.resize:
            self.Ly, self.Lx = self.Lyr, self.Lxr
        else:
            self.Ly, self.Lx = self.Ly0, self.Lx0

        if self.masksOn or self.outlinesOn:
            if self.restore and "upsample" in self.restore:
                if self.resize:
                    self.cellpix = self.cellpix_resize.copy()
                    self.outpix = self.outpix_resize.copy()
                else:
                    self.cellpix = self.cellpix_orig.copy()
                    self.outpix = self.outpix_orig.copy()

        #print(self.cellpix.shape, self.outpix.shape, self.cellpix.max(), self.outpix.max())
        self.layerz = np.zeros((self.Ly, self.Lx, 4), np.uint8)
        if self.masksOn:
            self.layerz[..., :3] = self.cellcolors[self.cellpix[self.currentZ], :]
            self.layerz[..., 3] = self.opacity * (self.cellpix[self.currentZ]
                                                  > 0).astype(np.uint8)
            if self.selected > 0:
                self.layerz[self.cellpix[self.currentZ] == self.selected] = np.array(
                    [255, 255, 255, self.opacity])
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:, 1], stroke[:,
                                                     2]] = np.array([255, 0, 255, 100])
        else:
            self.layerz[..., 3] = 0

        if self.outlinesOn:
            self.layerz[self.outpix[self.currentZ] > 0] = np.array(
                self.outcolor).astype(np.uint8)

    def set_restore_button(self):
        keys = self.denoise_text
        for i, key in enumerate(keys):
            if key != "none" and (self.restore and key in self.restore):
                self.DenoiseButtons[i].setStyleSheet(self.stylePressed)
            elif key == "none" and self.restore is None:
                self.DenoiseButtons[i].setStyleSheet(self.stylePressed)
            else:
                if self.DenoiseButtons[i].isEnabled():
                    self.DenoiseButtons[i].setStyleSheet(self.styleUnpressed)

    def set_normalize_params(self, normalize_params):
        from cellpose.models import normalize_default
        if self.restore != "filter":
            keys = list(normalize_params.keys()).copy()
            for key in keys:
                if key != "percentile":
                    normalize_params[key] = normalize_default[key]
        normalize_params = {**normalize_default, **normalize_params}
        percentile = self.check_percentile_params(normalize_params["percentile"])
        out = self.check_filter_params(normalize_params["sharpen_radius"],
                                       normalize_params["smooth_radius"],
                                       normalize_params["tile_norm_blocksize"],
                                       normalize_params["tile_norm_smooth3D"],
                                       normalize_params["norm3D"],
                                       normalize_params["invert"])

    def check_percentile_params(self, percentile):
        # check normalization params
        if percentile is not None and not (percentile[0] >= 0 and percentile[1] > 0 and
                                           percentile[0] < 100 and percentile[1] <= 100
                                           and percentile[1] > percentile[0]):
            print(
                "GUI_ERROR: percentiles need be between 0 and 100, and upper > lower, using defaults"
            )
            self.norm_edits[0].setText("1.")
            self.norm_edits[1].setText("99.")
            percentile = [1., 99.]
        elif percentile is None:
            percentile = [1., 99.]
        self.norm_edits[0].setText(str(percentile[0]))
        self.norm_edits[1].setText(str(percentile[1]))
        return percentile

    def check_filter_params(self, sharpen, smooth, tile_norm, smooth3D, norm3D, invert):
        tile_norm = 0 if tile_norm < 0 else tile_norm
        sharpen = 0 if sharpen < 0 else sharpen
        smooth = 0 if smooth < 0 else smooth
        smooth3D = 0 if smooth3D < 0 else smooth3D
        norm3D = bool(norm3D)
        invert = bool(invert)
        if tile_norm > self.Ly and tile_norm > self.Lx:
            print(
                "GUI_ERROR: tile size (tile_norm) bigger than both image dimensions, disabling"
            )
            tile_norm = 0
        self.filt_edits[0].setText(str(sharpen))
        self.filt_edits[1].setText(str(smooth))
        self.filt_edits[2].setText(str(tile_norm))
        self.filt_edits[3].setText(str(smooth3D))
        self.norm3D_cb.setChecked(norm3D)
        self.invert_cb.setChecked(invert)
        return sharpen, smooth, tile_norm, smooth3D, norm3D, invert

    def get_normalize_params(self):
        percentile = [
            float(self.norm_edits[0].text()),
            float(self.norm_edits[1].text())
        ]
        self.check_percentile_params(percentile)
        normalize_params = {"percentile": percentile}
        norm3D = self.norm3D_cb.isChecked()
        normalize_params["norm3D"] = norm3D
        if self.restore == "filter":
            sharpen = float(self.filt_edits[0].text())
            smooth = float(self.filt_edits[1].text())
            tile_norm = float(self.filt_edits[2].text())
            smooth3D = float(self.filt_edits[3].text())
            invert = self.invert_cb.isChecked()
            out = self.check_filter_params(sharpen, smooth, tile_norm, smooth3D, norm3D,
                                           invert)
            sharpen, smooth, tile_norm, smooth3D, norm3D, invert = out
            normalize_params["sharpen_radius"] = sharpen
            normalize_params["smooth_radius"] = smooth
            normalize_params["tile_norm_blocksize"] = tile_norm
            normalize_params["tile_norm_smooth3D"] = smooth3D
            normalize_params["invert"] = invert

        from cellpose.models import normalize_default
        normalize_params = {**normalize_default, **normalize_params}

        return normalize_params

    def compute_saturation(self, return_img=False):
        norm = self.get_normalize_params()
        print(norm)
        sharpen, smooth = norm["sharpen_radius"], norm["smooth_radius"]
        percentile = norm["percentile"]
        tile_norm = norm["tile_norm_blocksize"]
        invert = norm["invert"]
        norm3D = norm["norm3D"]
        smooth3D = norm["tile_norm_smooth3D"]
        tile_norm = norm["tile_norm_blocksize"]

        # if grayscale, use gray img
        channels = self.get_channels()
        if channels[0] == 0:
            img_norm = self.stack.mean(axis=-1, keepdims=True)
        elif sharpen > 0 or smooth > 0 or tile_norm > 0:
            img_norm = self.stack.copy()
        else:
            img_norm = self.stack

        if sharpen > 0 or smooth > 0 or tile_norm > 0:
            self.clear_restore()
            self.restore = "filter"
            print(
                "GUI_INFO: computing filtered image because sharpen > 0 or tile_norm > 0"
            )
            print(
                "GUI_WARNING: will use memory to create filtered image -- make sure to have RAM for this"
            )
            img_norm = self.stack.copy()
            if sharpen > 0 or smooth > 0:
                img_norm = smooth_sharpen_img(self.stack, sharpen_radius=sharpen,
                                              smooth_radius=smooth)

            if tile_norm > 0:
                img_norm = normalize99_tile(img_norm, blocksize=tile_norm,
                                            lower=percentile[0], upper=percentile[1],
                                            smooth3D=smooth3D, norm3D=norm3D)
            # convert to 0->255
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= (img_norm_max - img_norm_min)
            img_norm *= 255
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                           1).setEnabled(True)
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)
        elif invert:
            img_norm = self.stack.copy()
        else:
            img_norm = self.stack if self.restore is None or self.restore == "filter" else self.stack_filtered

        self.saturation = []
        for c in range(img_norm.shape[-1]):
            self.saturation.append([])
            if np.ptp(img_norm[..., c]) > 1e-3:
                if norm3D:
                    x01 = np.percentile(img_norm[..., c], percentile[0])
                    x99 = np.percentile(img_norm[..., c], percentile[1])
                    if invert:
                        x01i = 255. - x99
                        x99i = 255. - x01
                        x01, x99 = x01i, x99i
                    for n in range(self.NZ):
                        self.saturation[-1].append([x01, x99])
                else:
                    for z in range(self.NZ):
                        if self.NZ > 1:
                            x01 = np.percentile(img_norm[z, :, :, c], percentile[0])
                            x99 = np.percentile(img_norm[z, :, :, c], percentile[1])
                        else:
                            x01 = np.percentile(img_norm[..., c], percentile[0])
                            x99 = np.percentile(img_norm[..., c], percentile[1])
                        if invert:
                            x01i = 255. - x99
                            x99i = 255. - x01
                            x01, x99 = x01i, x99i
                        self.saturation[-1].append([x01, x99])
            else:
                for n in range(self.NZ):
                    self.saturation[-1].append([0, 255.])
        # if only 2 restore channels, add blue
        if len(self.saturation) < 3:
            for i in range(3 - len(self.saturation)):
                self.saturation.append([])
                for n in range(self.NZ):
                    self.saturation[-1].append([0, 255.])
        print(self.saturation[2][self.currentZ])

        if invert:
            img_norm = 255. - img_norm
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                           1).setEnabled(True)
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)

        if img_norm.shape[-1] == 1:
            self.saturation.append(self.saturation[0])
            self.saturation.append(self.saturation[0])

        self.autobtn.setChecked(True)
        self.update_plot()

    def chanchoose(self, image):
        if image.ndim > 2 and self.nchan > 1:
            if self.ChannelChoose[0].currentIndex() == 0:
                return image.mean(axis=-1, keepdims=True)
            else:
                chanid = [self.ChannelChoose[0].currentIndex() - 1]
                if self.ChannelChoose[1].currentIndex() > 0:
                    chanid.append(self.ChannelChoose[1].currentIndex() - 1)
                return image[:, :, chanid]
        else:
            return image

    def get_model_path(self, custom=False):
        if custom:
            self.current_model = self.ModelChooseC.currentText()
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model))
        else:
            self.current_model = self.net_names[max(
                0,
                self.ModelChooseB.currentIndex() - 1)]
            self.current_model_path = models.model_path(self.current_model)

    def initialize_model(self, model_name=None, custom=False):
        if model_name == "dataset-specific models":
            raise ValueError("need to specify model (use dropdown)")
        elif model_name is None or custom:
            self.get_model_path(custom=custom)
            if not os.path.exists(self.current_model_path):
                raise ValueError("need to specify model (use dropdown)")

        if model_name is None or not isinstance(model_name, str):
            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                              pretrained_model=self.current_model_path)
        else:
            self.current_model = model_name
            if self.current_model == "cyto" or self.current_model == "nuclei":
                self.current_model_path = models.model_path(self.current_model, 0)
            else:
                self.current_model_path = os.fspath(
                    models.MODEL_DIR.joinpath(self.current_model))

            if self.current_model != "cyto3":
                diam_mean = 17. if self.current_model == "nuclei" else 30.
                self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                                  diam_mean=diam_mean,
                                                  model_type=self.current_model)
            else:
                self.model = models.Cellpose(gpu=self.useGPU.isChecked(),
                                             model_type=self.current_model)

    def add_model(self):
        io._add_model(self)
        return

    def remove_model(self):
        io._remove_model(self)
        return

    def new_model(self):
        if self.NZ != 1:
            print("ERROR: cannot train model on 3D data")
            return

        # train model
        image_names = self.get_files()[0]
        self.train_data, self.train_labels, self.train_files, restore, normalize_params = io._get_train_set(
            image_names)
        TW = guiparts.TrainWindow(self, models.MODEL_NAMES)
        train = TW.exec_()
        if train:
            self.logger.info(
                f"training with {[os.path.split(f)[1] for f in self.train_files]}")
            self.train_model(restore=restore, normalize_params=normalize_params)
        else:
            print("GUI_INFO: training cancelled")

    def train_model(self, restore=None, normalize_params=None):
        from cellpose.models import normalize_default
        if normalize_params is None:
            normalize_params = copy.deepcopy(normalize_default)
        if self.training_params["model_index"] < len(models.MODEL_NAMES):
            model_type = models.MODEL_NAMES[self.training_params["model_index"]]
            self.logger.info(f"training new model starting at model {model_type}")
        else:
            model_type = None
            self.logger.info(f"training new model starting from scratch")
        self.current_model = model_type
        self.channels = self.get_channels()
        self.logger.info(
            f"training with chan = {self.ChannelChoose[0].currentText()}, chan2 = {self.ChannelChoose[1].currentText()}"
        )

        self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                          model_type=model_type)
        self.SizeButton.setEnabled(False)
        save_path = os.path.dirname(self.filename)

        print("GUI_INFO: name of new model: " + self.training_params["model_name"])
        self.new_model_path, train_losses = train.train_seg(
            self.model.net, train_data=self.train_data, train_labels=self.train_labels,
            channels=self.channels, normalize=normalize_params, min_train_masks=0,
            save_path=save_path, nimg_per_epoch=max(8, len(self.train_data)), SGD=True,
            learning_rate=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
            n_epochs=self.training_params["n_epochs"],
            model_name=self.training_params["model_name"])[:2]
        # save train losses
        np.save(str(self.new_model_path) + "_train_losses.npy", train_losses)
        # run model on next image
        io._add_model(self, self.new_model_path)
        diam_labels = self.model.net.diam_labels.item()  #.copy()
        self.new_model_ind = len(self.model_strings)
        self.autorun = True
        channels = self.channels.copy()
        self.clear_all()
        # keep same channels
        self.ChannelChoose[0].setCurrentIndex(channels[0])
        self.ChannelChoose[1].setCurrentIndex(channels[1])
        self.diameter = diam_labels
        self.Diameter.setText("%0.2f" % self.diameter)
        self.logger.info(f">>>> diameter set to diam_labels ( = {diam_labels: 0.3f} )")
        self.restore = restore
        self.set_normalize_params(normalize_params)
        self.get_next_image(load_seg=True)

        self.compute_segmentation(custom=True)
        self.logger.info(
            f"!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!"
        )

    def compute_restore(self):
        if self.restore:
            self.logger.info(f"running image restoration {self.restore}")
            if self.restore != "filter":
                rstr = self.restore.split("_")
                model_type = rstr[0]
                if len(rstr) > 1:
                    dset = rstr[1]
                    if dset == "cyto3":
                        self.DenoiseChoose.setCurrentIndex(0)
                    else:
                        self.DenoiseChoose.setCurrentIndex(1)
                if "upsample" in self.restore:
                    i = self.DenoiseChoose.currentIndex()
                    diam_up = 30. if i==0 or i==1 else 17.
                    print(diam_up, self.ratio)
                    self.Diameter.setText(str(diam_up / self.ratio))
                self.compute_denoise_model(model_type=model_type)
            else:
                self.compute_saturation()

    def get_thresholds(self):
        try:
            flow_threshold = float(self.flow_threshold.text())
            cellprob_threshold = float(self.cellprob_threshold.text())
            if flow_threshold == 0.0 or self.NZ > 1:
                flow_threshold = None
            return flow_threshold, cellprob_threshold
        except Exception as e:
            print(
                "flow threshold or cellprob threshold not a valid number, setting to defaults"
            )
            self.flow_threshold.setText("0.4")
            self.cellprob_threshold.setText("0.0")
            return 0.4, 0.0

    def compute_cprob(self):
        if self.recompute_masks:
            flow_threshold, cellprob_threshold = self.get_thresholds()
            if flow_threshold is None:
                self.logger.info(
                    "computing masks with cell prob=%0.3f, no flow error threshold" %
                    (cellprob_threshold))
            else:
                self.logger.info(
                    "computing masks with cell prob=%0.3f, flow error threshold=%0.3f" %
                    (cellprob_threshold, flow_threshold))
            maski = dynamics.resize_and_compute_masks(
                self.flows[4][:-1], self.flows[4][-1], p=self.flows[3].copy(),
                cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold,
                resize=self.cellpix.shape[-2:])[0]

            self.masksOn = True
            if not self.OCheckBox.isChecked():
                self.MCheckBox.setChecked(True)
            if maski.ndim < 3:
                maski = maski[np.newaxis, ...]
            self.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
            io._masks_to_gui(self, maski, outlines=None)
            self.show()

    def compute_denoise_model(self, model_type=None):
        self.progress.setValue(0)
        try:
            tic = time.time()
            nstr = self.DenoiseChoose.currentText()
            nstr.replace("-", "")
            self.clear_restore()
            model_name = model_type + "_" + nstr
            print(model_name)
            # denoising model
            self.denoise_model = denoise.DenoiseModel(gpu=self.useGPU.isChecked(),
                                                      model_type=model_name)
            self.progress.setValue(10)
            diam_up = 30. if "cyto" in model_name else 17.

            # params
            channels = self.get_channels()
            self.diameter = float(self.Diameter.text())
            normalize_params = self.get_normalize_params()
            print("GUI_INFO: channels: ", channels)
            print("GUI_INFO: normalize_params: ", normalize_params)
            print("GUI_INFO: diameter (before upsampling): ", self.diameter)

            data = self.stack.copy()
            print(data.shape)
            self.Ly, self.Lx = data.shape[-3:-1]
            if "upsample" in model_name:
                # get upsampling factor
                if self.diameter >= diam_up:
                    print(
                        f"GUI_ERROR: cannot upsample, already set to pixel diameter >= {diam_up}"
                    )
                    self.progress.setValue(0)
                    return
                self.ratio = diam_up / self.diameter
                print(
                    "GUI_WARNING: upsampling image, this will also duplicate mask layer and resize it, will use more RAM"
                )
                print(
                    f"GUI_INFO: upsampling image to {diam_up} pixel diameter ({self.ratio:0.2f} times)"
                )
                self.Lyr, self.Lxr = int(self.Ly * self.ratio), int(self.Lx *
                                                                    self.ratio)
                self.Ly0, self.Lx0 = self.Ly, self.Lx
                # moved resize into eval
                #data = resize_image(data, Ly=self.Lyr, Lx=self.Lxr)
                #self.diameter = diam_up
                #self.Diameter.setText(str(diam_up))
            else:
                self.Lyr, self.Lxr = self.Ly, self.Lx
                self.Ly0, self.Lx0 = self.Ly, self.Lx
                diam_up = self.diameter

            img_norm = self.denoise_model.eval(data, channels=channels, z_axis=0,
                                               channel_axis=3, diameter=self.diameter,
                                               normalize=normalize_params)
            print(img_norm.shape)
            self.diameter = diam_up
            self.Diameter.setText(str(diam_up))

            if img_norm.ndim == 2:
                img_norm = img_norm[:, :, np.newaxis]
            if img_norm.ndim == 3:
                img_norm = img_norm[np.newaxis, ...]

            self.progress.setValue(100)
            self.logger.info(f"{model_name} finished in %0.3f sec" %
                             (time.time() - tic))

            # compute saturation
            percentile = normalize_params["percentile"]
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            chan = [0] if channels[0] == 0 else [channels[0] - 1, channels[1] - 1]
            self.saturation = [[], [], []]
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= (img_norm_max - img_norm_min)
                for z in range(self.NZ):
                    x01 = np.percentile(img_norm[z, :, :, c], percentile[0]) * 255.
                    x99 = np.percentile(img_norm[z, :, :, c], percentile[1]) * 255.
                    self.saturation[chan[c]].append([x01, x99])
            notchan = np.ones(3, "bool")
            notchan[np.array(chan)] = False
            notchan = np.nonzero(notchan)[0]
            for c in notchan:
                for z in range(self.NZ):
                    self.saturation[c].append([0, 255.])

            img_norm *= 255.
            self.autobtn.setChecked(True)

            # assign to denoised channels
            self.stack_filtered = np.zeros(
                (self.NZ, self.Lyr, self.Lxr, self.stack.shape[-1]), "float32")
            for i, c in enumerate(chan[:img_norm.shape[-1]]):
                for z in range(self.NZ):
                    self.stack_filtered[z, :, :, c] = img_norm[z, :, :, i]

            # make upsampled masks
            if model_type == "upsample":
                self.cellpix_orig = self.cellpix.copy()
                self.outpix_orig = self.outpix.copy()
                self.cellpix_resize = cv2.resize(
                    self.cellpix_orig[0], (self.Lxr, self.Lyr),
                    interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
                outlines = masks_to_outlines(self.cellpix_resize[0])[np.newaxis, :, :]
                self.outpix_resize = outlines * self.cellpix_resize

            self.restore = model_name

            # draw plot
            if model_type == "upsample":
                self.resize = True
            else:
                self.resize = False
            self.draw_layer()
            self.update_layer()
            self.update_scale()
            # if denoised in grayscale, show in grayscale
            if channels[0] == 0:
                self.RGBDropDown.setCurrentIndex(4)

            self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                           1).setEnabled(True)
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)

            self.update_plot()

        except Exception as e:
            print("ERROR: %s" % e)

    def compute_segmentation(self, custom=False, model_name=None, load_model=True):
        self.progress.setValue(0)
        try:
            tic = time.time()
            self.clear_all()
            self.flows = [[], [], []]
            if load_model:
                self.initialize_model(model_name=model_name, custom=custom)
            self.progress.setValue(10)
            do_3D = self.load_3D
            stitch_threshold = float(self.stitch_threshold.text()) if not isinstance(
                self.stitch_threshold, float) else self.stitch_threshold
            do_3D = False if stitch_threshold > 0. else do_3D

            channels = self.get_channels()
            if self.restore is not None and self.restore != "filter":
                data = self.stack_filtered.copy().squeeze()
            else:
                data = self.stack.copy().squeeze()
            flow_threshold, cellprob_threshold = self.get_thresholds()
            self.diameter = float(self.Diameter.text())
            niter = max(0, int(self.niter.text()))
            niter = None if niter == 0 else niter
            normalize_params = self.get_normalize_params()
            print(normalize_params)
            try:
                masks, flows = self.model.eval(
                    data, channels=channels, diameter=self.diameter,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, do_3D=do_3D, niter=niter,
                    normalize=normalize_params, stitch_threshold=stitch_threshold,
                    progress=self.progress)[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)

            # convert flows to uint8 and resize to original image size
            flows_new = []
            flows_new.append(flows[0].copy())  # RGB flow
            flows_new.append((np.clip(normalize99(flows[2].copy()), 0, 1) *
                              255).astype("uint8"))  # cellprob
            if self.load_3D:
                if stitch_threshold == 0.:
                    flows_new.append((flows[1][0] / 10 * 127 + 127).astype("uint8"))
                else:
                    flows_new.append(np.zeros(flows[1][0].shape, dtype="uint8"))

            if self.restore and "upsample" in self.restore:
                self.Ly, self.Lx = self.Lyr, self.Lxr

            if flows_new[0].shape[-3:-1] != (self.Ly, self.Lx):
                self.flows = []
                for j in range(len(flows_new)):
                    self.flows.append(
                        resize_image(flows_new[j], Ly=self.Ly, Lx=self.Lx,
                                     interpolation=cv2.INTER_NEAREST))
            else:
                self.flows = flows_new

            # add first axis
            if self.NZ == 1:
                masks = masks[np.newaxis, ...]
                self.flows = [
                    self.flows[n][np.newaxis, ...] for n in range(len(self.flows))
                ]

            self.logger.info("%d cells found with model in %0.3f sec" %
                             (len(np.unique(masks)[1:]), time.time() - tic))
            self.progress.setValue(80)
            z = 0

            io._masks_to_gui(self, masks, outlines=None)
            self.masksOn = True
            self.MCheckBox.setChecked(True)
            self.progress.setValue(100)
            if self.restore != "filter" and self.restore is not None:
                self.compute_saturation()
            if not do_3D and not stitch_threshold > 0:
                self.recompute_masks = True
            else:
                self.recompute_masks = False
        except Exception as e:
            print("ERROR: %s" % e)
