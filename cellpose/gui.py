import sys, os, warnings, datetime, tempfile, glob
from natsort import natsorted
from tqdm import tqdm

from PyQt5 import QtGui, QtCore, Qt, QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
import matplotlib.pyplot as plt

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from skimage import io
from skimage import transform, draw, measure, segmentation

import mxnet as mx
from mxnet import nd

from . import utils, transforms, models, guiparts, plot

try:
    from google.cloud import storage
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        'key/cellpose-data-writer.json')
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False


class QHLine(QtGui.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtGui.QFrame.HLine)
        self.setFrameShadow(QtGui.QFrame.Sunken)

def make_bwr():
    # make a bwr colormap
    b = np.append(255*np.ones(128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255*np.ones(128))[:,np.newaxis]
    g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
    color = np.concatenate((r,g,b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
    return bwr

def make_cmap(cm=0):
    # make a single channel colormap
    r = np.arange(0,256)
    color = np.zeros((256,3))
    color[:,cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
    return cmap

def run(zstack=None, images=None):
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QtGui.QApplication(sys.argv)
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'logo/logo.png'
    )
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app.setWindowIcon(app_icon)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    models.download_model_weights()
    MainW(zstack=zstack, images=images)
    ret = app.exec_()
    sys.exit(ret)

def get_unique_points(set):
    cps = np.zeros((len(set),3), np.int32)
    for k,pp in enumerate(set):
        cps[k,:] = np.array(pp)
    set = list(np.unique(cps, axis=0))
    return set

class MainW(QtGui.QMainWindow):
    def __init__(self, zstack=None, images=None):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle("cellpose")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = os.path.abspath(os.path.join(
            self.cp_path, "logo/logo.png")
        )
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        self.setWindowIcon(app_icon)

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")
        # load processed data
        loadImg = QtGui.QAction("&Load image (*.tif, *.png, *.jpg)", self)
        loadImg.setShortcut("Ctrl+L")
        loadImg.triggered.connect(lambda: self.load_images(images))
        file_menu.addAction(loadImg)

        self.loadMasks = QtGui.QAction("Load &masks (*.tif, *.png, *.jpg)", self)
        self.loadMasks.setShortcut("Ctrl+M")
        self.loadMasks.triggered.connect(lambda: self.load_masks(None))
        file_menu.addAction(self.loadMasks)
        self.loadMasks.setEnabled(False)

        loadManual = QtGui.QAction("Load &processed/labelled image (*_seg.npy)", self)
        loadManual.setShortcut("Ctrl+P")
        loadManual.triggered.connect(lambda: self.load_manual(None))
        file_menu.addAction(loadManual)

        loadStack = QtGui.QAction("Load &numpy z-stack (*.npy nimgs x nchan x pixels x pixels)", self)
        loadStack.setShortcut("Ctrl+N")
        loadStack.triggered.connect(lambda: self.load_zstack(None))
        file_menu.addAction(loadStack)

        self.saveSet = QtGui.QAction("&Save masks and images (as *.npy)", self)
        self.saveSet.setShortcut("Ctrl+S")
        self.saveSet.triggered.connect(self.save_sets)
        file_menu.addAction(self.saveSet)
        self.saveSet.setEnabled(False)

        self.saveServer = QtGui.QAction("Send manually labelled data to server", self)
        self.saveServer.triggered.connect(self.save_server)
        file_menu.addAction(self.saveServer)
        self.saveServer.setEnabled(False)

        edit_menu = main_menu.addMenu("&Edit")
        self.undo = QtGui.QAction('Undo previous mask/trace', self)
        self.undo.setShortcut("Ctrl+Z")
        self.undo.triggered.connect(self.undo_action)
        self.undo.setEnabled(False)
        edit_menu.addAction(self.undo)

        self.ClearButton = QtGui.QAction('Clear all masks', self)
        self.ClearButton.setShortcut("Ctrl+0")
        self.ClearButton.triggered.connect(self.clear_all)
        self.ClearButton.setEnabled(False)
        edit_menu.addAction(self.ClearButton)

        self.remcell = QtGui.QAction('Remove selected cell (Ctrl+CLICK)', self)
        self.remcell.setShortcut("Ctrl+Click")
        self.remcell.triggered.connect(self.remove_action)
        self.remcell.setEnabled(False)
        edit_menu.addAction(self.remcell)

        help_menu = main_menu.addMenu("&Help")
        openHelp = QtGui.QAction("&Help window", self)
        openHelp.setShortcut("Ctrl+H")
        openHelp.triggered.connect(self.help_window)
        help_menu.addAction(openHelp)

        guiparts.HelpWindow()

        self.cell_types = ["cytoplasm", "membrane", "plantsy", "nucleus only",
                            "bio (other)", "miscellaneous"]

        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "border-color: white;"
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                                "border-color: white;"
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(30,30,30); "
                             "border-color: white;"
                              "color:rgb(80,80,80);}")
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)
        self.l0.setVerticalSpacing(4)

        self.imask = 0

        b = self.make_buttons()

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 0,3, b, 20)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        for i in range(3):
            self.cmap.append(make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False))

        self.colormap = (plt.get_cmap('gist_ncar')(np.linspace(0.0,.9,1000)) * 255).astype(np.uint8)
        self.reset()

        self.is_stack = True # always loading images of same FOV
        # if called with zstack / images, load them
        if zstack is not None:
            self.filename = zstack
            self.load_zstack(self.filename)
        elif images is not None:
            self.filename = images
            self.load_images(self.filename)

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    def help_window(self):
        HW = guiparts.HelpWindow(self)
        HW.show()

    def make_buttons(self):
        self.boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        self.smallfont = QtGui.QFont("Arial", 8)
        self.headings = ('color: rgb(150,255,150);')
        self.dropdowns = ("color: white;"
                        "background-color: rgb(40,40,40);"
                        "selection-color: white;"
                        "selection-background-color: rgb(50,100,50);")
        self.checkstyle = "color: rgb(190,190,190);"

        label = QtGui.QLabel('Views:')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, 0,0,1,1)

        label = QtGui.QLabel('[W/S]')
        label.setStyleSheet('color: white')
        #label.setFont(self.smallfont)
        self.l0.addWidget(label, 1,0,1,1)

        label = QtGui.QLabel('[pageup/down]')
        label.setStyleSheet('color: white')
        label.setFont(self.smallfont)
        self.l0.addWidget(label, 1,1,1,1)

        b=2
        self.view = 0 # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0 # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBChoose = guiparts.RGBRadioButtons(self, b,1)
        self.RGBDropDown = QtGui.QComboBox()
        self.RGBDropDown.addItems(["RGB","gray","red","green","blue"])
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.RGBDropDown.setFixedWidth(60)
        self.RGBDropDown.setStyleSheet(self.dropdowns)

        self.l0.addWidget(self.RGBDropDown, b,0,1,1)
        b+=3

        self.resize = -1
        self.X2 = 0

        b+=1
        line = QHLine()
        line.setStyleSheet('color: white;')
        self.l0.addWidget(line, b,0,1,2)
        b+=1
        label = QtGui.QLabel('Drawing:')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,0,1,2)

        b+=1
        self.brush_size = 3
        self.BrushChoose = QtGui.QComboBox()
        self.BrushChoose.addItems(["1","3","5","7","9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(60)
        self.BrushChoose.setStyleSheet(self.dropdowns)
        self.l0.addWidget(self.BrushChoose, b, 1,1,1)
        label = QtGui.QLabel('brush size: [, .]')
        label.setStyleSheet('color: white;')
        self.l0.addWidget(label, b,0,1,1)

        # cross-hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)

        b+=1
        # turn on draw mode
        self.SCheckBox = QtGui.QCheckBox('single stroke')
        self.SCheckBox.setStyleSheet(self.checkstyle)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.l0.addWidget(self.SCheckBox, b,0,1,2)

        b+=1
        # turn on crosshairs
        self.CHCheckBox = QtGui.QCheckBox('cross-hairs')
        self.CHCheckBox.setStyleSheet(self.checkstyle)
        self.CHCheckBox.toggled.connect(self.cross_hairs)
        self.l0.addWidget(self.CHCheckBox, b,0,1,1)

        b+=1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QtGui.QCheckBox('MASKS ON [X]')
        self.MCheckBox.setStyleSheet(self.checkstyle)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.l0.addWidget(self.MCheckBox, b,0,1,2)

        b+=1
        # turn off outlines
        self.outlinesOn = True
        self.OCheckBox = QtGui.QCheckBox('outlines on [Z]')
        self.OCheckBox.setStyleSheet(self.checkstyle)
        self.OCheckBox.setChecked(True)
        self.OCheckBox.toggled.connect(self.toggle_masks)
        self.l0.addWidget(self.OCheckBox, b,0,1,2)

        b+=1
        # send to server
        self.ServerButton = QtGui.QPushButton(' send manual seg. to server')
        self.ServerButton.clicked.connect(self.save_server)
        self.l0.addWidget(self.ServerButton, b,0,1,2)
        self.ServerButton.setEnabled(False)
        self.ServerButton.setStyleSheet(self.styleInactive)
        self.ServerButton.setFont(self.boldfont)

        b+=1
        line = QHLine()
        line.setStyleSheet('color: white;')
        self.l0.addWidget(line, b,0,1,2)
        b+=1
        label = QtGui.QLabel('Segmentation:')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,0,1,2)

        b+=1
        self.diameter = 30
        label = QtGui.QLabel('cell diameter (pix):')
        label.setStyleSheet('color: white;')
        self.l0.addWidget(label, b, 0,1,2)
        self.Diameter = QtGui.QLineEdit()
        self.Diameter.setText(str(self.diameter))
        self.Diameter.returnPressed.connect(self.compute_scale)
        self.Diameter.setFixedWidth(50)
        b+=1
        self.l0.addWidget(self.Diameter, b, 0,1,2)

        # recompute model
        self.SizeButton = QtGui.QPushButton('  calibrate')
        self.SizeButton.clicked.connect(self.calibrate_size)
        self.l0.addWidget(self.SizeButton, b,1,1,1)
        self.SizeButton.setEnabled(False)
        self.SizeButton.setStyleSheet(self.styleInactive)
        self.SizeButton.setFont(self.boldfont)

        # scale toggle
        b+=1
        self.scale_on = True
        self.ScaleOn = QtGui.QCheckBox('scale disk on')
        self.ScaleOn.setStyleSheet('color: red;')
        self.ScaleOn.setChecked(True)
        self.ScaleOn.toggled.connect(self.toggle_scale)
        self.l0.addWidget(self.ScaleOn, b,0,1,2)

        # use GPU
        b+=1
        self.useGPU = QtGui.QCheckBox('use GPU')
        self.useGPU.setStyleSheet(self.checkstyle)
        self.useGPU.setChecked(False)
        self.l0.addWidget(self.useGPU, b,0,1,2)

        b+=1
        # choose models
        self.ModelChoose = QtGui.QComboBox()
        self.model_dir = os.path.abspath(os.path.join(self.cp_path, 'models/'))
        #models = glob(self.model_dir+'/*')
        #models = [os.path.split(m)[-1] for m in models]
        models = ['cyto', 'nuclei']
        self.ModelChoose.addItems(models)
        self.ModelChoose.setFixedWidth(70)
        self.ModelChoose.setStyleSheet(self.dropdowns)
        self.l0.addWidget(self.ModelChoose, b, 1,1,1)
        label = QtGui.QLabel('model: ')
        label.setStyleSheet('color: white;')
        self.l0.addWidget(label, b, 0,1,1)

        b+=1
        # choose channel
        self.ChannelChoose = [QtGui.QComboBox(), QtGui.QComboBox()]
        self.ChannelChoose[0].addItems(['gray','red','green','blue'])
        self.ChannelChoose[1].addItems(['none','red','green','blue'])
        cstr = ['chan to seg', 'chan2 (opt)']
        for i in range(2):
            self.ChannelChoose[i].setFixedWidth(70)
            self.ChannelChoose[i].setStyleSheet(self.dropdowns)
            label = QtGui.QLabel(cstr[i])
            label.setStyleSheet('color: white;')
            self.l0.addWidget(label, b, 0,1,1)
            self.l0.addWidget(self.ChannelChoose[i], b, 1,1,1)
            b+=1

        b+=1
        # recompute model
        self.ModelButton = QtGui.QPushButton('  run segmentation')
        self.ModelButton.clicked.connect(self.compute_model)
        self.l0.addWidget(self.ModelButton, b,0,1,2)
        self.ModelButton.setEnabled(False)
        self.ModelButton.setStyleSheet(self.styleInactive)
        self.ModelButton.setFont(self.boldfont)
        b+=1
        self.progress = QtGui.QProgressBar(self)
        self.progress.setStyleSheet('color: gray;')
        self.l0.addWidget(self.progress, b,0,1,2)

        self.autobtn = QtGui.QCheckBox('auto-adjust')
        self.autobtn.setStyleSheet(self.checkstyle)
        self.autobtn.setChecked(True)
        self.l0.addWidget(self.autobtn, b+2,0,1,1)
        
        b+=1
        label = QtGui.QLabel('saturation')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,1,1,1)

        b+=1
        self.slider = guiparts.RangeSlider(self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setLow(0)
        self.slider.setHigh(255)
        self.slider.setTickPosition(QtGui.QSlider.TicksRight)
        self.l0.addWidget(self.slider, b,1,1,1)
        self.l0.setRowStretch(b, 1)

        b+=2
        # add scrollbar underneath
        self.scroll = QtGui.QScrollBar(QtCore.Qt.Horizontal)
        self.scroll.setMaximum(10)
        self.scroll.valueChanged.connect(self.move_in_Z)
        self.l0.addWidget(self.scroll, b,3,1,20)
        return b

    
    def get_channels(self):
        channels = [self.ChannelChoose[0].currentIndex(), self.ChannelChoose[1].currentIndex()]
        if self.current_model=='nuclei':
            channels[1] = 0
        return channels

    def calibrate_size(self):
        self.initialize_model()
        diams, _ = self.model.sz.eval([self.stack[self.currentZ].copy()],
                                   channels=self.get_channels(), progress=self.progress)
        diams = np.maximum(5.0, diams)
        diams *= 2 / np.pi**0.5
        print('estimated diameter of cells using %s model = %0.1f pixels'%
                (self.current_model, diams))
        self.Diameter.setText('%0.1f'%diams[0])
        self.diameter = diams[0]
        self.compute_scale()
        self.progress.setValue(100)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def keyPressEvent(self, event):
        if self.loaded:
            #self.p0.setMouseEnabled(x=True, y=True)
            if (event.modifiers() != QtCore.Qt.ControlModifier and
                event.modifiers() != QtCore.Qt.ShiftModifier and
                event.modifiers() != QtCore.Qt.AltModifier):
                if not self.in_stroke:
                    if len(self.current_point_set) > 0:
                        if event.key() == QtCore.Qt.Key_Return:
                            self.add_set()
                    else:
                        if event.key() == QtCore.Qt.Key_X:
                            self.MCheckBox.toggle()
                        if event.key() == QtCore.Qt.Key_Z:
                            self.OCheckBox.toggle()
                        if event.key() == QtCore.Qt.Key_Left:
                            self.currentZ = max(0,self.currentZ-1)
                            if self.NZ==1:
                                self.get_prev_image()
                        elif event.key() == QtCore.Qt.Key_Right:
                            self.currentZ = min(self.NZ-1, self.currentZ+1)
                            if self.NZ==1:
                                self.get_next_image()
                        elif event.key() == QtCore.Qt.Key_A:
                            self.currentZ = max(0,self.currentZ-1)
                            if self.NZ==1:
                                self.get_prev_image()
                        elif event.key() == QtCore.Qt.Key_D:
                            self.currentZ = min(self.NZ-1, self.currentZ+1)
                            if self.NZ==1:
                                self.get_next_image()
                        elif (event.key() == QtCore.Qt.Key_Comma or
                            event.key() == QtCore.Qt.Key_Period):
                            count = self.BrushChoose.count()
                            gci = self.BrushChoose.currentIndex()
                            if event.key() == QtCore.Qt.Key_Comma:
                                gci = max(0, gci-1)
                            else:
                                gci = min(count-1, gci+1)
                            self.BrushChoose.setCurrentIndex(gci)
                            self.brush_choose()
                        elif event.key() == QtCore.Qt.Key_PageDown:
                            self.view = (self.view+1)%(len(self.RGBChoose.bstr))
                            self.RGBChoose.button(self.view).setChecked(True)
                        elif event.key() == QtCore.Qt.Key_PageUp:
                            self.view = (self.view-1)%(len(self.RGBChoose.bstr))
                            self.RGBChoose.button(self.view).setChecked(True)

                    # can change background if stroke not finished
                    if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                        self.color = (self.color-1)%(5)
                        self.RGBDropDown.setCurrentIndex(self.color)
                    elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_S:
                        self.color = (self.color+1)%(5)
                        self.RGBDropDown.setCurrentIndex(self.color)
                self.update_plot()
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                if event.key() == QtCore.Qt.Key_Z:
                    self.undo_action()
                if event.key() == QtCore.Qt.Key_0:
                    self.clear_all()

    def toggle_removals(self):
        if self.ncells>0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)

    def remove_action(self):
        if self.selected>-1:
            self.remove_cell(self.selected)

    def undo_action(self):
        if (len(self.strokes) > 0 and
            self.strokes[-1][0][0]==self.currentZ):
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells> 0:
                self.remove_cell(self.ncells-1)

    def get_files(self):
        images = []
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.png'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.jpg'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.jpeg'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.tif'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.tiff'))
        images = natsorted(images)

        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]

        idx = np.nonzero(np.array(fnames)==f0)[0][0]

        #idx = np.nonzero(np.array(images)==self.filename[0])[0][0]
        return images, idx

    def get_prev_image(self):
        images, idx = self.get_files()
        idx = (idx-1)%len(images)
        #print(images[idx-1])
        self.load_images(filename=images[idx])

    def get_next_image(self):
        images, idx = self.get_files()
        idx = (idx+1)%len(images)
        #print(images[idx+1])
        self.load_images(filename=images[idx])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        print(files)
        self.load_images(filename=files[0])

    def is_cells(self):
        if self.IsCells.isChecked():
            self.iscells = True
        else:
            self.iscells = False
        self.save_sets()

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
            self.redraw_masks(masks=self.masksOn, outlines=self.outlinesOn)
        if self.loaded:
            self.update_plot()

    def move_in_Z(self):
        if self.loaded:
            self.currentZ = min(self.NZ, max(0, int(self.scroll.value())))
            self.update_plot()

    def make_viewbox(self):
        self.p0 = guiparts.ViewBoxNoRightDrag(
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True
        )
        self.brush_size=3
        self.win.addItem(self.p0, 0, 0)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0,255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0,255])
        self.p0.scene().contextMenuItem = self.p0
        #self.p0.setMouseEnabled(x=False,y=False)
        self.Ly,self.Lx = 512,512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = -1
        self.X2 = 0
        self.resize = -1
        self.onechan = False
        self.loaded = False
        self.channel = [0,1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.cells = []
        self.medians = []
        self.ncells = 0
        self.cellcolors = [np.array([255,255,255])]
        # -- set menus to default -- #
        self.color = 0
        self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        self.RGBChoose.button(self.view).setChecked(True)
        self.BrushChoose.setCurrentIndex(1)
        self.CHCheckBox.setChecked(False)
        self.SCheckBox.setChecked(True)

        # -- zero out image stack -- #
        self.opacity = 128 # how opaque masks should be
        self.outcolor = [200,200,255,200]
        self.NZ, self.Ly, self.Lx = 1,512,512
        if self.autobtn.isChecked():
            self.saturation = [[0,255] for n in range(self.NZ)]
        self.currentZ = 0
        self.flows = [[],[],[],[]]
        self.stack = np.zeros((1,self.Ly,self.Lx,3))
        self.layers = 0*np.ones((1,self.Ly,self.Lx,4), np.uint8)
        self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = -1*np.ones((1,self.Ly,self.Lx), np.int32)
        self.outpix = -1*np.ones((1,self.Ly,self.Lx), np.int32)
        self.update_plot()
        self.basename = []
        self.filename = []
        self.loaded = False

    def brush_choose(self):
        self.brush_size = self.BrushChoose.currentIndex()*2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_plot()

    def autosave_on(self):
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def cross_hairs(self):
        if self.CHCheckBox.isChecked():
            self.p0.addItem(self.vLine, ignoreBounds=True)
            self.p0.addItem(self.hLine, ignoreBounds=True)
        else:
            self.p0.removeItem(self.vLine)
            self.p0.removeItem(self.hLine)

    def clear_all(self):
        self.selected = -1
        #self.layers_undo, self.cellpix_undo, self.outpix_undo = [],[],[]
        self.layers = 0*np.ones((self.NZ,self.Ly,self.Lx,4), np.uint8)
        self.cellpix = -1*np.ones((self.NZ,self.Ly,self.Lx), np.int32)
        self.outpix = -1*np.ones((self.NZ,self.Ly,self.Lx), np.int32)
        self.cellcolors = [np.array([255,255,255])]
        self.ncells = 0
        print('removed all cells')
        self.toggle_removals()
        self.update_plot()

    def select_cell(self, idx):
        self.selected = idx
        if self.selected > -1:
            self.layers[self.cellpix==idx] = np.array([255,255,255,self.opacity])
            #if self.outlinesOn:
            #    self.layers[self.outpix==idx] = np.array(self.outcolor)
            self.update_plot()

    def unselect_cell(self):
        if self.selected > -1:
            idx = self.selected
            if idx < self.ncells:
                self.layers[self.cellpix==idx] = np.append(self.cellcolors[idx+1], self.opacity)
                if self.outlinesOn:
                    self.layers[self.outpix==idx] = np.array(self.outcolor)
                    #[0,0,0,self.opacity])
                self.update_plot()
        self.selected = -1

    def remove_cell(self, idx):
        for z in range(self.NZ):
            cp = self.cellpix[z]==idx
            op = self.outpix[z]==idx
            self.layers[z, cp] = np.array([0,0,0,0])
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = -1
            self.outpix[z, op] = -1
            # reduce other pixels by -1
            self.cellpix[z, self.cellpix[z]>idx] -= 1
            self.outpix[z, self.outpix[z]>idx] -= 1
        self.update_plot()
        del self.cellcolors[idx+1]
        self.ncells -= 1
        print('removed cell %d'%idx)
        if self.ncells==0:
            self.ClearButton.setEnabled(False)
        self.save_sets()

    def remove_stroke(self, delete_points=True):
        #self.current_stroke = get_unique_points(self.current_stroke)
        stroke = np.array(self.strokes[-1])
        cZ = stroke[0,0]
        outpix = self.outpix[cZ][stroke[:,1],stroke[:,2]]>-1
        self.layers[cZ][stroke[~outpix,1],stroke[~outpix,2]] = np.array([0,0,0,0])
        if self.masksOn:
            cellpix = (self.cellpix[cZ][stroke[:,1], stroke[:,2]]).astype(np.int32)
            ccol = np.array(self.cellcolors.copy())
            if self.selected > -1:
                ccol[self.selected+1] = np.array([255,255,255])
            col2mask = ccol[cellpix+1]
            col2mask = np.concatenate((col2mask, self.opacity*(cellpix[:,np.newaxis]>0)), axis=-1)
            self.layers[cZ][stroke[:,1], stroke[:,2], :] = col2mask
        if self.outlinesOn:
            self.layers[cZ][stroke[outpix,1],stroke[outpix,2]] = np.array(self.outcolor)
        if delete_points:
            self.current_point_set = self.current_point_set[:-1*(stroke[:,-1]==1).sum()]
        del self.strokes[-1]
        self.update_plot()

    def plot_clicked(self, event):
        if event.double():
            if event.button()==QtCore.Qt.LeftButton:
                if (event.modifiers() != QtCore.Qt.ShiftModifier and
                    event.modifiers() != QtCore.Qt.AltModifier):
                    try:
                        self.p0.setYRange(0,self.Ly+self.pr)
                    except:
                        self.p0.setYRange(0,self.Ly)
                    self.p0.setXRange(0,self.Lx)

    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)
        for x in items:
            if x==self.p0:
                mousePoint = self.p0.mapSceneToView(pos)
                if self.CHCheckBox.isChecked():
                    self.vLine.setPos(mousePoint.x())
                    self.hLine.setPos(mousePoint.y())
            #else:
            #    QtWidgets.QApplication.restoreOverrideCursor()
                #QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.DefaultCursor)


    def color_choose(self):
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.RGBChoose.button(self.view).setChecked(True)
        self.update_plot()

    def update_plot(self):
        self.scroll.setValue(self.currentZ)
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
        if self.view==0:
            image = self.stack[self.currentZ]
            if self.color==0:
                if self.onechan:
                    # show single channel
                    image = self.stack[self.currentZ][:,:,0]
                self.img.setImage(image, autoLevels=False, lut=None)
            elif self.color==1:
                image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                self.img.setImage(image, autoLevels=False, lut=None)
            elif self.color>1:
                image = image[:,:,self.color-2]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color-2])
            self.img.setLevels(self.saturation[self.currentZ])
        else:
            image = np.zeros((self.Ly,self.Lx), np.uint8)
            if hasattr(self, 'flows') and len(self.flows[self.view-1])>0:
                image = self.flows[self.view-1][self.currentZ]
            if self.view>2:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0,255.0])
        #self.img.set_ColorMap(self.bwr)
        if self.masksOn or self.outlinesOn:
            self.layer.setImage(self.layers[self.currentZ], autoLevels=False)
        self.slider.setLow(self.saturation[self.currentZ][0])
        self.slider.setHigh(self.saturation[self.currentZ][1])
        self.win.show()
        self.show()

    def add_set(self):
        if len(self.current_point_set) > 0:
            self.current_point_set = np.array(self.current_point_set)
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            col_rand = np.random.randint(1000)
            color = self.colormap[col_rand,:3]
            median = self.add_mask(points=self.current_point_set, color=color)
            if median is not None:
                self.toggle_mask_ops()
                self.cellcolors.append(color)
                self.ncells+=1
                if self.NZ==1:
                    # only save after each cell if single image
                    self.save_sets()
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_plot()

    def add_mask(self, points=None, color=(100,200,50), mask=None):
        # loop over z values
        median = []
        if points.shape[1] < 3:
            points = np.concatenate((np.zeros((points.shape[0],1), np.int32), points), axis=1)
        for z in np.unique(points[:,0]):
            iz = points[:,0] == z
            vr = points[iz,1]
            vc = points[iz,2]
            try:
                vr, vc = draw.polygon_perimeter(vr, vc, self.layers[z].shape[:2])
                if mask is None:
                    ar, ac = draw.polygon(vr, vc, self.layers[z].shape[:2])
                else:
                    ar, ac = mask
                med = np.array([np.median(ar), np.median(ac)])
                # if these pixels are overlapping with another cell, reassign them
                ioverlap = self.cellpix[z][ar, ac] > -1
                if (~ioverlap).sum() < 5:
                    print('cell too small without overlaps')
                    return
                elif ioverlap.sum() > 0:
                    ar, ac = ar[~ioverlap], ac[~ioverlap]
                    mask = np.zeros((self.Ly,self.Lx), np.int32)
                    mask[ar,ac] = 1
                    outlines = plot.masks_to_outlines(mask)
                    pix = np.array(np.nonzero(outlines)).T
                    if pix is not None:
                        vr = pix[:,0].copy()
                        vc = pix[:,1].copy()
                        self.outpix[z][vr, vc] = int(self.ncells)
                        if mask is None:
                            self.cellpix[z][vr, vc] = int(self.ncells)
                        self.cellpix[z][ar, ac] = int(self.ncells)
                else:
                    if mask is None:
                        self.cellpix[z][vr, vc] = int(self.ncells)
                    self.cellpix[z][ar, ac] = int(self.ncells)
                    self.outpix[z][vr, vc] = int(self.ncells)
                if self.masksOn:
                    self.layers[z][ar, ac, :3] = color
                    self.layers[z][ar, ac, -1] = self.opacity
                if self.outlinesOn:
                    self.layers[z][vr, vc] = np.array(self.outcolor)#np.array([0,0,0])
                median.append(np.array([np.median(ar), np.median(ac)]))
            except Exception as e:
                print('ERROR: %s'%e)
                return None
        return median

    def save_sets(self):
        if self.is_stack:
            base = os.path.splitext(self.filename)[0]
        else:
            base = os.path.splitext(self.filename[self.currentZ])[0]
        if self.NZ > 1 and self.is_stack:
            np.save(base + '_seg.npy',
                    {'outlines': self.outpix,
                     'colors': self.cellcolors[1:],
                     'masks': self.cellpix,
                     'current_channel': (self.color-2)%5,
                     'filename': self.filename})
        else:
            image = self.chanchoose(self.stack[self.currentZ].copy())
            if image.ndim < 4:
                image = image[np.newaxis,...]
            np.save(base + '_seg.npy',
                    {'outlines': self.outpix.squeeze(),
                     'colors': self.cellcolors[1:],
                     'masks': self.cellpix.squeeze(),
                     'chan_choose': [self.ChannelChoose[0].currentIndex(),
                                     self.ChannelChoose[1].currentIndex()],
                     'img': image.squeeze(),
                     'X2': self.X2,
                     'filename': self.filename,
                     'flows': self.flows})
        #print(self.point_sets)
        print('--- %d ROIs saved chan1 %s, chan2 %s'%(self.ncells,
                                                      self.ChannelChoose[0].currentText(),
                                                      self.ChannelChoose[1].currentText()))

    def save_server(self):
        """Uploads a file to the bucket."""
        q = QtGui.QMessageBox.question(
                                        self, 
                                        "Send to server", 
                                        "Are you sure? Only send complete and fully manually segmented data.\n (do not send partially automated segmentations)", 
                                        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
                                      )
        if q == QtGui.QMessageBox.Yes:
            bucket_name = 'cellpose_data'
            base = os.path.splitext(self.filename)[0]
            source_file_name = base + '_seg.npy'
            print(source_file_name)
            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
            filestring = time + '.npy'
            print(filestring)
            destination_blob_name = filestring
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            blob.upload_from_filename(source_file_name)

            print(
                "File {} uploaded to {}.".format(
                    source_file_name, destination_blob_name
                )
            )

    def initialize_images(self, image, resize, X2):
        self.onechan=False
        if image.ndim > 3:
            # tiff is Z x channels x W x H
            if image.shape[1] < 3:
                shape = image.shape
                image = np.concatenate((image,
                                np.zeros((shape[0], 3-shape[1], shape[2], shape[3]), dtype=np.uint8)), axis=1)
                if 3-shape[1]>1:
                    self.onechan=True
            image = np.transpose(image, (0,2,3,1))
        elif image.ndim==3:
            if image.shape[0] < 5:
                image = np.transpose(image, (1,2,0))

            if image.shape[-1] < 3:
                shape = image.shape
                image = np.concatenate((image,
                                           np.zeros((shape[0], shape[1], 3-shape[2]),
                                            dtype=type(image[0,0,0]))), axis=-1)
                if 3-shape[2]>1:
                    self.onechan=True
                image = image[np.newaxis,...]
            elif image.shape[-1]<5 and image.shape[-1]>2:
                image = image[:,:,:3]
                image = image[np.newaxis,...]
        else:
            image = image[np.newaxis,...]
       
        self.stack = image
        self.NZ = len(self.stack)
        self.scroll.setMaximum(self.NZ-1)
        self.layers, self.outpix, self.cellpix = [],[],[]
        if self.stack.max()>255 or self.stack.min()<0.0 or self.stack.max()<=50.0:
            self.stack = self.stack.astype(np.float32)
            self.stack -= self.stack.min()
            self.stack /= self.stack.max()
            self.stack *= 255
        del image

        self.stack = list(self.stack)
        self.orig_stack = self.stack.copy()
        for k,img in enumerate(self.stack):
            # if grayscale make 3D
            if resize != -1:
                img = utils.image_resizer(img, resize=resize, to_uint8=False)
            self.orig_stack[k] = img.copy()
            if img.ndim==2:
                img = np.tile(img[:,:,np.newaxis], (1,1,3))
                self.onechan=True
            if X2!=0:
                img = utils.X2zoom(img, X2=X2)
            self.stack[k] = img
            self.layers.append(255*np.ones((img.shape[0], img.shape[1], 4), dtype=np.uint8))
            self.layers[-1][:,:,-1] = 0 # set transparent
            self.cellpix.append(-1*np.ones(img.shape[:2], np.int32))
            self.outpix.append(-1*np.ones(img.shape[:2], np.int32))
        self.imask=0
        print(self.NZ, self.stack[0].shape)
        self.Ly, self.Lx = img.shape[0], img.shape[1]
        self.currentZ = int(np.floor(self.NZ/2))
        self.stack = np.array(self.stack)
        self.layers = np.array(self.layers)
        self.cellpix = np.array(self.cellpix)
        self.outpix = np.array(self.outpix)
        if self.autobtn.isChecked() or len(self.saturation)!=self.NZ:
            self.compute_saturation()
        self.compute_scale()

    def compute_scale(self):
        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))
        radii = np.zeros((self.Ly+self.pr,self.Lx), np.uint8)
        self.radii = np.zeros((self.Ly+self.pr,self.Lx,4), np.uint8)
        yy,xx = plot.disk([self.Ly+self.pr/2-1, self.pr/2+1], 
                            self.pr/2, self.Ly+self.pr, self.Lx)
        self.radii[yy,xx,0] = 255
        self.radii[yy,xx,-1] = 255#self.opacity * (radii>0)
        self.update_plot()
        self.p0.setYRange(0,self.Ly+self.pr)
        self.p0.setXRange(0,self.Lx)

    def redraw_masks(self, masks=True, outlines=True):
        if not outlines and masks:
            self.draw_masks()
            self.cellcolors = np.array(self.cellcolors)
            self.layers[...,:3] = self.cellcolors[self.cellpix+1,:]
            self.layers[...,3] = self.opacity * (self.cellpix>-1).astype(np.uint8)
            self.cellcolors = list(self.cellcolors)
            if self.selected>-1:
                self.layers[self.cellpix==self.selected] = np.array([255,255,255,self.opacity])
        else:
            if masks:
                self.layers[...,3] = self.opacity * (self.cellpix>-1).astype(np.uint8)
            else:
                self.layers[...,3] = 0
            self.layers[self.outpix>-1] = np.array(self.outcolor)
            #if masks and self.selected>-1:
            #    self.layers[self.outpix==self.selected] = np.array([0,0,0,self.opacity])

    def draw_masks(self):
        self.cellcolors = np.array(self.cellcolors)
        self.layers[...,:3] = self.cellcolors[self.cellpix+1,:]
        self.layers[...,3] = self.opacity * (self.cellpix>-1).astype(np.uint8)
        self.cellcolors = list(self.cellcolors)
        self.layers[self.outpix>-1] = np.array(self.outcolor)
        if self.selected>-1:
            self.layers[self.outpix==self.selected] = np.array([0,0,0,self.opacity])


    def load_manual(self, filename=None, image=None, image_file=None):
        if filename is None:
            name = QtGui.QFileDialog.getOpenFileName(
                self, "Load manual labels", filter="*_seg.npy"
                )
            filename = name[0]
        try:
            dat = np.load(filename, allow_pickle=True).item()
            dat['outlines']
            self.loaded = True
        except:
            self.loaded = False
            print('not NPY')
            return

        self.reset()
        if image is None:
            if 'filename' in dat:
                self.filename = dat['filename']
                if image is None:
                    if os.path.isfile(self.filename):
                        self.filename = dat['filename']
                    else:
                        imgname = os.path.split(self.filename)[1]
                        root = os.path.split(filename)[0]
                        self.filename = root+'/'+imgname
                try:
                    image = io.imread(self.filename)
                except:
                    self.loaded = False
                    print('ERROR: cannot find image')
                    return
            else:
                self.filename = filename[:-11]
                if image is None:
                    image = dat['img']
        else:
            self.filename = image_file
        print(self.filename)

        if 'X2' in dat:
            self.X2 = dat['X2']
        else:
            self.X2 = 0
        if 'resize' in dat:
            self.resize = dat['resize']
        elif 'img' in dat:
            if max(image.shape) > max(dat['img'].shape):
                self.resize = max(dat['img'].shape)
        else:
            self.resize = -1
        self.initialize_images(image, resize=self.resize, X2=self.X2)
        if 'chan_choose' in dat:
            self.ChannelChoose[0].setCurrentIndex(dat['chan_choose'][0])
            self.ChannelChoose[1].setCurrentIndex(dat['chan_choose'][1])
        if 'outlines' in dat:
            if isinstance(dat['outlines'], list):
                dat['outlines'] = dat['outlines'][::-1]
                for k, outline in enumerate(dat['outlines']):
                    if 'colors' in dat:
                        color = dat['colors'][k]
                    else:
                        col_rand = np.random.randint(1000)
                        color = self.colormap[col_rand,:3]
                    median = self.add_mask(points=outline, color=color)
                    if median is not None:
                        self.cellcolors.append(color)
                        self.ncells+=1
            else:
                if dat['masks'].ndim==2:
                    dat['masks'] = dat['masks'][np.newaxis,:,:]
                    dat['outlines'] = dat['outlines'][np.newaxis,:,:]
                self.cellpix = dat['masks']
                self.outpix = dat['outlines']
                self.cellcolors.extend(dat['colors'])
                self.ncells = self.cellpix.max()+1
                self.draw_masks()
                if self.masksOn or self.outlinesOn and not (self.masksOn and self.outlinesOn):
                    self.redraw_masks(masks=self.masksOn, outlines=self.outlinesOn)
            self.loaded = True
            print('%d masks found'%(self.ncells))
        else:
            self.clear_all()

        if 'current_channel' in dat:
            self.color = (dat['current_channel']+2)%5
            self.RGBDropDown.setCurrentIndex(self.color)

        self.enable_buttons()

    def compute_saturation(self):
        # compute percentiles from stack
        self.saturation = []
        for n in range(len(self.stack)):
            self.saturation.append([np.percentile(self.stack[n].astype(np.float32),1),
                                    np.percentile(self.stack[n].astype(np.float32),99)])

    def load_masks(self, filename=None):
        name = QtGui.QFileDialog.getOpenFileName(
            self, "Load masks (color channels = nucleus, cytoplasm, ...)"
            )
        masks = io.imread(name[0])
        outlines = None
        if masks.ndim>3:
            # Z x nchannels x Ly x Lx
            if masks.shape[-1]>5:
                self.flows = list(np.transpose(masks[:,:,:,2:], (3,0,1,2)))
                outlines = masks[...,1]
                masks = masks[...,0]
            else:
                self.flows = list(np.transpose(masks[:,:,:,1:], (3,0,1,2)))
                masks = masks[...,0]
        elif masks.ndim==3:
            if masks.shape[-1]<5:
                masks = masks[np.newaxis,:,:,0]
        elif masks.ndim<3:
            masks = masks[np.newaxis,:,:]
        # masks should be Z x Ly x Lx
        if masks.shape[0]!=self.NZ:
            print('ERROR: masks are not same depth (number of planes) as image stack')
            return
        print('%d masks found'%(len(np.unique(masks))-1))

        self.masks_to_gui(masks, outlines)

        self.update_plot()

    def masks_to_gui(self, masks, outlines=None):
        # get unique values
        shape = masks.shape
        _, masks = np.unique(masks, return_inverse=True)
        masks = np.reshape(masks, shape)
        self.cellpix = masks-1
        # get outlines
        if outlines is None:
            self.outpix = -1*np.ones(masks.shape, np.int32)
            for z in range(self.NZ):
                outlines = plot.masks_to_outlines(masks[z])
                self.outpix[z] = ((outlines * masks[z]) - 1).astype(np.int32)
                if z%50==0:
                    print('plane %d outlines processed'%z)
        else:
            self.outpix = outlines
            shape = self.outpix.shape
            _,self.outpix = np.unique(self.outpix, return_inverse=True)
            self.outpix = np.reshape(self.outpix, shape)
            self.outpix -= 1

        self.ncells = self.cellpix.max()+1
        colors = self.colormap[np.random.randint(0,1000,size=self.ncells), :3]
        self.cellcolors = list(np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8))
        self.draw_masks()
        if self.ncells>0:
            self.toggle_mask_ops()

    def chanchoose(self, image):
        if image.ndim > 2:
            if self.ChannelChoose[0].currentIndex()==0:
                image = image.astype(np.float32).mean(axis=-1)[...,np.newaxis]
            else:
                chanid = [self.ChannelChoose[0].currentIndex()-1]
                if self.ChannelChoose[1].currentIndex()>0:
                    chanid.append(self.ChannelChoose[1].currentIndex()-1)
                image = image[:,:,chanid].astype(np.float32)
        return image

    def initialize_model(self):
        if self.useGPU.isChecked():
            device = mx.gpu()
        else:
            device = mx.cpu()

        change=False
        if not hasattr(self, 'model') or self.ModelChoose.currentText() != self.current_model:
            self.current_model = self.ModelChoose.currentText()
            change=True
        elif (self.model.device==mx.gpu() and not self.useGPU.isChecked() or
                self.model.device==mx.cpu() and self.useGPU.isChecked()):
            # if device has changed, reload model
            self.current_model = self.ModelChoose.currentText()

        if change:
            if self.current_model=='cyto':
                szmean = 27.
            else:
                szmean = 15.
            self.model_list = ['%s_%d'%(self.current_model, i) for i in range(4)]
            cpmodel_path = [os.path.abspath(os.path.join(self.cp_path, 'models/', self.model_list[i]))
                                for i in range(len(self.model_list))]
            szmodel_path = os.path.abspath(os.path.join(self.cp_path, 'models/', 'size_%s_0.npy'%self.current_model))
            self.model = models.Cellpose(device=device,
                                        pretrained_model=cpmodel_path,
                                        pretrained_size=szmodel_path,
                                        diam_mean=szmean
                                        )

    def compute_model(self):
        self.progress.setValue(0)
        try:
            self.clear_all()
            self.flows = [[],[],[],[]]
            self.initialize_model()

            print('using model %s'%self.current_model)
            self.progress.setValue(10)
            do_3D = False
            if self.NZ > 1:
                do_3D = True
                data = self.stack.copy()
            else:
                data = self.stack[0].copy()
            channels = self.get_channels()
            self.diameter = float(self.Diameter.text())
            try:
                rescale = np.array([27/(self.diameter*(np.pi**0.5/2))])
                masks, flows, _, _ = self.model.eval([data], channels=channels,
                                                rescale=rescale,
                                                do_3D=do_3D, progress=self.progress)
            except Exception as e:
                print('NET ERROR: %s'%e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)
            masks = masks[0]
            flows = flows[0]
            self.flows[0] = flows[0][np.newaxis,...]
            self.flows[1] = (np.clip(utils.normalize99(flows[2]),0,1) * 255).astype(np.uint8)[np.newaxis,...]
            self.flows[2] = (flows[1][-1]/10 * 127 + 127).astype(np.uint8)[np.newaxis,...]

            print('%d cells found with unet'%(len(np.unique(masks)[1:])))
            self.progress.setValue(80)
            z=0
            self.masksOn = True
            self.outlinesOn = True
            self.MCheckBox.setChecked(True)
            self.OCheckBox.setChecked(True)

            self.masks_to_gui(masks[np.newaxis,:,:], outlines=None)
            self.progress.setValue(100)

            self.toggle_server(off=True)

        except Exception as e:
            print('ERROR: %s'%e)

        #self.ModelButton.setStyleSheet(self.styleUnpressed)

    def load_images(self, filename=None):
        #QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if filename is None:
            name = QtGui.QFileDialog.getOpenFileName(
                self, "Load image"
                )
            filename = name[0]
        manual_file = os.path.splitext(filename)[0]+'_seg.npy'
        if os.path.isfile(manual_file):
            print(manual_file)
            self.load_manual(manual_file, image=io.imread(filename), image_file=filename)
            return
        elif os.path.isfile(os.path.splitext(filename)[0]+'_manual.npy'):
            manual_file = os.path.splitext(filename)[0]+'_manual.npy'
            self.load_manual(manual_file, image=io.imread(filename), image_file=filename)
            return
        try:
            image = io.imread(filename)
            self.loaded = True
        except:
            print('images not compatible')

        self.prediction = False
        if self.loaded:
            self.reset()
            self.filename = filename
            print(filename)
            self.basename, filename = os.path.split(self.filename)
            #self.resize = int(self.MaxSize.text())
            self.initialize_images(image, resize=self.resize, X2=0)
            #self.stack = np.transpose(self.stack[:,:,:,0,1], (2,0,1))
            if self.prediction:
                self.compute_model()
            self.loaded = True
            self.enable_buttons()
        #QtWidgets.QApplication.restoreOverrideCursor()


    def enable_buttons(self):
        #self.X2Up.setEnabled(True)
        #self.X2Down.setEnabled(True)
        self.ModelButton.setEnabled(True)
        self.SizeButton.setEnabled(True)
        self.ModelButton.setStyleSheet(self.styleUnpressed)
        self.SizeButton.setStyleSheet(self.styleUnpressed)
        self.loadMasks.setEnabled(True)
        self.saveSet.setEnabled(True)
        self.toggle_mask_ops()
            
        self.update_plot()
        self.setWindowTitle(self.filename)

    def toggle_server(self, off=False):
        if SERVER_UPLOAD:
            if self.ncells>0 and not off:
                self.saveServer.setEnabled(True)
                self.ServerButton.setEnabled(True)
                self.ServerButton.setStyleSheet(self.styleUnpressed)
            else:
                self.saveServer.setEnabled(False)
                self.ServerButton.setEnabled(False)
                self.ServerButton.setStyleSheet(self.styleInactive)
        
    def toggle_mask_ops(self):
        self.toggle_removals()
        self.toggle_server()
            
    def load_zstack(self, filename=None):
        #QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if filename is None:
            name = QtGui.QFileDialog.getOpenFileName(
                self, "Load matrix of images", filter="*.npy"
                )
            filename = name[0]

        try:
            stack = np.load(filename)
            self.loaded = True
        except:
            print('not NPY')
            #QtWidgets.QApplication.restoreOverrideCursor()
            return

        manual_file = os.path.splitext(filename)[0]+'_seg.npy'
        if os.path.isfile(manual_file):
            print(manual_file)
            self.load_manual(manual_file, image=stack, image_file=filename)
            return

        self.prediction = False
        if self.loaded:
            self.reset()
            self.filename = filename
            print(filename)
            self.basename, filename = os.path.split(self.filename)
            #self.resize = int(self.MaxSize.text())
            self.initialize_images(stack, -1, 0)
            #self.stack = np.transpose(self.stack[:,:,:,0,1], (2,0,1))
            if self.prediction:
                self.compute_model()
            self.loaded = True
            self.enable_buttons()
        #QtWidgets.QApplication.restoreOverrideCursor()
