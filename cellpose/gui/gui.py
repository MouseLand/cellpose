import sys, os, pathlib, warnings, datetime, tempfile, glob, time
import gc
from natsort import natsorted
from tqdm import tqdm

from PyQt5 import QtGui, QtCore, Qt, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit
import pyqtgraph as pg
from pyqtgraph import GraphicsScene

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from . import guiparts, menus, io
from .. import models
from ..utils import download_url_to_file, masks_to_outlines, normalize99
from ..io import save_server
from ..transforms import resize_image
from ..plot import disk

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

try:
    from google.cloud import storage
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        'key/cellpose-data-writer.json')
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

def avg3d(C):
    """ smooth value of c across nearby points
        (c is center of grid directly below point)
        b -- a -- b
        a -- c -- a
        b -- a -- b
    """
    Ly, Lx = C.shape
    # pad T by 2
    T = np.zeros((Ly+2, Lx+2), np.float32)
    M = np.zeros((Ly, Lx), np.float32)
    T[1:-1, 1:-1] = C.copy()
    y,x = np.meshgrid(np.arange(0,Ly,1,int), np.arange(0,Lx,1,int), indexing='ij')
    y += 1
    x += 1
    a = 1./2 #/(z**2 + 1)**0.5
    b = 1./(1+2**0.5) #(z**2 + 2)**0.5
    c = 1.
    M = (b*T[y-1, x-1] + a*T[y-1, x] + b*T[y-1, x+1] +
         a*T[y, x-1]   + c*T[y, x]   + a*T[y, x+1] +
         b*T[y+1, x-1] + a*T[y+1, x] + b*T[y+1, x+1])
    M /= 4*a + 4*b + c
    return M

def interpZ(mask, zdraw):
    """ find nearby planes and average their values using grid of points
        zfill is in ascending order
    """
    ifill = np.ones(mask.shape[0], np.bool)
    zall = np.arange(0, mask.shape[0], 1, int)
    ifill[zdraw] = False
    zfill = zall[ifill]
    zlower = zdraw[np.searchsorted(zdraw, zfill, side='left')-1]
    zupper = zdraw[np.searchsorted(zdraw, zfill, side='right')]
    for k,z in enumerate(zfill):
        Z = zupper[k] - zlower[k]
        zl = (z-zlower[k])/Z
        plower = avg3d(mask[zlower[k]]) * (1-zl)
        pupper = avg3d(mask[zupper[k]]) * zl
        mask[z] = (plower + pupper) > 0.33
        #Ml, norml = avg3d(mask[zlower[k]], zl)
        #Mu, normu = avg3d(mask[zupper[k]], 1-zl)
        #mask[z] = (Ml + Mu) / (norml + normu)  > 0.5
    return mask, zfill


def make_bwr():
    # make a bwr colormap
    b = np.append(255*np.ones(128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255*np.ones(128))[:,np.newaxis]
    g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
    color = np.concatenate((r,g,b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
    return bwr

def make_spectral():
    # make spectral colormap
    r = np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95,99,103,107,111,115,119,123,127,131,135,139,143,147,151,155,159,163,167,171,175,179,183,187,191,195,199,203,207,211,215,219,223,227,231,235,239,243,247,251,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255])
    g = np.array([0,1,2,3,4,5,6,7,8,9,10,9,9,8,8,7,7,6,6,5,5,5,4,4,3,3,2,2,1,1,0,0,0,7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127,135,143,151,159,167,175,183,191,199,207,215,223,231,239,247,255,247,239,231,223,215,207,199,191,183,175,167,159,151,143,135,128,129,131,132,134,135,137,139,140,142,143,145,147,148,150,151,153,154,156,158,159,161,162,164,166,167,169,170,172,174,175,177,178,180,181,183,185,186,188,189,191,193,194,196,197,199,201,202,204,205,207,208,210,212,213,215,216,218,220,221,223,224,226,228,229,231,232,234,235,237,239,240,242,243,245,247,248,250,251,253,255,251,247,243,239,235,231,227,223,219,215,211,207,203,199,195,191,187,183,179,175,171,167,163,159,155,151,147,143,139,135,131,127,123,119,115,111,107,103,99,95,91,87,83,79,75,71,67,63,59,55,51,47,43,39,35,31,27,23,19,15,11,7,3,0,8,16,24,32,41,49,57,65,74,82,90,98,106,115,123,131,139,148,156,164,172,180,189,197,205,213,222,230,238,246,254])
    b = np.array([0,7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127,135,143,151,159,167,175,183,191,199,207,215,223,231,239,247,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,251,247,243,239,235,231,227,223,219,215,211,207,203,199,195,191,187,183,179,175,171,167,163,159,155,151,147,143,139,135,131,128,126,124,122,120,118,116,114,112,110,108,106,104,102,100,98,96,94,92,90,88,86,84,82,80,78,76,74,72,70,68,66,64,62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,16,24,32,41,49,57,65,74,82,90,98,106,115,123,131,139,148,156,164,172,180,189,197,205,213,222,230,238,246,254])
    color = (np.vstack((r,g,b)).T).astype(np.uint8)
    spectral = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
    return spectral
    

def make_cmap(cm=0):
    # make a single channel colormap
    r = np.arange(0,256)
    color = np.zeros((256,3))
    color[:,cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
    return cmap

def run(image=None):
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath('.cellpose', 'logo.png')
    guip_path = pathlib.Path.home().joinpath('.cellpose', 'cellpose_gui.png')
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath('.cellpose')
        cp_dir.mkdir(exist_ok=True)
        print('downloading logo')
        download_url_to_file('https://www.cellpose.org/static/images/cellpose_transparent.png', icon_path, progress=True)
    if not guip_path.is_file():
        download_url_to_file('https://github.com/MouseLand/cellpose/raw/master/docs/_static/cellpose_gui.png', guip_path, progress=True)
    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    models.download_model_weights()
    MainW(image=image)
    ret = app.exec_()
    sys.exit(ret)

def get_unique_points(set):
    cps = np.zeros((len(set),3), np.int32)
    for k,pp in enumerate(set):
        cps[k,:] = np.array(pp)
    set = list(np.unique(cps, axis=0))
    return set

class MainW(QMainWindow):
    def __init__(self, image=None):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle("cellpose")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath('.cellpose', 'logo.png')
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.helpmenu(self)

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
        self.cwidget = QWidget(self)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)
        self.l0.setVerticalSpacing(6)

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
        # spectral colormap
        self.cmap.append(make_spectral().getLookupTable(start=0.0, stop=255.0, alpha=False))
        # single channel colormaps
        for i in range(3):
            self.cmap.append(make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False))

        if MATPLOTLIB:
            self.colormap = (plt.get_cmap('gist_ncar')(np.linspace(0.0,.9,1000)) * 255).astype(np.uint8)
        else:
            self.colormap = ((np.random.rand(1000,3)*0.8+0.1)*255).astype(np.uint8)
        self.reset()

        self.is_stack = True # always loading images of same FOV
        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename)

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    def help_window(self):
        HW = guiparts.HelpWindow(self)
        HW.show()

    def gui_window(self):
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        label_style = """QLabel{
                            color: white
                            } 
                         QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.boldfont = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 10)
        self.smallfont = QtGui.QFont("Arial", 8)
        self.headings = ('color: rgb(150,255,150);')
        self.dropdowns = ("color: white;"
                        "background-color: rgb(40,40,40);"
                        "selection-color: white;"
                        "selection-background-color: rgb(50,100,50);")
        self.checkstyle = "color: rgb(190,190,190);"

        label = QLabel('Views:')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, 0,0,1,1)

        label = QLabel('[up/down or W/S]')
        label.setStyleSheet(label_style)
        label.setFont(self.smallfont)
        self.l0.addWidget(label, 1,0,1,1)

        label = QLabel('[pageup/down]')
        label.setStyleSheet(label_style)
        label.setFont(self.smallfont)
        self.l0.addWidget(label, 1,1,1,1)

        b=2
        self.view = 0 # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0 # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBChoose = guiparts.RGBRadioButtons(self, b,1)
        self.RGBDropDown = QComboBox()
        self.RGBDropDown.addItems(["RGB","gray","spectral","red","green","blue"])
        self.RGBDropDown.setFont(self.medfont)
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
        label = QLabel('Drawing:')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,0,1,2)

        b+=1
        self.brush_size = 3
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["1","3","5","7","9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(60)
        self.BrushChoose.setStyleSheet(self.dropdowns)
        self.BrushChoose.setFont(self.medfont)
        self.l0.addWidget(self.BrushChoose, b, 1,1,1)
        label = QLabel('brush size: [, .]')
        label.setStyleSheet(label_style)
        label.setFont(self.medfont)
        self.l0.addWidget(label, b,0,1,1)

        # cross-hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)

        b+=1
        # turn on draw mode
        self.SCheckBox = QCheckBox('single stroke')
        self.SCheckBox.setStyleSheet(self.checkstyle)
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.l0.addWidget(self.SCheckBox, b,0,1,2)

        b+=1
        # turn on crosshairs
        self.CHCheckBox = QCheckBox('cross-hairs')
        self.CHCheckBox.setStyleSheet(self.checkstyle)
        self.CHCheckBox.setFont(self.medfont)
        self.CHCheckBox.toggled.connect(self.cross_hairs)
        self.l0.addWidget(self.CHCheckBox, b,0,1,1)

        b+=1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox('MASKS ON [X]')
        self.MCheckBox.setStyleSheet(self.checkstyle)
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.l0.addWidget(self.MCheckBox, b,0,1,2)

        b+=1
        # turn off outlines
        self.outlinesOn = True
        self.OCheckBox = QCheckBox('outlines on [Z]')
        self.OCheckBox.setStyleSheet(self.checkstyle)
        self.OCheckBox.setFont(self.medfont)
        self.OCheckBox.setChecked(True)
        self.OCheckBox.toggled.connect(self.toggle_masks)
        self.l0.addWidget(self.OCheckBox, b,0,1,2)

        b+=1
        # send to server
        self.ServerButton = QPushButton(' send manual seg. to server')
        self.ServerButton.clicked.connect(lambda: save_server(self))
        self.l0.addWidget(self.ServerButton, b,0,1,2)
        self.ServerButton.setEnabled(False)
        self.ServerButton.setStyleSheet(self.styleInactive)
        self.ServerButton.setFont(self.boldfont)

        b+=1
        line = QHLine()
        line.setStyleSheet('color: white;')
        self.l0.addWidget(line, b,0,1,2)
        b+=1
        label = QLabel('Segmentation:')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,0,1,2)

        b+=1
        self.diameter = 30
        label = QLabel('cell diameter (pixels) (click ENTER):')
        label.setStyleSheet(label_style)
        label.setFont(self.medfont)
        label.setToolTip('you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)')
        self.l0.addWidget(label, b, 0,1,2)
        self.Diameter = QLineEdit()
        self.Diameter.setToolTip('you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)')
        self.Diameter.setText(str(self.diameter))
        self.Diameter.setFont(self.medfont)
        self.Diameter.returnPressed.connect(self.compute_scale)
        self.Diameter.setFixedWidth(50)
        b+=1
        self.l0.addWidget(self.Diameter, b, 0,1,2)

        # recompute model
        self.SizeButton = QPushButton('  calibrate')
        self.SizeButton.clicked.connect(self.calibrate_size)
        self.l0.addWidget(self.SizeButton, b,1,1,1)
        self.SizeButton.setEnabled(False)
        self.SizeButton.setStyleSheet(self.styleInactive)
        self.SizeButton.setFont(self.boldfont)

        # scale toggle
        b+=1
        self.scale_on = True
        self.ScaleOn = QCheckBox('scale disk on')
        self.ScaleOn.setFont(self.medfont)
        self.ScaleOn.setStyleSheet('color: red;')
        self.ScaleOn.setChecked(True)
        self.ScaleOn.setToolTip('see current diameter as red disk at bottom')
        self.ScaleOn.toggled.connect(self.toggle_scale)
        self.l0.addWidget(self.ScaleOn, b,0,1,2)

        # use GPU
        b+=1
        self.useGPU = QCheckBox('use GPU')
        self.useGPU.setStyleSheet(self.checkstyle)
        self.useGPU.setFont(self.medfont)
        self.useGPU.setToolTip('if you have specially installed the <i>cuda</i> version of mxnet, then you can activate this, but it won’t give huge speedups when running single 2D images in the GUI.')
        self.check_gpu()
        self.l0.addWidget(self.useGPU, b,0,1,1)

        # fast mode
        self.NetAvg = QComboBox()
        self.NetAvg.addItems(['average 4 nets', '+ resample (slow)', 'run 1 net (fast)', ])
        self.NetAvg.setFont(self.medfont)
        self.NetAvg.setToolTip('average 4 different fit networks (default) + resample for smooth masks (slow) or run 1 network (fast)')
        self.l0.addWidget(self.NetAvg, b,1,1,1)

        b+=1
        # choose models
        self.ModelChoose = QComboBox()
        self.model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')
        models = ['cyto', 'nuclei', 'cyto2']
        self.ModelChoose.addItems(models)
        self.ModelChoose.setFixedWidth(70)
        self.ModelChoose.setStyleSheet(self.dropdowns)
        self.ModelChoose.setFont(self.medfont)
        self.l0.addWidget(self.ModelChoose, b, 1,1,1)
        label = QLabel('model: ')
        label.setStyleSheet(label_style)
        label.setFont(self.medfont)
        label.setToolTip('there is a <em>cyto</em> model, a new <em>cyto2</em> model from user submissions, and a <em>nuclei</em> model')
        self.ModelChoose.setToolTip('there is a <em>cyto</em> model, a new <em>cyto2</em> model from user submissions, and a <em>nuclei</em> model')
        self.l0.addWidget(label, b, 0,1,1)

        b+=1
        # choose channel
        self.ChannelChoose = [QComboBox(), QComboBox()]
        self.ChannelChoose[0].addItems(['gray','red','green','blue'])
        self.ChannelChoose[1].addItems(['none','red','green','blue'])
        cstr = ['chan to segment:', 'chan2 (optional): ']
        for i in range(2):
            self.ChannelChoose[i].setFixedWidth(70)
            self.ChannelChoose[i].setStyleSheet(self.dropdowns)
            self.ChannelChoose[i].setFont(self.medfont)
            label = QLabel(cstr[i])
            label.setStyleSheet(label_style)
            label.setFont(self.medfont)
            if i==0:
                label.setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
                self.ChannelChoose[i].setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
            else:
                label.setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
                self.ChannelChoose[i].setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
            self.l0.addWidget(label, b, 0,1,1)
            self.l0.addWidget(self.ChannelChoose[i], b, 1,1,1)
            b+=1

        # use inverted image for running cellpose
        b+=1
        self.invert = QCheckBox('invert grayscale')
        self.invert.setStyleSheet(self.checkstyle)
        self.invert.setFont(self.medfont)
        self.l0.addWidget(self.invert, b,0,1,2)

        b+=1
        # recompute model
        self.ModelButton = QPushButton('  run segmentation')
        self.ModelButton.clicked.connect(self.compute_model)
        self.l0.addWidget(self.ModelButton, b,0,1,2)
        self.ModelButton.setEnabled(False)
        self.ModelButton.setStyleSheet(self.styleInactive)
        self.ModelButton.setFont(self.boldfont)
        b+=1
        self.progress = QProgressBar(self)
        self.progress.setStyleSheet('color: gray;')
        self.l0.addWidget(self.progress, b,0,1,2)

        # post-hoc paramater tuning

        b+=1
        label = QLabel('model match threshold:')
        label.setToolTip('threshold on gradient match to accept a mask (set lower to get more cells)')
        label.setStyleSheet(label_style)
        label.setFont(self.medfont)
        self.l0.addWidget(label, b, 0,1,2)

        b+=1
        self.threshold = 0.4
        self.threshslider = QSlider()
        self.threshslider.setOrientation(QtCore.Qt.Horizontal)
        self.threshslider.setMinimum(1.0)
        self.threshslider.setMaximum(30.0)
        self.threshslider.setValue(31 - 4)
        self.l0.addWidget(self.threshslider, b, 0,1,2)
        self.threshslider.valueChanged.connect(self.compute_cprob)
        self.threshslider.setStyleSheet(guiparts.horizontal_slider_style())
        self.threshslider.setEnabled(False)
        
        b+=1
        label = QLabel('cell prob threshold:')
        label.setStyleSheet(label_style)
        label.setFont(self.medfont)
        self.l0.addWidget(label, b, 0,1,2)
        label.setToolTip('cell probability threshold (set lower to get more cells)')
        
        b+=1
        self.probslider = QSlider()
        self.probslider.setOrientation(QtCore.Qt.Horizontal)
        self.probslider.setMinimum(-6.0)
        self.probslider.setMaximum(6.0)
        self.probslider.setValue(0.0)
        self.cellprob = 0.0
        self.l0.addWidget(self.probslider, b, 0,1,2)
        self.probslider.valueChanged.connect(self.compute_cprob)
        self.probslider.setStyleSheet(guiparts.horizontal_slider_style())
        self.probslider.setEnabled(False)

        b+=1
        line = QHLine()
        line.setStyleSheet('color: white;')
        self.l0.addWidget(line, b,0,1,2)

        self.autobtn = QCheckBox('auto-adjust')
        self.autobtn.setStyleSheet(self.checkstyle)
        self.autobtn.setFont(self.medfont)
        self.autobtn.setChecked(True)
        self.l0.addWidget(self.autobtn, b+2,0,1,1)

        b+=1
        label = QLabel('Image saturation:')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l0.addWidget(label, b,0,1,2)

        b+=1
        self.slider = guiparts.RangeSlider(self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setLow(0)
        self.slider.setHigh(255)
        self.slider.setTickPosition(QSlider.TicksRight)
        self.l0.addWidget(self.slider, b,1,1,1)
        self.l0.setRowStretch(b, 1)
        
        b+=2
        # add z position underneath
        self.currentZ = 0
        label = QLabel('Z:')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setStyleSheet(label_style)
        self.l0.addWidget(label, b, 0,1,1)
        self.zpos = QLineEdit()
        self.zpos.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.zpos.setText(str(self.currentZ))
        self.zpos.returnPressed.connect(self.compute_scale)
        self.zpos.setFixedWidth(60)
        self.l0.addWidget(self.zpos, b, 1,1,1)

        # add scrollbar underneath
        self.scroll = QScrollBar(QtCore.Qt.Horizontal)
        self.scroll.setMaximum(10)
        self.scroll.valueChanged.connect(self.move_in_Z)
        self.l0.addWidget(self.scroll, b,3,1,20)
        return b

    def keyPressEvent(self, event):
        if self.loaded:
            #self.p0.setMouseEnabled(x=True, y=True)
            if (event.modifiers() != QtCore.Qt.ControlModifier and
                event.modifiers() != QtCore.Qt.ShiftModifier and
                event.modifiers() != QtCore.Qt.AltModifier) and not self.in_stroke:
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                    if self.NZ>1:
                        if event.key() == QtCore.Qt.Key_Left:
                            self.currentZ = max(0,self.currentZ-1)
                            self.zpos.setText(str(self.currentZ))
                        elif event.key() == QtCore.Qt.Key_Right:
                            self.currentZ = min(self.NZ-1, self.currentZ+1)
                            self.zpos.setText(str(self.currentZ))
                else:
                    if event.key() == QtCore.Qt.Key_X:
                        self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Z:
                        self.OCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Left:
                        if self.NZ==1:
                            self.get_prev_image()
                        else:
                            self.currentZ = max(0,self.currentZ-1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
                    elif event.key() == QtCore.Qt.Key_Right:
                        if self.NZ==1:
                            self.get_next_image()
                        else:
                            self.currentZ = min(self.NZ-1, self.currentZ+1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
                    elif event.key() == QtCore.Qt.Key_A:
                        if self.NZ==1:
                            self.get_prev_image()
                        else:
                            self.currentZ = max(0,self.currentZ-1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
                    elif event.key() == QtCore.Qt.Key_D:
                        if self.NZ==1:
                            self.get_next_image()
                        else:
                            self.currentZ = min(self.NZ-1, self.currentZ+1)
                            self.scroll.setValue(self.currentZ)
                            updated = True

                    elif event.key() == QtCore.Qt.Key_PageDown:
                        self.view = (self.view+1)%(len(self.RGBChoose.bstr))
                        self.RGBChoose.button(self.view).setChecked(True)
                    elif event.key() == QtCore.Qt.Key_PageUp:
                        self.view = (self.view-1)%(len(self.RGBChoose.bstr))
                        self.RGBChoose.button(self.view).setChecked(True)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                    self.color = (self.color-1)%(6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_S:
                    self.color = (self.color+1)%(6)
                    self.RGBDropDown.setCurrentIndex(self.color)
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
                if not updated:
                    self.update_plot()
                elif event.modifiers() == QtCore.Qt.ControlModifier:
                    if event.key() == QtCore.Qt.Key_Z:
                        self.undo_action()
                    if event.key() == QtCore.Qt.Key_0:
                        self.clear_all()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def check_gpu(self, torch=True):
        # also decide whether or not to use torch
        self.torch = torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)    
        if self.torch and models.use_gpu(istorch=True):
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        elif models.MXNET_ENABLED:
            if models.use_gpu(istorch=False):
                print('>>> will run model on GPU in mxnet <<<')
                self.torch = False
                self.useGPU.setEnabled(True)
                self.useGPU.setChecked(True)
            elif models.check_mkl(istorch=False):
                print('>>> will run model on CPU (MKL-accelerated) in mxnet <<<')
                self.torch = False
                self.useGPU.setStyleSheet("color: rgb(80,80,80);")
            elif not self.torch:
                print('>>> will run model on CPU (not MKL-accelerated) in mxnet <<<')
                self.useGPU.setStyleSheet("color: rgb(80,80,80);")
            else:
                self.useGPU.setStyleSheet("color: rgb(80,80,80);")
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")

    def get_channels(self):
        channels = [self.ChannelChoose[0].currentIndex(), self.ChannelChoose[1].currentIndex()]
        if self.current_model=='nuclei':
            channels[1] = 0
        return channels

    def calibrate_size(self):
        self.initialize_model()
        diams, _ = self.model.sz.eval(self.stack[self.currentZ].copy(), invert=self.invert.isChecked(),
                                   channels=self.get_channels(), progress=self.progress)
        diams = np.maximum(5.0, diams)
        print('estimated diameter of cells using %s model = %0.1f pixels'%
                (self.current_model, diams))
        self.Diameter.setText('%0.1f'%diams)
        self.diameter = diams
        self.compute_scale()
        self.progress.setValue(100)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

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
        if self.selected>0:
            self.remove_cell(self.selected)

    def undo_action(self):
        if (len(self.strokes) > 0 and
            self.strokes[-1][0][0]==self.currentZ):
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells> 0:
                self.remove_cell(self.ncells)

    def undo_remove_action(self):
        self.undo_remove_cell()

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
        return images, idx

    def get_prev_image(self):
        images, idx = self.get_files()
        idx = (idx-1)%len(images)
        io._load_image(self, filename=images[idx])

    def get_next_image(self):
        images, idx = self.get_files()
        idx = (idx+1)%len(images)
        io._load_image(self, filename=images[idx])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == '.npy':
            io._load_seg(self, filename=files[0])
        else:
            io._load_image(self, filename=files[0])

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
            self.zpos.setText(str(self.currentZ))
            self.update_plot()

    def make_viewbox(self):
        self.p0 = guiparts.ViewBoxNoRightDrag(
            parent=self,
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

        guiparts.make_quadrants(self)

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.X2 = 0
        self.resize = -1
        self.onechan = False
        self.loaded = False
        self.channel = [0,1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
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
        self.flows = [[],[],[],[],[[]]]
        self.stack = np.zeros((1,self.Ly,self.Lx,3))
        # masks matrix
        self.layers = 0*np.ones((1,self.Ly,self.Lx,4), np.uint8)
        # image matrix with a scale disk
        self.radii = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((1,self.Ly,self.Lx), np.uint16)
        self.outpix = np.zeros((1,self.Ly,self.Lx), np.uint16)
        self.ismanual = np.zeros(0, np.bool)
        self.update_plot()
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
        self.prev_selected = 0
        self.selected = 0
        #self.layers_undo, self.cellpix_undo, self.outpix_undo = [],[],[]
        self.layers = 0*np.ones((self.NZ,self.Ly,self.Lx,4), np.uint8)
        self.cellpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint16)
        self.outpix = np.zeros((self.NZ,self.Ly,self.Lx), np.uint16)
        self.cellcolors = [np.array([255,255,255])]
        self.ncells = 0
        print('removed all cells')
        self.toggle_removals()
        self.update_plot()

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            self.layers[self.cellpix==idx] = np.array([255,255,255,self.opacity])
            #if self.outlinesOn:
            #    self.layers[self.outpix==idx] = np.array(self.outcolor)
            self.update_plot()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < self.ncells+1:
                self.layers[self.cellpix==idx] = np.append(self.cellcolors[idx], self.opacity)
                if self.outlinesOn:
                    self.layers[self.outpix==idx] = np.array(self.outcolor).astype(np.uint8)
                    #[0,0,0,self.opacity])
                self.update_plot()
        self.selected = 0

    def remove_cell(self, idx):
        # remove from manual array
        self.selected = 0
        for z in range(self.NZ):
            cp = self.cellpix[z]==idx
            op = self.outpix[z]==idx
            # remove from mask layer
            self.layers[z, cp] = np.array([0,0,0,0])
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            # reduce other pixels by -1
            self.cellpix[z, self.cellpix[z]>idx] -= 1
            self.outpix[z, self.outpix[z]>idx] -= 1
        self.update_plot()
        if self.NZ==1:
            self.removed_cell = [self.ismanual[idx-1], self.cellcolors[idx], np.nonzero(cp), np.nonzero(op)]
            self.redo.setEnabled(True)
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx-1)
        del self.cellcolors[idx]
        del self.zdraw[idx-1]
        self.ncells -= 1
        print('removed cell %d'%(idx-1))
        if self.ncells==0:
            self.ClearButton.setEnabled(False)
        if self.NZ==1:
            io._save_sets(self)

    def merge_cells(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z]==self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z]==self.selected)
                touching = np.logical_and((ar0[:,np.newaxis] - ar1)==1,
                                            (ac0[:,np.newaxis] - ac1)==1).sum()
                print(touching)
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                if touching:
                    mask = np.zeros((np.ptp(ar)+4, np.ptp(ac)+4), np.uint8)
                    mask[ar-ar.min()+2, ac-ac.min()+2] = 1
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    pvc, pvr = contours[-2][0].squeeze().T            
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
                else:
                    vr0, vc0 = np.nonzero(self.outpix[z]==self.prev_selected)
                    vr1, vc1 = np.nonzero(self.outpix[z]==self.selected)
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print('merged two cells')
            self.update_plot()
            io._save_sets(self)
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
            self.cellcolors.append(color)
            self.ncells+=1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print('added back removed cell')
            self.update_plot()
            io._save_sets(self)
            self.removed_cell = []
            self.redo.setEnabled(False)


    def remove_stroke(self, delete_points=True):
        #self.current_stroke = get_unique_points(self.current_stroke)
        stroke = np.array(self.strokes[-1])
        cZ = stroke[0,0]
        outpix = self.outpix[cZ][stroke[:,1],stroke[:,2]]>0
        self.layers[cZ][stroke[~outpix,1],stroke[~outpix,2]] = np.array([0,0,0,0])
        if self.masksOn:
            cellpix = self.cellpix[cZ][stroke[:,1], stroke[:,2]]
            ccol = np.array(self.cellcolors.copy())
            if self.selected > 0:
                ccol[self.selected] = np.array([255,255,255])
            col2mask = ccol[cellpix]
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

    def update_ztext(self):
        zpos = self.currentZ
        try:
            zpos = int(self.zpos.text())
        except:
            print('ERROR: zposition is not a number')
        self.currentZ = max(0, min(self.NZ-1, zpos))
        self.zpos.setText(str(self.currentZ))
        self.scroll.setValue(self.currentZ)

    def update_plot(self):
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
            elif self.color==2:
                image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
            elif self.color>2:
                image = image[:,:,self.color-3]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color-2])
            self.img.setLevels(self.saturation[self.currentZ])
        else:
            image = np.zeros((self.Ly,self.Lx), np.uint8)
            if len(self.flows)>=self.view-1 and len(self.flows[self.view-1])>0:
                image = self.flows[self.view-1][self.currentZ]
            if self.view>1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0,255.0])
        #self.img.set_ColorMap(self.bwr)
        if self.masksOn or self.outlinesOn:
            self.layer.setImage(self.layers[self.currentZ], autoLevels=False)
        #self.slider.setLow(self.saturation[self.currentZ][0])
        #self.slider.setHigh(self.saturation[self.currentZ][1])
        self.win.show()
        self.show()

    def add_set(self):
        if len(self.current_point_set) > 0:
            self.current_point_set = np.array(self.current_point_set)
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set) > 8:
                col_rand = np.random.randint(1000)
                color = self.colormap[col_rand,:3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors.append(color)
                    self.ncells+=1
                    self.ismanual = np.append(self.ismanual, True)
                    if self.NZ==1:
                        # only save after each cell if single image
                        io._save_sets(self)
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_plot()

    def add_mask(self, points=None, color=(100,200,50)):
        # loop over z values
        median = []
        if points.shape[1] < 3:
            points = np.concatenate((np.zeros((points.shape[0],1), np.int32), points), axis=1)

        zdraw = np.unique(points[:,0])
        zrange = np.arange(zdraw.min(), zdraw.max()+1, 1, int)
        zmin = zdraw.min()
        pix = np.zeros((2,0), np.uint16)
        mall = np.zeros((len(zrange), self.Ly, self.Lx), np.bool)
        k=0
        for z in zdraw:
            iz = points[:,0] == z
            vr = points[iz,1]
            vc = points[iz,2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr)+4, np.ptp(vc)+4), np.uint8)
            pts = np.stack((vc-vc.min()+2,vr-vr.min()+2), axis=-1)[:,np.newaxis,:]
            mask = cv2.fillPoly(mask, [pts], (255,0,0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar+vr.min()-2, ac+vc.min()-2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0].squeeze().T            
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 8:
                print('ERROR: cell too small without overlaps, not drawn')
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(ar)+4, np.ptp(ac)+4), np.uint8)
                mask[ar-ar.min()+2, ac-ac.min()+2] = 1
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0].squeeze().T            
                vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
            self.draw_mask(z, ar, ac, vr, vc, color)

            median.append(np.array([np.median(ar), np.median(ac)]))
            mall[z-zmin, ar, ac] = True
            pix = np.append(pix, np.vstack((ar, ac)), axis=-1)

        mall = mall[:, pix[0].min():pix[0].max()+1, pix[1].min():pix[1].max()+1].astype(np.float32)
        ymin, xmin = pix[0].min(), pix[1].min()
        if len(zdraw) > 1:
            mall, zfill = interpZ(mall, zdraw - zmin)
            for z in zfill:
                mask = mall[z].copy()
                ar, ac = np.nonzero(mask)
                ioverlap = self.cellpix[z+zmin][ar+ymin, ac+xmin] > 0
                if (~ioverlap).sum() < 5:
                    print('WARNING: stroke on plane %d not included due to overlaps'%z)
                elif ioverlap.sum() > 0:
                    mask[ar[ioverlap], ac[ioverlap]] = 0
                    ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of mask
                outlines = masks_to_outlines(mask)
                vr, vc = np.nonzero(outlines)
                vr, vc = vr+ymin, vc+xmin
                ar, ac = ar+ymin, ac+xmin
                self.draw_mask(z+zmin, ar, ac, vr, vc, color)
        self.zdraw.append(zdraw)

        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        ''' draw single mask using outlines and area '''
        if idx is None:
            idx = self.ncells+1
        self.cellpix[z][vr, vc] = idx
        self.cellpix[z][ar, ac] = idx
        self.outpix[z][vr, vc] = idx
        if self.masksOn:
            self.layers[z][ar, ac, :3] = color
            self.layers[z][ar, ac, -1] = self.opacity
        if self.outlinesOn:
            self.layers[z][vr, vc] = np.array(self.outcolor)


    def compute_scale(self):
        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))
        radii = np.zeros((self.Ly+self.pr,self.Lx), np.uint8)
        self.radii = np.zeros((self.Ly+self.pr,self.Lx,4), np.uint8)
        yy,xx = disk([self.Ly+self.pr/2-1, self.pr/2+1],
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
            self.layers[...,:3] = self.cellcolors[self.cellpix,:]
            self.layers[...,3] = self.opacity * (self.cellpix>0).astype(np.uint8)
            self.cellcolors = list(self.cellcolors)
            if self.selected>0:
                self.layers[self.cellpix==self.selected] = np.array([255,255,255,self.opacity])
        else:
            if masks:
                self.layers[...,3] = self.opacity * (self.cellpix>0).astype(np.uint8)
            else:
                self.layers[...,3] = 0
            self.layers[self.outpix>0] = np.array(self.outcolor).astype(np.uint8)

    def draw_masks(self):
        self.cellcolors = np.array(self.cellcolors)
        self.layers[...,:3] = self.cellcolors[self.cellpix,:]
        self.layers[...,3] = self.opacity * (self.cellpix>0).astype(np.uint8)
        self.cellcolors = list(self.cellcolors)
        self.layers[self.outpix>0] = np.array(self.outcolor)
        if self.selected>0:
            self.layers[self.outpix==self.selected] = np.array([0,0,0,self.opacity])

    def compute_saturation(self):
        # compute percentiles from stack
        self.saturation = []
        for n in range(len(self.stack)):
            self.saturation.append([np.percentile(self.stack[n].astype(np.float32),1),
                                    np.percentile(self.stack[n].astype(np.float32),99)])

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
        self.current_model = self.ModelChoose.currentText()
        print(self.current_model)
        self.model = models.Cellpose(gpu=self.useGPU.isChecked(), 
                                     torch=self.torch,
                                     model_type=self.current_model)

    def compute_cprob(self):
        rerun = False
        if self.cellprob != self.probslider.value():
            rerun = True
            self.cellprob = self.probslider.value()
        if self.threshold != (31 - self.threshslider.value())/10.:
            rerun = True
            self.threshold = (31 - self.threshslider.value())/10.
        if not rerun:
            return
        
        if self.threshold==3.0 or self.NZ>1:
            thresh = None
            print('computing masks with cell prob=%0.3f, no flow error threshold'%
                    (self.cellprob))
        else:
            thresh = self.threshold
            print('computing masks with cell prob=%0.3f, flow error threshold=%0.3f'%
                    (self.cellprob, thresh))

        maski = self.model.cp._compute_masks(self.flows[4][:-1],
                                             self.flows[4][-1],
                                             p=self.flows[3].copy(),
                                             cellprob_threshold=self.cellprob,
                                             flow_threshold=thresh,
                                             resize=self.cellpix.shape[-2:])[0]
        
        self.masksOn = True
        self.outlinesOn = True
        self.MCheckBox.setChecked(True)
        self.OCheckBox.setChecked(True)
        if maski.ndim<3:
            maski = maski[np.newaxis,...]
        print('%d cells found'%(len(np.unique(maski)[1:])))
        io._masks_to_gui(self, maski, outlines=None)
        self.show()

    def compute_model(self):
        self.progress.setValue(0)
        if 1:
            tic=time.time()
            self.clear_all()
            self.flows = [[],[],[]]
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
                net_avg = self.NetAvg.currentIndex()<2
                resample = self.NetAvg.currentIndex()==1
                masks, flows, _, _ = self.model.eval(data, channels=channels,
                                                    diameter=self.diameter, invert=self.invert.isChecked(),
                                                    net_avg=net_avg, augment=False, resample=resample,
                                                    do_3D=do_3D, progress=self.progress)
            except Exception as e:
                print('NET ERROR: %s'%e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)

            #if not do_3D:
            #    masks = masks[0][np.newaxis,:,:]
            #    flows = flows[0]
            self.flows[0] = flows[0]
            self.flows[1] = (np.clip(normalize99(flows[2].copy()),0,1) * 255).astype(np.uint8)
            if not do_3D:
                masks = masks[np.newaxis,...]
                self.flows[0] = resize_image(self.flows[0], masks.shape[-2], masks.shape[-1],
                                                        interpolation=cv2.INTER_NEAREST)
                self.flows[1] = resize_image(self.flows[1], masks.shape[-2], masks.shape[-1])
            if not do_3D:
                self.flows[2] = np.zeros(masks.shape[1:], dtype=np.uint8)
                self.flows = [self.flows[n][np.newaxis,...] for n in range(len(self.flows))]
            else:
                self.flows[2] = (flows[1][0]/10 * 127 + 127).astype(np.uint8)
                
            if len(flows)>2:
                self.flows.append(flows[3].squeeze())
                self.flows.append(np.concatenate((flows[1], flows[2][np.newaxis,...]), axis=0))
                
            print('%d cells found with cellpose net in %0.3f sec'%(len(np.unique(masks)[1:]), time.time()-tic))
            self.progress.setValue(80)
            z=0
            self.masksOn = True
            self.outlinesOn = True
            self.MCheckBox.setChecked(True)
            self.OCheckBox.setChecked(True)

            io._masks_to_gui(self, masks, outlines=None)
            self.progress.setValue(100)

            self.toggle_server(off=True)
            if not do_3D:
                self.threshslider.setEnabled(True)
                self.probslider.setEnabled(True)
        else: #except Exception as e:
            print('ERROR: %s'%e)


    def enable_buttons(self):
        #self.X2Up.setEnabled(True)
        #self.X2Down.setEnabled(True)
        self.ModelButton.setEnabled(True)
        self.SizeButton.setEnabled(True)
        self.ModelButton.setStyleSheet(self.styleUnpressed)
        self.SizeButton.setStyleSheet(self.styleUnpressed)
        self.loadMasks.setEnabled(True)
        self.saveSet.setEnabled(True)
        self.savePNG.setEnabled(True)
        self.saveOutlines.setEnabled(True)
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
