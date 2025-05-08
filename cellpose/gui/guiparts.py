"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
from qtpy import QtGui, QtCore
from qtpy.QtGui import QPixmap, QDoubleValidator
from qtpy.QtWidgets import QWidget, QDialog, QGridLayout, QPushButton, QLabel, QLineEdit, QDialogButtonBox, QComboBox, QCheckBox, QVBoxLayout
import pyqtgraph as pg
import numpy as np
import pathlib, os


def stylesheet():
    return """
        QToolTip { 
                            background-color: black; 
                            color: white; 
                            border: black solid 1px
                            }
        QComboBox {color: white;
                    background-color: rgb(40,40,40);}
                    QComboBox::item:enabled { color: white;
                    background-color: rgb(40,40,40);
                    selection-color: white;
                    selection-background-color: rgb(50,100,50);}
                    QComboBox::item:!enabled {
                            background-color: rgb(40,40,40);
                            color: rgb(100,100,100);
                        }
        QScrollArea > QWidget > QWidget
                {
                    background: transparent;
                    border: none;
                    margin: 0px 0px 0px 0px;
                } 
                           
        QGroupBox 
            { border: 1px solid white; color: rgb(255,255,255);
                           border-radius: 6px;
                            margin-top: 8px;
                            padding: 0px 0px;}            
                           
        QPushButton:pressed {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:!pressed {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:disabled {Text-align: center; 
                             background-color: rgb(30,30,30);
                             border-color: white;
                              color:rgb(80,80,80);}
                               QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
                        
        """


class DarkPalette(QtGui.QPalette):
    """Class that inherits from pyqtgraph.QtGui.QPalette and renders dark colours for the application.
    (from pykilosort/kilosort4)
    """

    def __init__(self):
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        self.setColor(QtGui.QPalette.Window, QtGui.QColor(40, 40, 40))
        self.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Base, QtGui.QColor(34, 27, 24))
        self.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        self.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,
                      QtGui.QColor(128, 128, 128))
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.ButtonText,
            QtGui.QColor(128, 128, 128),
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.WindowText,
            QtGui.QColor(128, 128, 128),
        )


# def create_channel_choose():
#     # choose channel
#     ChannelChoose = [QComboBox(), QComboBox()]
#     ChannelLabels = []
#     ChannelChoose[0].addItems(["gray", "red", "green", "blue"])
#     ChannelChoose[1].addItems(["none", "red", "green", "blue"])
#     cstr = ["chan to segment:", "chan2 (optional): "]
#     for i in range(2):
#         ChannelLabels.append(QLabel(cstr[i]))
#         if i == 0:
#             ChannelLabels[i].setToolTip(
#                 "this is the channel in which the cytoplasm or nuclei exist \
#             that you want to segment")
#             ChannelChoose[i].setToolTip(
#                 "this is the channel in which the cytoplasm or nuclei exist \
#             that you want to segment")
#         else:
#             ChannelLabels[i].setToolTip(
#                 "if <em>cytoplasm</em> model is chosen, and you also have a \
#             nuclear channel, then choose the nuclear channel for this option")
#             ChannelChoose[i].setToolTip(
#                 "if <em>cytoplasm</em> model is chosen, and you also have a \
#             nuclear channel, then choose the nuclear channel for this option")

#     return ChannelChoose, ChannelLabels


class ModelButton(QPushButton):

    def __init__(self, parent, model_name, text):
        super().__init__()
        self.setEnabled(False)
        self.setText(text)
        self.setFont(parent.boldfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = "cpsam"

    def press(self, parent):
        parent.compute_segmentation(model_name="cpsam")


class FilterButton(QPushButton):

    def __init__(self, parent, text):
        super().__init__()
        self.setEnabled(False)
        self.model_type = text
        self.setText(text)
        self.setFont(parent.medfont)
        self.clicked.connect(lambda: self.press(parent))

    def press(self, parent):
        if self.model_type == "filter":
            parent.restore = "filter"
            normalize_params = parent.get_normalize_params()
            if (normalize_params["sharpen_radius"] == 0 and
                    normalize_params["smooth_radius"] == 0 and
                    normalize_params["tile_norm_blocksize"] == 0):
                print(
                    "GUI_ERROR: no filtering settings on (use custom filter settings)")
                parent.restore = None
                return
            parent.restore = self.model_type
            parent.compute_saturation()
        # elif self.model_type != "none":
        #     parent.compute_denoise_model(model_type=self.model_type)
        else:
            parent.clear_restore()
        # parent.set_restore_button()


class ObservableVariable(QtCore.QObject):
    valueChanged = QtCore.Signal(object) 

    def __init__(self, initial=None):
        super().__init__()
        self._value = initial

    def set(self, new_value):
        """ Use this method to get emit the value changing and update the ROI count"""
        if new_value != self._value:
            self._value = new_value
            self.valueChanged.emit(new_value)

    def get(self):
        return self._value

    def __call__(self):
        return self._value
    
    def reset(self):
        self.set(0)

    def __iadd__(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Value must be numeric.")
        self.set(self._value + amount)
        return self
    
    def __radd__(self, other):
        return other + self._value

    def __add__(self, other):
        return other + self._value
        
    def __isub__(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Value must be numeric.")
        self.set(self._value - amount)
        return self
    
    def __str__(self):
        return str(self._value)
    
    def __lt__(self, x):
        return self._value < x
    
    def __gt__(self, x):
        return self._value > x
    
    def __eq__(self, x):
        return self._value == x


class NormalizationSettings(QWidget):
    # TODO
    pass


class SegmentationSettings(QWidget):
    """ Container for gui settings. Validation is done automatically so any attributes can 
    be acessed without concern.  
    """
    def __init__(self, font):
        super().__init__()

        # Put everything in a grid layout:
        grid_layout = QGridLayout()
        widget_container = QWidget()
        widget_container.setLayout(grid_layout)
        row = 0

        ########################### Diameter ###########################
        # TODO: Validate inputs
        diam_qlabel = QLabel("diameter:")
        diam_qlabel.setToolTip("diameter of cells in pixels. If not 30, image will be resized to this")
        diam_qlabel.setFont(font)
        grid_layout.addWidget(diam_qlabel, row, 0, 1, 2)
        self.diameter_box = QLineEdit()
        self.diameter_box.setToolTip("diameter of cells in pixels. If not blank, image will be resized relative to 30 pixel cell diameters")
        self.diameter_box.setFont(font)
        self.diameter_box.setFixedWidth(40)
        self.diameter_box.setText(' ')
        grid_layout.addWidget(self.diameter_box, row, 2, 1, 2)

        row += 1

        ########################### Flow threshold ###########################
        # TODO: Validate inputs
        flow_threshold_qlabel = QLabel("flow\nthreshold:")
        flow_threshold_qlabel.setToolTip("threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run")
        flow_threshold_qlabel.setFont(font)
        grid_layout.addWidget(flow_threshold_qlabel, row, 0, 1, 2)
        self.flow_threshold_box = QLineEdit()
        self.flow_threshold_box.setText("0.4")
        self.flow_threshold_box.setFixedWidth(40)
        self.flow_threshold_box.setFont(font)
        grid_layout.addWidget(self.flow_threshold_box, row, 2, 1, 2)
        self.flow_threshold_box.setToolTip("threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run")
        
        ########################### Cellprob threshold ###########################
        # TODO: Validate inputs
        cellprob_qlabel = QLabel("cellprob\nthreshold:")
        cellprob_qlabel.setToolTip("threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run")
        cellprob_qlabel.setFont(font)
        grid_layout.addWidget(cellprob_qlabel, row, 4, 1, 2)
        self.cellprob_threshold_box = QLineEdit()
        self.cellprob_threshold_box.setText("0.0")
        self.cellprob_threshold_box.setFixedWidth(40)
        self.cellprob_threshold_box.setFont(font)
        self.cellprob_threshold_box.setToolTip("threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run")
        grid_layout.addWidget(self.cellprob_threshold_box, row, 6, 1, 2)

        row += 1

        ########################### Norm percentiles ###########################
        norm_percentiles_qlabel = QLabel("norm percentiles:")
        norm_percentiles_qlabel.setToolTip("sets normalization percentiles for segmentation and denoising\n(pixels at lower percentile set to 0.0 and at upper set to 1.0 for network)")
        norm_percentiles_qlabel.setFont(font)
        grid_layout.addWidget(norm_percentiles_qlabel, row, 0, 1, 8)

        row += 1
        validator = QDoubleValidator(0.0, 100.0, 2)
        validator.setNotation(QDoubleValidator.StandardNotation)

        low_norm_qlabel = QLabel('lower:')
        low_norm_qlabel.setToolTip("pixels at this percentile set to 0 (default 1.0)")
        low_norm_qlabel.setFont(font)
        grid_layout.addWidget(low_norm_qlabel, row, 0, 1, 2)
        self.norm_percentile_low_box = QLineEdit()
        self.norm_percentile_low_box.setText("1.0")
        self.norm_percentile_low_box.setFont(font)
        self.norm_percentile_low_box.setFixedWidth(40)
        self.norm_percentile_low_box.setToolTip("pixels at this percentile set to 0 (default 1.0)")
        self.norm_percentile_low_box.setValidator(validator)
        self.norm_percentile_low_box.editingFinished.connect(self.validate_normalization_range)
        grid_layout.addWidget(self.norm_percentile_low_box, row, 2, 1, 1)

        high_norm_qlabel = QLabel('upper:')
        high_norm_qlabel.setToolTip("pixels at this percentile set to 1 (default 99.0)")
        high_norm_qlabel.setFont(font)
        grid_layout.addWidget(high_norm_qlabel, row, 4, 1, 2)
        self.norm_percentile_high_box = QLineEdit()
        self.norm_percentile_high_box.setText("99.0")
        self.norm_percentile_high_box.setFont(font)
        self.norm_percentile_high_box.setFixedWidth(40)
        self.norm_percentile_high_box.setToolTip("pixels at this percentile set to 1 (default 99.0)")
        self.norm_percentile_high_box.setValidator(validator)
        self.norm_percentile_high_box.editingFinished.connect(self.validate_normalization_range)
        grid_layout.addWidget(self.norm_percentile_high_box, row, 6, 1, 2)

        row += 1

        ########################### niter ###########################
        # TODO: change this to follow the same default logic as 'diameter' above
        # TODO: input validation
        niter_qlabel = QLabel("niter dynamics:")
        niter_qlabel.setFont(font)
        niter_qlabel.setToolTip("number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria")
        grid_layout.addWidget(niter_qlabel, row, 0, 1, 4)
        self.niter_box = QLineEdit()
        self.niter_box.setText("0")
        self.niter_box.setFixedWidth(40)
        self.niter_box.setFont(font)
        self.niter_box.setToolTip("number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria")
        grid_layout.addWidget(self.niter_box, row, 4, 1, 2)

        self.setLayout(grid_layout)

    def validate_normalization_range(self):
        low_text = self.norm_percentile_low_box.text()
        high_text = self.norm_percentile_high_box.text()
        
        if not low_text or low_text.isspace():
            self.norm_percentile_low_box.setText('1.0')
            low_text = '1.0'
        elif not high_text or high_text.isspace():
            self.norm_percentile_high_box.setText('1.0')
            high_text = '99.0'

        low = float(low_text)
        high = float(high_text)

        if low >= high:
            # Invalid: show error and mark fields
            self.norm_percentile_low_box.setStyleSheet("border: 1px solid red;")
            self.norm_percentile_high_box.setStyleSheet("border: 1px solid red;")
        else:
            # Valid: clear style
            self.norm_percentile_low_box.setStyleSheet("")
            self.norm_percentile_high_box.setStyleSheet("")

    @property
    def low_percentile(self):
        """ Also validate the low input by returning 1.0 if text doesn't work """
        low_text = self.norm_percentile_low_box.text()
        if not low_text or low_text.isspace():
            self.norm_percentile_low_box.setText('1.0')
            low_text = '1.0'
        return float(self.norm_percentile_low_box.text())
    
    @property
    def high_percentile(self):
        """ Also validate the high input by returning 99.0 if text doesn't work """
        high_text = self.norm_percentile_high_box.text()
        if not high_text or high_text.isspace():
            self.norm_percentile_high_box.setText('99.0')
            high_text = '99.0'
        return float(self.norm_percentile_high_box.text())
    
    @property
    def diameter(self):
        """ Get the diameter from the diameter box, if box isn't a number return None"""
        try:
            d = float(self.diameter_box.text())
        except ValueError:
            d = None
        return d 
    
    @property
    def flow_threshold(self):
        return float(self.flow_threshold_box.text())
    
    @property
    def cellprob_threshold(self):
        return float(self.cellprob_threshold_box.text())
    
    @property
    def niter(self):
        num = int(self.niter_box.text())
        if num < 1:
            self.niter_box.setText('200')
            return 200
        else:
            return num



class TrainWindow(QDialog):

    def __init__(self, parent, model_strings):
        super().__init__(parent)
        self.setGeometry(100, 100, 900, 550)
        self.setWindowTitle("train settings")
        self.win = QWidget(self)
        self.l0 = QGridLayout()
        self.win.setLayout(self.l0)

        yoff = 0
        qlabel = QLabel("train model w/ images + _seg.npy in current folder >>")
        qlabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))

        qlabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff, 0, 1, 2)

        # choose initial model
        yoff += 1
        self.ModelChoose = QComboBox()
        self.ModelChoose.addItems(model_strings)
        self.ModelChoose.setFixedWidth(150)
        self.ModelChoose.setCurrentIndex(parent.training_params["model_index"])
        self.l0.addWidget(self.ModelChoose, yoff, 1, 1, 1)
        qlabel = QLabel("initial model: ")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff, 0, 1, 1)

        # choose parameters
        labels = ["learning_rate", "weight_decay", "n_epochs", "model_name"]
        self.edits = []
        yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i + yoff, 0, 1, 1)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            self.edits[-1].setFixedWidth(200)
            self.l0.addWidget(self.edits[-1], i + yoff, 1, 1, 1)

        yoff += len(labels)

        yoff += 1
        self.use_norm = QCheckBox(f"use restored/filtered image")
        self.use_norm.setChecked(True)

        yoff += 2
        qlabel = QLabel(
            "(to remove files, click cancel then remove \nfrom folder and reopen train window)"
        )
        self.l0.addWidget(qlabel, yoff, 0, 2, 4)

        # click button
        yoff += 3
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda: self.accept(parent))
        self.buttonBox.rejected.connect(self.reject)
        self.l0.addWidget(self.buttonBox, yoff, 0, 1, 4)

        # list files in folder
        qlabel = QLabel("filenames")
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0, 4, 1, 1)
        qlabel = QLabel("# of masks")
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0, 5, 1, 1)

        for i in range(10):
            if i > len(parent.train_files) - 1:
                break
            elif i == 9 and len(parent.train_files) > 10:
                label = "..."
                nmasks = "..."
            else:
                label = os.path.split(parent.train_files[i])[-1]
                nmasks = str(parent.train_labels[i].max())
            qlabel = QLabel(label)
            self.l0.addWidget(qlabel, i + 1, 4, 1, 1)
            qlabel = QLabel(nmasks)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i + 1, 5, 1, 1)

    def accept(self, parent):
        # set training params
        parent.training_params = {
            "model_index": self.ModelChoose.currentIndex(),
            "learning_rate": float(self.edits[0].text()),
            "weight_decay": float(self.edits[1].text()),
            "n_epochs": int(self.edits[2].text()),
            "model_name": self.edits[3].text(),
            #"use_norm": True if self.use_norm.isChecked() else False,
        }
        self.done(1)


class ExampleGUI(QDialog):

    def __init__(self, parent=None):
        super(ExampleGUI, self).__init__(parent)
        self.setGeometry(100, 100, 1300, 900)
        self.setWindowTitle("GUI layout")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        guip_path = pathlib.Path.home().joinpath(".cellpose", "cellposeSAM_gui.png")
        guip_path = str(guip_path.resolve())
        pixmap = QPixmap(guip_path)
        label = QLabel(self)
        label.setPixmap(pixmap)
        pixmap.scaled
        layout.addWidget(label, 0, 0, 1, 1)


class HelpWindow(QDialog):

    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        self.setGeometry(100, 50, 700, 1000)
        self.setWindowTitle("cellpose help")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        text_file = pathlib.Path(__file__).parent.joinpath("guihelpwindowtext.html")
        with open(str(text_file.resolve()), "r") as f:
            text = f.read()

        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class TrainHelpWindow(QDialog):

    def __init__(self, parent=None):
        super(TrainHelpWindow, self).__init__(parent)
        self.setGeometry(100, 50, 700, 300)
        self.setWindowTitle("training instructions")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        text_file = pathlib.Path(__file__).parent.joinpath(
            "guitrainhelpwindowtext.html")
        with open(str(text_file.resolve()), "r") as f:
            text = f.read()

        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class ViewBoxNoRightDrag(pg.ViewBox):

    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True,
                 invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse, invertY,
                            enableMenu, name, invertX)
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)

        """
        ev.accept()
        if ev.text() == "-":
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ["+", "="]:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()


class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = QtCore.Signal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        self.levels = np.array([0, 255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = "row-major"
        self.removable = False

        self.parent = parent
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        if (self.parent.masksOn or
                self.parent.outlinesOn) and not self.parent.removing_region:
            is_right_click = ev.button() == QtCore.Qt.RightButton
            if self.parent.loaded \
                    and (is_right_click or ev.modifiers() & QtCore.Qt.ShiftModifier and not ev.double())\
                    and not self.parent.deleting_multiple:
                if not self.parent.in_stroke:
                    ev.accept()
                    self.create_start(ev.pos())
                    self.parent.stroke_appended = False
                    self.parent.in_stroke = True
                    self.drawAt(ev.pos(), ev)
                else:
                    ev.accept()
                    self.end_stroke()
                    self.parent.in_stroke = False
            elif not self.parent.in_stroke:
                y, x = int(ev.pos().y()), int(ev.pos().x())
                if y >= 0 and y < self.parent.Ly and x >= 0 and x < self.parent.Lx:
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():
                        idx = self.parent.cellpix[self.parent.currentZ][y, x]
                        if idx > 0:
                            if ev.modifiers() & QtCore.Qt.ControlModifier:
                                # delete mask selected
                                self.parent.remove_cell(idx)
                            elif ev.modifiers() & QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif self.parent.masksOn and not self.parent.deleting_multiple:
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                            elif self.parent.deleting_multiple:
                                if idx in self.parent.removing_cells_list:
                                    self.parent.unselect_cell_multi(idx)
                                    self.parent.removing_cells_list.remove(idx)
                                else:
                                    self.parent.select_cell_multi(idx)
                                    self.parent.removing_cells_list.append(idx)

                        elif self.parent.masksOn and not self.parent.deleting_multiple:
                            self.parent.unselect_cell()

    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    self.end_stroke()
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                          pen=pg.mkPen(color=(255, 0, 0),
                                                       width=self.parent.brush_size),
                                          size=max(3 * 2,
                                                   self.parent.brush_size * 1.8 * 2),
                                          brush=None)
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size * 3)
        thresh_in = max(3, self.parent.brush_size * 1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (((stroke[1:, 1:] -
                      stroke[:1, 1:][np.newaxis, :, :])**2).sum(axis=-1))**0.5
            dist = dist.flatten()
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4, first_left + 1):] < thresh_in).sum()
                if has_returned > 0:
                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:, 3] == 1
            self.parent.current_point_set.append(
                list(self.parent.current_stroke[ioutline]))
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if len(self.parent.current_point_set) and len(
                self.parent.current_point_set[0]) > 0 and self.parent.autosave:
            self.parent.add_set()
        self.parent.in_stroke = False

    def tabletEvent(self, ev):
        pass

    def drawAt(self, pos, ev=None):
        mask = self.strokemask
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0, dk.shape[0]]
        sy = [0, dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0] + dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1] + dk.shape[1]]
        kcent = kc.copy()
        if tx[0] <= 0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx = sx
            kcent[0] = 0
        if ty[0] <= 0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly - 1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1] - tx[0] - 1
        if ty[1] >= self.parent.Lx - 1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1] - ty[0] - 1

        ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
        ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
        self.image[ts] = mask[ss]

        for ky, y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx, x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx == kcent[0], ky == kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs, bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [
            int(np.floor(kernel.shape[0] / 2)),
            int(np.floor(kernel.shape[1] / 2))
        ]
        onmask = 255 * kernel[:, :, np.newaxis]
        offmask = np.zeros((bs, bs, 1))
        opamask = 100 * kernel[:, :, np.newaxis]
        self.redmask = np.concatenate((onmask, offmask, offmask, onmask), axis=-1)
        self.strokemask = np.concatenate((onmask, offmask, onmask, opamask), axis=-1)
