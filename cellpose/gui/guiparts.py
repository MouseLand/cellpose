"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtGui import QPainter, QPixmap, QImage, QFont
from qtpy.QtWidgets import QApplication, QRadioButton, QWidget, QDialog, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QGridLayout, QPushButton, QLabel, QLineEdit, QDialogButtonBox, QComboBox, QCheckBox, QDockWidget
from qtpy.QtCore import QEvent
from qtpy.QtGui import QFont
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
import pathlib, os
from . import io



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


def create_channel_choose():
    # choose channel
    ChannelChoose = [QComboBox(), QComboBox()]
    ChannelLabels = []
    ChannelChoose[0].addItems(["gray", "red", "green", "blue"])
    ChannelChoose[1].addItems(["none", "red", "green", "blue"])
    cstr = ["chan to segment:", "chan2 (optional): "]
    for i in range(2):
        ChannelLabels.append(QLabel(cstr[i]))
        if i == 0:
            ChannelLabels[i].setToolTip(
                "this is the channel in which the cytoplasm or nuclei exist \
            that you want to segment"q)
            ChannelChoose[i].setToolTip(
                "this is the channel in which the cytoplasm or nuclei exist \
            that you want to segment")
        else:
            ChannelLabels[i].setToolTip(
                "if <em>cytoplasm</em> model is chosen, and you also have a \
            nuclear channel, then choose the nuclear channel for this option")
            ChannelChoose[i].setToolTip(
                "if <em>cytoplasm</em> model is chosen, and you also have a \
            nuclear channel, then choose the nuclear channel for this option")

    return ChannelChoose, ChannelLabels


class ModelButton(QPushButton):

    def __init__(self, parent, model_name, text):
        super().__init__()
        self.setEnabled(False)
        self.setText(text)
        self.setFont(parent.boldfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name if "cyto3" not in model_name else "cyto3"

    def press(self, parent):
        parent.compute_segmentation(model_name=self.model_name)


class DenoiseButton(QPushButton):

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
        elif self.model_type != "none":
            parent.compute_denoise_model(model_type=self.model_type)
        else:
            parent.clear_restore()
        parent.set_restore_button()

class TrainWindow(QDialog):

    def __init__(self, parent, model_strings):
        super().__init__(parent)
        self.setGeometry(100, 100, 900, 350)
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
        self.ModelChoose.addItems(["scratch"])
        self.ModelChoose.setFixedWidth(150)
        self.ModelChoose.setCurrentIndex(parent.training_params["model_index"])
        self.l0.addWidget(self.ModelChoose, yoff, 1, 1, 1)
        qlabel = QLabel("initial model: ")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff, 0, 1, 1)

        # choose channels
        self.ChannelChoose, self.ChannelLabels = create_channel_choose()
        for i in range(2):
            yoff += 1
            self.ChannelChoose[i].setFixedWidth(150)
            self.ChannelChoose[i].setCurrentIndex(
                parent.ChannelChoose[i].currentIndex())
            self.l0.addWidget(self.ChannelLabels[i], yoff, 0, 1, 1)
            self.l0.addWidget(self.ChannelChoose[i], yoff, 1, 1, 1)

        # choose parameters
        labels = ["learning_rate", "weight_decay", "n_epochs", "model_name"]
        self.edits = []
        self.parameter_explanations = ["The learning rate determines how quickly or slowly the model learns from data. A higher learning rate may lead to faster learning but could cause the model to overshoot the optimal solution. Conversely, a lower learning rate may result in slower learning but is safer and more likely to find the best solution.",
                                       "Weight decay helps prevent overfitting by penalizing large parameter values in the model. \n Increasing weight decay encourages the model to learn simpler patterns from the data,\n improving its ability to generalize to new, unseen examples.",
                                       "The number of times the entire dataset is passed forward and backward through the machine learning model during training. Increasing the number of epochs allows the model to see the data more times, potentially improving its accuracy. However, too many epochs can lead to overfitting, where the model memorizes the training data instead of learning generalizable patterns.",
                     ""]
        yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            qlabel.setToolTip(self.parameter_explanations[i])
            self.l0.addWidget(qlabel, i + yoff, 0, 1, 1)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            self.edits[-1].setFixedWidth(200)
            self.l0.addWidget(self.edits[-1], i + yoff, 1, 1, 1)

        yoff += len(labels)

        yoff += 1
        self.use_norm = QCheckBox(f"use restored/filtered image")
        self.use_norm.setChecked(True)
        #self.l0.addWidget(self.use_norm, yoff, 0, 2, 4)

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
        # set channels
        for i in range(2):
            parent.ChannelChoose[i].setCurrentIndex(
                self.ChannelChoose[i].currentIndex())
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

        guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")
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
        self.setGeometry(200, 200, 1000, 700)
        self.setMinimumSize(300, 200)
        self.setWindowTitle("Training Instructions")

        layout = QGridLayout()
        self.setLayout(layout)

        text_file = pathlib.Path(__file__).parent.joinpath(
            "guitrainhelpwindowtext.html")
        with open(str(text_file.resolve()), "r") as f:
            text = f.read()

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 0, 0, 1, 1)

        # Dropdown menu for font size
        self.font_size_combo = QComboBox(self)
        self.font_size_combo.addItems([str(size) for size in range(8, 45, 3)])
        # Set fixed size (width, height)
        self.font_size_combo.setFixedSize(55, 25)
        # Set default index to 17
        self.font_size_combo.setCurrentText("17")

        # The line "self.font_size_combo.currentIndexChanged.connect(self.adjust_font_size)"
        # directly connects the currentIndexChanged signal of the "self.font_size_combo" object
        # to the "self.adjust_font_size" method.
        # The "self.adjust_font_size" method is automatically called whenever the index changes
        # in ("self.font_size_combo") the dropdown menu.
        self.font_size_combo.currentIndexChanged.connect(self.adjust_font_size)
        layout.addWidget(self.font_size_combo, 1, 0, 1, 1)

        self.adjust_font_size()  # Initial font size adjustment

        self.show()

    def adjust_font_size(self):
        # Get the current font size from the combo box
        font_size = int(self.font_size_combo.currentText())
        # Calculate the new font size based on window height and width
        new_font_size = max(5, int((self.height() * self.width())**0.5 / 45))
        # Set the font size for the label
        # The if statement prevents the font from being too big to fit the screen/window
        if new_font_size < font_size:
            font = QFont("Arial", new_font_size)
        else:
            font = QFont("Arial", font_size)
        self.label.setFont(font)

    def resizeEvent(self, event):
        # Call adjust_font_size when the window is resized
        self.adjust_font_size()
        super().resizeEvent(event)

# window displaying a minimap of the current image
class MinimapWindow(QWidget):
    """
    Method to initialize the Minimap Window.
    It creates a title for the window and a QWidget with a basic layout.
    It also takes the current picture stored in parent.filename and loads it in a QPixmap.
    The proportions of this image stay constant.
    This is then set to a QLabel that will display the image.
    """
    def __init__(self, parent=None):
        super(MinimapWindow, self).__init__(parent)
        # Set the title and geometry of the window
        self.title = "Minimap"
        self.setWindowTitle(self.title)

        # In practice, this line allows the window to be resized infinitely big, but sets the
        # given dimensions as a lower boundry. The image is first presented in its original size.
        self.setGeometry(100, 100, 400, 300)

        # Create a QWidget and set its layout to QGridLayout
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        # Create a QLabel to display the image
        self.label = QLabel(self)
        self.label.setScaledContents(False)  # Allow the image to scale with the QLabel (does not maintain aspect ratio)

        # load the image
        self.filename = parent.filename

        # Load the default image into a QPixmap
        self.pixmap = QPixmap(self.filename)

        # This line scales the QPixmap object 'self.pixmap' to a new size of 600x400 pixels.
        # The aspect ratio is maintained to avoid distortions of the image.
        self.pixmap = self.pixmap.scaled(600, 400, QtCore.Qt.KeepAspectRatio)

        # Set the QPixmap to the QLabel (that will display the image)
        self.label.setPixmap(self.pixmap)

        # Add the QLabel to the layout
        layout.addWidget(self.label, 0, 0, 1, 1)

        self.update_image(self.filename)

        # This line sets the size of the window to match the width and height of the pixmap.
        self.setFixedSize(self.pixmap.width(), self.pixmap.height())
        # This line sets the size of the label to match the width and height of the pixmap.
        self.label.setFixedSize(self.pixmap.width(), self.pixmap.height())


    def update_image(self, image):
        """
        Method to update the displayed image.
        If the image is not None, it loads the image, creates a QImage object from the image file,
        creates a QPixmap from the QImage, and sets the QPixmap to the QLabel.
        If the image is None, it sets the QLabel's text to "No image available".
        """
        if image is not None:
            self.filename = image

            # Create QImage object from the image file.
            # Casts the image to a QImage object which adds functionality to easily mutate the image.
            qimage = QImage(self.filename)
            # Create QPixmap from the QImage
            pixmap = QPixmap.fromImage(qimage)
            # Set the QPixmap to the QLabel
            self.label.setPixmap(pixmap)

            # Create a QDockWidget to accommodate the minimap image
            self.dock = QDockWidget("Minimap", self)
            # Set the dock to use our resize methode.
            self.dock.resizeEvent = self.resizeEvent
            # Set the QLabel as the widget for the dock
            self.dock.setWidget(self.label)
            # Set the dock to be floating, so it is detached from the main window (can be freely moved around)
            self.dock.setFloating(True)
        else:
            self.label.setText("No image available")

    def adjust_image_size(self, new_size=None):
        """
        Adjusts the size of the image to fit the dock. If a new size is provided,
        the image is resized to this new size. If no new size is provided, the image
        is resized to the current size of the dock. The aspect ratio of the image is
        maintained during the resizing.

        Parameters:
        new_size (QSize, optional): The new size to which the image should be resized.
                                    If not provided, the current width and height of the window are used.
        """

        # Default to the size of the dock if no new size is provided
        if new_size is None:
            new_size = self.dock.size()

        # Create a resized version of the pixmap with the new size while keeping the aspect ratio
        resized_pixmap = self.pixmap.scaled(new_size, QtCore.Qt.KeepAspectRatio)
        # Set the resized pixmap to the label
        self.label.setPixmap(resized_pixmap)

    def resizeEvent(self, event):
        """
        Overrides the parent class' resizeEvent method. This method is called when
        the dock is resized. It resizes the image to fit the new size of the dock
        and then calls the parent class' resizeEvent method.

        Parameters:
        event (QResizeEvent): The event parameters for the resize event.
        """

        # Resize the image to fit the new size of the dock
        self.adjust_image_size()
        super().resizeEvent(event)

        """
        This method overrides the parent class' closeEvent method.
        It informs the MainW class that the minimap has been closed,
        so that the menu button can be untoggled.
        This is called automatically when the window is closed.
        """
        def closeEvent(self, event:QEvent):
            # Notify the parent that the window is closing
            self.parent.minimap_closed()
            event.accept()  # Accept the event and close the window


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
        #self.image=None
        #self.viewbox=viewbox
        self.levels = np.array([0, 255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = "row-major"
        self.removable = False

        self.parent = parent
        #kernel[1,1] = 1
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
        #QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    #self.parent.in_stroke = False
                    self.end_stroke()
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            #ev.acceptClicks(QtCore.Qt.LeftButton)

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
            #print(dist)
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
        #print(ev.device())
        #print(ev.pointerType())
        #print(ev.pressure())

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
