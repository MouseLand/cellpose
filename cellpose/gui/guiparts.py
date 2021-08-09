from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QRadioButton, QWidget, QDialog, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QGridLayout, QPushButton, QLabel
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
import pathlib

def make_quadrants(parent):
    """ make quadrant buttons """
    parent.quadbtns = QButtonGroup(parent)
    for b in range(9):
        btn = QuadButton(b, ' '+str(b+1), parent)
        parent.quadbtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0 + parent.quadbtns.button(b).ypos, 29 + parent.quadbtns.button(b).xpos, 1, 1)
        btn.setEnabled(True)
        b += 1
    parent.quadbtns.setExclusive(True)

class QuadButton(QPushButton):
    """ custom QPushButton class for quadrant plotting
        requires buttons to put into a QButtonGroup (parent.quadbtns)
         allows only 1 button to pressed at a time
    """
    def __init__(self, bid, Text, parent=None):
        super(QuadButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleUnpressed)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.setMaximumWidth(22)
        self.xpos = bid%3
        self.ypos = int(np.floor(bid/3))
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        for b in range(9):
            if parent.quadbtns.button(b).isEnabled():
                parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        self.xrange = np.array([self.xpos-.2, self.xpos+1.2]) * parent.Lx/3
        self.yrange = np.array([self.ypos-.2, self.ypos+1.2]) * parent.Ly/3
        # change the zoom
        parent.p0.setXRange(self.xrange[0], self.xrange[1])
        parent.p0.setYRange(self.yrange[0], self.yrange[1])
        parent.show()

def horizontal_slider_style():
    return """QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: black;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
            background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
                stop: 0 black, stop: 1 rgb(150,255,150));
            background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
                stop: 0 black, stop: 1 rgb(150,255,150));
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::add-page:horizontal {
            background: black;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eee, stop:1 #ccc);
            border: 1px solid #777;
            width: 13px;
            margin-top: -2px;
            margin-bottom: -2px;
            border-radius: 4px;
            }

            QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #fff, stop:1 #ddd);
            border: 1px solid #444;
            border-radius: 4px;
            }

            QSlider::sub-page:horizontal:disabled {
            background: #bbb;
            border-color: #999;
            }

            QSlider::add-page:horizontal:disabled {
            background: #eee;
            border-color: #999;
            }

            QSlider::handle:horizontal:disabled {
            background: #eee;
            border: 1px solid #aaa;
            border-radius: 4px;
            }"""

class ExampleGUI(QDialog):
    def __init__(self, parent=None):
        super(ExampleGUI, self).__init__(parent)
        self.setGeometry(100,100,1300,900)
        self.setWindowTitle('GUI layout')
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        guip_path = pathlib.Path.home().joinpath('.cellpose', 'cellpose_gui.png')
        guip_path = str(guip_path.resolve())
        pixmap = QPixmap(guip_path)
        label = QLabel(self)
        label.setPixmap(pixmap)
        pixmap.scaled
        layout.addWidget(label, 0, 0, 1, 1)

class HelpWindow(QDialog):
    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        self.setGeometry(100,100,700,800)
        self.setWindowTitle('cellpose help')
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        
        text = ('''
            <p class="has-line-data" data-line-start="5" data-line-end="6">Main GUI mouse controls:</p>
            <ul>
            <li class="has-line-data" data-line-start="7" data-line-end="8">Pan  = left-click  + drag</li>
            <li class="has-line-data" data-line-start="8" data-line-end="9">Zoom = scroll wheel (or +/= and - buttons) </li>
            <li class="has-line-data" data-line-start="9" data-line-end="10">Full view = double left-click</li>
            <li class="has-line-data" data-line-start="10" data-line-end="11">Select mask = left-click on mask</li>
            <li class="has-line-data" data-line-start="11" data-line-end="12">Delete mask = Ctrl (or COMMAND on Mac) + left-click</li>
            <li class="has-line-data" data-line-start="11" data-line-end="12">Merge masks = Alt + left-click (will merge last two)</li>
            <li class="has-line-data" data-line-start="12" data-line-end="13">Start draw mask = right-click</li>
            <li class="has-line-data" data-line-start="13" data-line-end="15">End draw mask = right-click, or return to circle at beginning</li>
            </ul>
            <p class="has-line-data" data-line-start="15" data-line-end="16">Overlaps in masks are NOT allowed. If you draw a mask on top of another mask, it is cropped so that it doesn’t overlap with the old mask. Masks in 2D should be single strokes (single stroke is checked). If you want to draw masks in 3D (experimental), then you can turn this option off and draw a stroke on each plane with the cell and then press ENTER. 3D labelling will fill in planes that you have not labelled so that you do not have to as densely label.</p>
            <p class="has-line-data" data-line-start="17" data-line-end="18">!NOTE!: The GUI automatically saves after you draw a mask in 2D but NOT after 3D mask drawing and NOT after segmentation. Save in the file menu or with Ctrl+S. The output file is in the same folder as the loaded image with <code>_seg.npy</code> appended.</p>
            <table class="table table-striped table-bordered">
            <br><br>
            <thead>
            <tr>
            <th>Keyboard shortcuts</th>
            <th>Description</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>=/+  button // - button</td>
            <td>zoom in // zoom out</td>
            </tr>
            <tr>
            <td>CTRL+Z</td>
            <td>undo previously drawn mask/stroke</td>
            </tr>
            <tr>
            <td>CTRL+Y</td>
            <td>undo remove mask</td>
            </tr>
            <tr>
            <td>CTRL+0</td>
            <td>clear all masks</td>
            </tr>
            <tr>
            <td>CTRL+L</td>
            <td>load image (can alternatively drag and drop image)</td>
            </tr>
            <tr>
            <td>CTRL+S</td>
            <td>SAVE MASKS IN IMAGE to <code>_seg.npy</code> file</td>
            </tr>
            <tr>
            <td>CTRL+P</td>
            <td>load <code>_seg.npy</code> file (note: it will load automatically with image if it exists)</td>
            </tr>
            <tr>
            <td>CTRL+M</td>
            <td>load masks file (must be same size as image with 0 for NO mask, and 1,2,3… for masks)</td>
            </tr>
            <tr>
            <td>CTRL+N</td>
            <td>load numpy stack (NOT WORKING ATM)</td>
            </tr>
            <tr>
            <td>A/D or LEFT/RIGHT</td>
            <td>cycle through images in current directory</td>
            </tr>
            <tr>
            <td>W/S or UP/DOWN</td>
            <td>change color (RGB/gray/red/green/blue)</td>
            </tr>
            <tr>
            <td>PAGE-UP / PAGE-DOWN</td>
            <td>change to flows and cell prob views (if segmentation computed)</td>
            </tr>
            <tr>
            <td>, / .</td>
            <td>increase / decrease brush size for drawing masks</td>
            </tr>
            <tr>
            <td>X</td>
            <td>turn masks ON or OFF</td>
            </tr>
            <tr>
            <td>Z</td>
            <td>toggle outlines ON or OFF</td>
            </tr>
            <tr>
            <td>C</td>
            <td>cycle through labels for image type (saved to <code>_seg.npy</code>)</td>
            </tr>
            </tbody>
            </table>
            <p class="has-line-data" data-line-start="36" data-line-end="37"><strong>Segmentation options (2D only) </strong></p>
            <p class="has-line-data" data-line-start="38" data-line-end="39">SIZE: you can manually enter the approximate diameter for your cells, or press “calibrate” to let the model estimate it. The size is represented by a disk at the bottom of the view window (can turn this disk of by unchecking “scale disk on”).</p>
            <p class="has-line-data" data-line-start="40" data-line-end="41">use GPU: if you have specially installed the cuda version of mxnet, then you can activate this, but it won’t give huge speedups when running single 2D images in the GUI.</p>
            <p class="has-line-data" data-line-start="42" data-line-end="43">MODEL: there is a <em>cytoplasm</em> model and a <em>nuclei</em> model, choose what you want to segment</p>
            <p class="has-line-data" data-line-start="44" data-line-end="45">CHAN TO SEG: this is the channel in which the cytoplasm or nuclei exist</p>
            <p class="has-line-data" data-line-start="46" data-line-end="47">CHAN2 (OPT): if <em>cytoplasm</em> model is chosen, then choose the nuclear channel for this option</p>
            ''')
        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()

class TypeRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(TypeRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = self.parent.cell_types
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            button.setStyleSheet('color: rgb(190,190,190);')
            button.setFont(QtGui.QFont("Arial", 10))
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.l0.addWidget(button, row+b,col,1,2)
        self.setExclusive(True)
        #self.buttons.

    def btnpress(self, parent):
       b = self.checkedId()
       self.parent.cell_type = b

class RGBRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(RGBRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = ["image", "gradXY", "cellprob", "gradZ"]
        #self.buttons = QButtonGroup()
        self.dropdown = []
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            button.setStyleSheet('color: white;')
            button.setFont(QtGui.QFont("Arial", 10))
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.l0.addWidget(button, row+b,col,1,1)
        self.setExclusive(True)
        #self.buttons.

    def btnpress(self, parent):
       b = self.checkedId()
       self.parent.view = b
       if self.parent.loaded:
           self.parent.update_plot()


class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse,
                            invertY, enableMenu, name, invertX)
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
        if ev.text() == '-':
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ['+', '=']:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()
    
    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        if self.parent is None or (self.parent is not None and not self.parent.in_stroke):
            ev.accept()  ## we accept all buttons

            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos
            dif = dif * -1

            ## Ignore axes if mouse is disabled
            mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1-axis] = 0.0

            ## Scale or translate based on mouse button
            if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
                if self.state['mouseMode'] == pg.ViewBox.RectMode:
                    if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                        #print "finish"
                        self.rbScaleBox.hide()
                        ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                        ax = self.childGroup.mapRectFromParent(ax)
                        self.showAxRect(ax)
                        self.axHistoryPointer += 1
                        self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                    else:
                        ## update shape of scale box
                        self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                else:
                    tr = dif*mask
                    tr = self.mapToView(tr) - self.mapToView(Point(0,0))
                    x = tr.x() if mask[0] == 1 else None
                    y = tr.y() if mask[1] == 1 else None

                    self._resetTarget()
                    if x is not None or y is not None:
                        self.translateBy(x=x, y=y)
                    self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

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

    sigImageChanged = QtCore.pyqtSignal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        #self.image=None
        #self.viewbox=viewbox
        self.levels = np.array([0,255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = 'row-major'
        self.removable = False

        self.parent = parent
        #kernel[1,1] = 1
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        if self.parent.masksOn or self.parent.outlinesOn:
            if  self.parent.loaded and (ev.button()==QtCore.Qt.RightButton or 
                    ev.modifiers() == QtCore.Qt.ShiftModifier and not ev.double()):
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
                y,x = int(ev.pos().y()), int(ev.pos().x())
                if y>=0 and y<self.parent.Ly and x>=0 and x<self.parent.Lx:
                    if ev.button()==QtCore.Qt.LeftButton and not ev.double():
                        idx = self.parent.cellpix[self.parent.currentZ][y,x]
                        if idx > 0:
                            if ev.modifiers()==QtCore.Qt.ControlModifier:
                                # delete mask selected
                                self.parent.remove_cell(idx)
                            elif ev.modifiers()==QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif self.parent.masksOn:
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                        elif self.parent.masksOn:
                            self.parent.unselect_cell()
                    else:
                        ev.ignore()
                        return


    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    self.end_stroke()
                    self.parent.in_stroke = False
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            #ev.acceptClicks(QtCore.Qt.LeftButton)

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                        pen=pg.mkPen(color=(255,0,0), width=self.parent.brush_size),
                                        size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size*3)
        thresh_in = max(3, self.parent.brush_size*1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (((stroke[1:,1:] - stroke[:1,1:][np.newaxis,:,:])**2).sum(axis=-1))**0.5
            dist = dist.flatten()
            #print(dist)
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4,first_left+1):] < thresh_in).sum()
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
            ioutline = self.parent.current_stroke[:,3]==1
            self.parent.current_point_set.extend(list(self.parent.current_stroke[ioutline]))
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if len(self.parent.current_point_set) > 0 and self.parent.autosave:
            self.parent.add_set()

    def tabletEvent(self, ev):
        pass
        #print(ev.device())
        #print(ev.pointerType())
        #print(ev.pressure())

    def drawAt(self, pos, ev=None):
        mask = self.greenmask
        set = self.parent.current_point_set
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        kcent = kc.copy()
        if tx[0]<=0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx    = sx
            kcent[0] = 0
        if ty[0]<=0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty    = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly-1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1]-tx[0]-1
        if ty[1] >= self.parent.Lx-1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1]-ty[0]-1


        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        self.image[ts] = mask[ss]

        for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs,bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
                                 int(np.floor(kernel.shape[1]/2))]
        onmask = 255 * kernel[:,:,np.newaxis]
        offmask = np.zeros((bs,bs,1))
        opamask = 100 * kernel[:,:,np.newaxis]
        self.redmask = np.concatenate((onmask,offmask,offmask,onmask), axis=-1)
        self.greenmask = np.concatenate((onmask,offmask,onmask,opamask), axis=-1)


class RangeSlider(QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged

        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """
    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QStyle.SC_None
        self.hover_control = QStyle.SC_None
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Vertical)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(\
                "QSlider::handle:horizontal {\
                background-color: white;\
                border: 1px solid #5c5c5c;\
                border-radius: 0px;\
                border-color: black;\
                height: 8px;\
                width: 6px;\
                margin: -8px 2; \
                }")


        #self.opt = QStyleOptionSlider()
        #self.opt.orientation=QtCore.Qt.Vertical
        #self.initStyleOption(self.opt)
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def level_change(self):
        if self.parent is not None:
            if self.parent.loaded:
                self.parent.ops_plot['saturation'] = [self._low, self._high]
                self.parent.update_plot()

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QPainter(self)
        style = QApplication.style()

        for i, value in enumerate([self._low, self._high]):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QStyle.SC_SliderHandle#QStyle.SC_SliderGroove | QStyle.SC_SliderHandle
            else:
                opt.subControls = QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QStyle.CC_Slider, opt, painter, self)


    def mousePressEvent(self, event):
        event.accept()

        style = QApplication.style()
        button = event.button()
        # In a normal slider control, when the user clicks on a point in the
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts
        if button:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)

                    break

            if self.active_slider < 0:
                self.pressed_control = QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.level_change()

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()


    def __pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)
