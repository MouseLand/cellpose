"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import sys, pathlib, warnings

from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QApplication, QScrollBar, QCheckBox, QLabel, QLineEdit
import pyqtgraph as pg

import numpy as np
from scipy.stats import mode
import cv2

from . import guiparts, io
from ..utils import download_url_to_file, masks_to_outlines
from .gui import MainW

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False


def avg3d(C):
    """ smooth value of c across nearby points
        (c is center of grid directly below point)
        b -- a -- b
        a -- c -- a
        b -- a -- b
    """
    Ly, Lx = C.shape
    # pad T by 2
    T = np.zeros((Ly + 2, Lx + 2), "float32")
    M = np.zeros((Ly, Lx), "float32")
    T[1:-1, 1:-1] = C.copy()
    y, x = np.meshgrid(np.arange(0, Ly, 1, int), np.arange(0, Lx, 1, int),
                       indexing="ij")
    y += 1
    x += 1
    a = 1. / 2  #/(z**2 + 1)**0.5
    b = 1. / (1 + 2**0.5)  #(z**2 + 2)**0.5
    c = 1.
    M = (b * T[y - 1, x - 1] + a * T[y - 1, x] + b * T[y - 1, x + 1] + a * T[y, x - 1] +
         c * T[y, x] + a * T[y, x + 1] + b * T[y + 1, x - 1] + a * T[y + 1, x] +
         b * T[y + 1, x + 1])
    M /= 4 * a + 4 * b + c
    return M


def interpZ(mask, zdraw):
    """ find nearby planes and average their values using grid of points
        zfill is in ascending order
    """
    ifill = np.ones(mask.shape[0], "bool")
    zall = np.arange(0, mask.shape[0], 1, int)
    ifill[zdraw] = False
    zfill = zall[ifill]
    zlower = zdraw[np.searchsorted(zdraw, zfill, side="left") - 1]
    zupper = zdraw[np.searchsorted(zdraw, zfill, side="right")]
    for k, z in enumerate(zfill):
        Z = zupper[k] - zlower[k]
        zl = (z - zlower[k]) / Z
        plower = avg3d(mask[zlower[k]]) * (1 - zl)
        pupper = avg3d(mask[zupper[k]]) * zl
        mask[z] = (plower + pupper) > 0.33
    return mask, zfill


def run(image=None):
    from ..io import logger_setup
    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")
    style_path = pathlib.Path.home().joinpath(".cellpose", "style_choice.npy")
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
    MainW_3d(image=image, logger=logger)
    ret = app.exec_()
    sys.exit(ret)


class MainW_3d(MainW):

    def __init__(self, image=None, logger=None):
        # MainW init
        MainW.__init__(self, image=image, logger=logger)

        # add gradZ view
        self.ViewDropDown.insertItem(3, "gradZ")

        # turn off single stroke
        self.SCheckBox.setChecked(False)

        ### add orthoviews and z-bar
        # ortho crosshair lines
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.vLineOrtho = [
            pg.InfiniteLine(angle=90, movable=False),
            pg.InfiniteLine(angle=90, movable=False)
        ]
        self.hLineOrtho = [
            pg.InfiniteLine(angle=0, movable=False),
            pg.InfiniteLine(angle=0, movable=False)
        ]
        self.make_orthoviews()

        # z scrollbar underneath
        self.scroll = QScrollBar(QtCore.Qt.Horizontal)
        self.scroll.setMaximum(10)
        self.scroll.valueChanged.connect(self.move_in_Z)
        self.lmain.addWidget(self.scroll, 40, 9, 1, 30)

        b = 22

        label = QLabel("stitch\nthreshold:")
        label.setToolTip(
            "for 3D volumes, turn on stitch_threshold to stitch masks across planes instead of running cellpose in 3D (see docs for details)"
        )
        label.setFont(self.medfont)
        self.segBoxG.addWidget(label, b, 0, 1, 4)
        self.stitch_threshold = QLineEdit()
        self.stitch_threshold.setText("0.0")
        self.stitch_threshold.setFixedWidth(30)
        self.stitch_threshold.setFont(self.medfont)
        self.stitch_threshold.setToolTip(
            "for 3D volumes, turn on stitch_threshold to stitch masks across planes instead of running cellpose in 3D (see docs for details)"
        )
        self.segBoxG.addWidget(self.stitch_threshold, b, 3, 1, 1)

        label = QLabel("flow3D\nsmooth:")
        label.setToolTip(
            "for 3D volumes, smooth flows by a Gaussian with standard deviation flow3D_smooth (see docs for details)"
        )
        label.setFont(self.medfont)
        self.segBoxG.addWidget(label, b, 4, 1, 3)
        self.flow3D_smooth = QLineEdit()
        self.flow3D_smooth.setText("0.0")
        self.flow3D_smooth.setFixedWidth(30)
        self.flow3D_smooth.setFont(self.medfont)
        self.flow3D_smooth.setToolTip(
            "for 3D volumes, smooth flows by a Gaussian with standard deviation flow3D_smooth (see docs for details)"
        )
        self.segBoxG.addWidget(self.flow3D_smooth, b, 7, 1, 1)

        b+=1
        label = QLabel("anisotropy:")
        label.setToolTip(
            "for 3D volumes, increase in sampling in Z vs XY as a ratio, e.g. set set to 2.0 if Z is sampled half as dense as X or Y (see docs for details)"
        )
        label.setFont(self.medfont)
        self.segBoxG.addWidget(label, b, 0, 1, 3)
        self.anisotropy = QLineEdit()
        self.anisotropy.setText("1.0")
        self.anisotropy.setFixedWidth(30)
        self.anisotropy.setFont(self.medfont)
        self.anisotropy.setToolTip(
            "for 3D volumes, increase in sampling in Z vs XY as a ratio, e.g. set set to 2.0 if Z is sampled half as dense as X or Y (see docs for details)"
        )
        self.segBoxG.addWidget(self.anisotropy, b, 3, 1, 1)

        b+=1
        label = QLabel("min\nsize:")
        label.setToolTip(
            "all masks less than this size in pixels (volume) will be removed"
        )
        label.setFont(self.medfont)
        self.segBoxG.addWidget(label, b, 0, 1, 4)
        self.min_size = QLineEdit()
        self.min_size.setText("15")
        self.min_size.setFixedWidth(50)
        self.min_size.setFont(self.medfont)
        self.min_size.setToolTip(
            "all masks less than this size in pixels (volume) will be removed"
        )
        self.segBoxG.addWidget(self.min_size, b, 3, 1, 1)

        b += 1
        self.orthobtn = QCheckBox("ortho")
        self.orthobtn.setToolTip("activate orthoviews with 3D image")
        self.orthobtn.setFont(self.medfont)
        self.orthobtn.setChecked(False)
        self.l0.addWidget(self.orthobtn, b, 0, 1, 2)
        self.orthobtn.toggled.connect(self.toggle_ortho)

        label = QLabel("dz:")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setFont(self.medfont)
        self.l0.addWidget(label, b, 2, 1, 1)
        self.dz = 10
        self.dzedit = QLineEdit()
        self.dzedit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.dzedit.setText(str(self.dz))
        self.dzedit.returnPressed.connect(self.update_ortho)
        self.dzedit.setFixedWidth(40)
        self.dzedit.setFont(self.medfont)
        self.l0.addWidget(self.dzedit, b, 3, 1, 2)

        label = QLabel("z-aspect:")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setFont(self.medfont)
        self.l0.addWidget(label, b, 5, 1, 2)
        self.zaspect = 1.0
        self.zaspectedit = QLineEdit()
        self.zaspectedit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.zaspectedit.setText(str(self.zaspect))
        self.zaspectedit.returnPressed.connect(self.update_ortho)
        self.zaspectedit.setFixedWidth(40)
        self.zaspectedit.setFont(self.medfont)
        self.l0.addWidget(self.zaspectedit, b, 7, 1, 2)

        b += 1
        # add z position underneath
        self.currentZ = 0
        label = QLabel("Z:")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(label, b, 5, 1, 2)
        self.zpos = QLineEdit()
        self.zpos.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.zpos.setText(str(self.currentZ))
        self.zpos.returnPressed.connect(self.update_ztext)
        self.zpos.setFixedWidth(40)
        self.zpos.setFont(self.medfont)
        self.l0.addWidget(self.zpos, b, 7, 1, 2)

        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename, load_3D=True)

        self.load_3D = True

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        # points is list of strokes

        points_all = np.concatenate(points, axis=0)

        # loop over z values
        median = []
        zdraw = np.unique(points_all[:, 0])
        zrange = np.arange(zdraw.min(), zdraw.max() + 1, 1, int)
        zmin = zdraw.min()
        pix = np.zeros((2, 0), "uint16")
        mall = np.zeros((len(zrange), self.Ly, self.Lx), "bool")
        k = 0
        for z in zdraw:
            ars, acs, vrs, vcs = np.zeros(0, "int"), np.zeros(0, "int"), np.zeros(
                0, "int"), np.zeros(0, "int")
            for stroke in points:
                stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
                iz = stroke[:, 0] == z
                vr = stroke[iz, 1]
                vc = stroke[iz, 2]
                if iz.sum() > 0:
                    # get points inside drawn points
                    mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), "uint8")
                    pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2),
                                   axis=-1)[:, np.newaxis, :]
                    mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
                    ar, ac = np.nonzero(mask)
                    ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
                    # get dense outline
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
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
                        mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), "uint8")
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
            mall[z - zmin, ars, acs] = True
            pix = np.append(pix, np.vstack((ars, acs)), axis=-1)

        mall = mall[:, pix[0].min():pix[0].max() + 1,
                    pix[1].min():pix[1].max() + 1].astype("float32")
        ymin, xmin = pix[0].min(), pix[1].min()
        if len(zdraw) > 1:
            mall, zfill = interpZ(mall, zdraw - zmin)
            for z in zfill:
                mask = mall[z].copy()
                ar, ac = np.nonzero(mask)
                ioverlap = self.cellpix[z + zmin][ar + ymin, ac + xmin] > 0
                if (~ioverlap).sum() < 5:
                    print("WARNING: stroke on plane %d not included due to overlaps" %
                          z)
                elif ioverlap.sum() > 0:
                    mask[ar[ioverlap], ac[ioverlap]] = 0
                    ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of mask
                outlines = masks_to_outlines(mask)
                vr, vc = np.nonzero(outlines)
                vr, vc = vr + ymin, vc + xmin
                ar, ac = ar + ymin, ac + xmin
                self.draw_mask(z + zmin, ar, ac, vr, vc, color)

        self.zdraw.append(zdraw)

        return median

    def move_in_Z(self):
        if self.loaded:
            self.currentZ = min(self.NZ, max(0, int(self.scroll.value())))
            self.zpos.setText(str(self.currentZ))
            self.update_plot()
            self.draw_layer()
            self.update_layer()

    def make_orthoviews(self):
        self.pOrtho, self.imgOrtho, self.layerOrtho = [], [], []
        for j in range(2):
            self.pOrtho.append(
                pg.ViewBox(lockAspect=True, name=f"plotOrtho{j}",
                           border=[100, 100, 100], invertY=True, enableMouse=False))
            self.pOrtho[j].setMenuEnabled(False)

            self.imgOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.imgOrtho[j].autoDownsample = False

            self.layerOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.layerOrtho[j].setLevels([0., 255.])

            #self.pOrtho[j].scene().contextMenuItem = self.pOrtho[j]
            self.pOrtho[j].addItem(self.imgOrtho[j])
            self.pOrtho[j].addItem(self.layerOrtho[j])
            self.pOrtho[j].addItem(self.vLineOrtho[j], ignoreBounds=False)
            self.pOrtho[j].addItem(self.hLineOrtho[j], ignoreBounds=False)

        self.pOrtho[0].linkView(self.pOrtho[0].YAxis, self.p0)
        self.pOrtho[1].linkView(self.pOrtho[1].XAxis, self.p0)

    def add_orthoviews(self):
        self.yortho = self.Ly // 2
        self.xortho = self.Lx // 2
        if self.NZ > 1:
            self.update_ortho()

        self.win.addItem(self.pOrtho[0], 0, 1, rowspan=1, colspan=1)
        self.win.addItem(self.pOrtho[1], 1, 0, rowspan=1, colspan=1)

        qGraphicsGridLayout = self.win.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 2)
        qGraphicsGridLayout.setColumnStretchFactor(1, 1)
        qGraphicsGridLayout.setRowStretchFactor(0, 2)
        qGraphicsGridLayout.setRowStretchFactor(1, 1)

        self.pOrtho[0].setYRange(0, self.Lx)
        self.pOrtho[0].setXRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
        self.pOrtho[1].setYRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
        self.pOrtho[1].setXRange(0, self.Ly)

        self.p0.addItem(self.vLine, ignoreBounds=False)
        self.p0.addItem(self.hLine, ignoreBounds=False)
        self.p0.setYRange(0, self.Lx)
        self.p0.setXRange(0, self.Ly)

        self.win.show()
        self.show()

    def remove_orthoviews(self):
        self.win.removeItem(self.pOrtho[0])
        self.win.removeItem(self.pOrtho[1])
        self.p0.removeItem(self.vLine)
        self.p0.removeItem(self.hLine)
        self.win.show()
        self.show()

    def update_crosshairs(self):
        self.yortho = min(self.Ly - 1, max(0, int(self.yortho)))
        self.xortho = min(self.Lx - 1, max(0, int(self.xortho)))
        self.vLine.setPos(self.xortho)
        self.hLine.setPos(self.yortho)
        self.vLineOrtho[1].setPos(self.xortho)
        self.hLineOrtho[1].setPos(self.zc)
        self.vLineOrtho[0].setPos(self.zc)
        self.hLineOrtho[0].setPos(self.yortho)

    def update_ortho(self):
        if self.NZ > 1 and self.orthobtn.isChecked():
            dzcurrent = self.dz
            self.dz = min(100, max(3, int(self.dzedit.text())))
            self.zaspect = max(0.01, min(100., float(self.zaspectedit.text())))
            self.dzedit.setText(str(self.dz))
            self.zaspectedit.setText(str(self.zaspect))
            if self.dz != dzcurrent:
                self.pOrtho[0].setXRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
                self.pOrtho[1].setYRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
            dztot = min(self.NZ, self.dz * 2)
            y = self.yortho
            x = self.xortho
            z = self.currentZ
            if dztot == self.NZ:
                zmin, zmax = 0, self.NZ
            else:
                if z - self.dz < 0:
                    zmin = 0
                    zmax = zmin + self.dz * 2
                elif z + self.dz >= self.NZ:
                    zmax = self.NZ
                    zmin = zmax - self.dz * 2
                else:
                    zmin, zmax = z - self.dz, z + self.dz
            self.zc = z - zmin
            self.update_crosshairs()
            if self.view == 0 or self.view == 4:
                for j in range(2):
                    if j == 0:
                        if self.view == 0:
                            image = self.stack[zmin:zmax, :, x].transpose(1, 0, 2).copy()
                        else:
                            image = self.stack_filtered[zmin:zmax, :,
                                                        x].transpose(1, 0, 2).copy()
                    else:
                        image = self.stack[
                            zmin:zmax,
                            y, :].copy() if self.view == 0 else self.stack_filtered[zmin:zmax,
                                                                             y, :].copy()
                    if self.nchan == 1:
                        # show single channel
                        image = image[..., 0]
                    if self.color == 0:
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                        if self.nchan > 1:
                            levels = np.array([
                                self.saturation[0][self.currentZ],
                                self.saturation[1][self.currentZ],
                                self.saturation[2][self.currentZ]
                            ])
                            self.imgOrtho[j].setLevels(levels)
                        else:
                            self.imgOrtho[j].setLevels(
                                self.saturation[0][self.currentZ])
                    elif self.color > 0 and self.color < 4:
                        if self.nchan > 1:
                            image = image[..., self.color - 1]
                        self.imgOrtho[j].setImage(image, autoLevels=False,
                                                  lut=self.cmap[self.color])
                        if self.nchan > 1:
                            self.imgOrtho[j].setLevels(
                                self.saturation[self.color - 1][self.currentZ])
                        else:
                            self.imgOrtho[j].setLevels(
                                self.saturation[0][self.currentZ])
                    elif self.color == 4:
                        if image.ndim > 2:
                            image = image.astype("float32").mean(axis=2).astype("uint8")
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                        self.imgOrtho[j].setLevels(self.saturation[0][self.currentZ])
                    elif self.color == 5:
                        if image.ndim > 2:
                            image = image.astype("float32").mean(axis=2).astype("uint8")
                        self.imgOrtho[j].setImage(image, autoLevels=False,
                                                  lut=self.cmap[0])
                        self.imgOrtho[j].setLevels(self.saturation[0][self.currentZ])
                self.pOrtho[0].setAspectLocked(lock=True, ratio=self.zaspect)
                self.pOrtho[1].setAspectLocked(lock=True, ratio=1. / self.zaspect)

            else:
                image = np.zeros((10, 10), "uint8")
                self.imgOrtho[0].setImage(image, autoLevels=False, lut=None)
                self.imgOrtho[0].setLevels([0.0, 255.0])
                self.imgOrtho[1].setImage(image, autoLevels=False, lut=None)
                self.imgOrtho[1].setLevels([0.0, 255.0])

        zrange = zmax - zmin
        self.layer_ortho = [
            np.zeros((self.Ly, zrange, 4), "uint8"),
            np.zeros((zrange, self.Lx, 4), "uint8")
        ]
        if self.masksOn:
            for j in range(2):
                if j == 0:
                    cp = self.cellpix[zmin:zmax, :, x].T
                else:
                    cp = self.cellpix[zmin:zmax, y]
                self.layer_ortho[j][..., :3] = self.cellcolors[cp, :]
                self.layer_ortho[j][..., 3] = self.opacity * (cp > 0).astype("uint8")
                if self.selected > 0:
                    self.layer_ortho[j][cp == self.selected] = np.array(
                        [255, 255, 255, self.opacity])

        if self.outlinesOn:
            for j in range(2):
                if j == 0:
                    op = self.outpix[zmin:zmax, :, x].T
                else:
                    op = self.outpix[zmin:zmax, y]
                self.layer_ortho[j][op > 0] = np.array(self.outcolor).astype("uint8")

        for j in range(2):
            self.layerOrtho[j].setImage(self.layer_ortho[j])
        self.win.show()
        self.show()

    def toggle_ortho(self):
        if self.orthobtn.isChecked():
            self.add_orthoviews()
        else:
            self.remove_orthoviews()

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
            elif self.loaded and not self.in_stroke:
                if self.orthobtn.isChecked():
                    items = self.win.scene().items(event.scenePos())
                    for x in items:
                        if x == self.p0:
                            pos = self.p0.mapSceneToView(event.scenePos())
                            x = int(pos.x())
                            y = int(pos.y())
                            if y >= 0 and y < self.Ly and x >= 0 and x < self.Lx:
                                self.yortho = y
                                self.xortho = x
                                self.update_ortho()

    def update_plot(self):
        super().update_plot()
        if self.NZ > 1 and self.orthobtn.isChecked():
            self.update_ortho()
        self.win.show()
        self.show()

    def keyPressEvent(self, event):
        if self.loaded:
            if not (event.modifiers() &
                    (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier |
                     QtCore.Qt.AltModifier) or self.in_stroke):
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                    if self.NZ > 1:
                        if event.key() == QtCore.Qt.Key_Left:
                            self.currentZ = max(0, self.currentZ - 1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
                        elif event.key() == QtCore.Qt.Key_Right:
                            self.currentZ = min(self.NZ - 1, self.currentZ + 1)
                            self.scroll.setValue(self.currentZ)
                            updated = True
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
                        self.currentZ = max(0, self.currentZ - 1)
                        self.scroll.setValue(self.currentZ)
                        updated = True
                    elif event.key() == QtCore.Qt.Key_Right or event.key(
                    ) == QtCore.Qt.Key_D:
                        self.currentZ = min(self.NZ - 1, self.currentZ + 1)
                        self.scroll.setValue(self.currentZ)
                        updated = True
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

    def update_ztext(self):
        zpos = self.currentZ
        try:
            zpos = int(self.zpos.text())
        except:
            print("ERROR: zposition is not a number")
        self.currentZ = max(0, min(self.NZ - 1, zpos))
        self.zpos.setText(str(self.currentZ))
        self.scroll.setValue(self.currentZ)
