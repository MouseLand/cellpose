import PyQt5
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QRadioButton, QWidget, QDialog, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QGridLayout, QPushButton, QLabel, QLineEdit, QDialogButtonBox, QComboBox, QCheckBox
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
import pathlib, os
from superqt import QRangeSlider

from .guiparts import create_channel_choose

Horizontal = QtCore.Qt.Orientation.Horizontal

class Slider(QRangeSlider):
    def __init__(self, parent, name, color):
        super().__init__(Horizontal)
        self.setEnabled(False)
        self.valueChanged.connect(lambda: self.levelChanged(parent))
        self.name = name
        #self._setBarColor(color)
        self.show()
        
    def levelChanged(self, parent):
        parent.level_change(self.name)


class NormWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setGeometry(100,100,900,350)
        self.setWindowTitle('normalization settings')
        self.win = QWidget(self)
        self.l0 = QGridLayout()
        self.win.setLayout(self.l0)

        yoff = 0
        qlabel = QLabel('set custom normalization')
        qlabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        
        qlabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff,0,1,2)

        # choose channels
        self.ChannelChoose, self.ChannelLabels = create_channel_choose()
        for i in range(2):
            yoff+=1
            self.ChannelChoose[i].setFixedWidth(150)
            self.ChannelChoose[i].setCurrentIndex(parent.ChannelChoose[i].currentIndex())
            self.l0.addWidget(self.ChannelLabels[i], yoff, 0,1,1)
            self.l0.addWidget(self.ChannelChoose[i], yoff, 1,1,1)

        # choose parameters        
        labels = ['lower percentile', 'higher percentile', 'tile_norm', 'sharpen', 'norm3D']
        self.edits = []
        yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i+yoff,0,1,1)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            self.edits[-1].setFixedWidth(200)
            self.l0.addWidget(self.edits[-1], i+yoff, 1,1,1)

        yoff+=len(labels)

        yoff+=1
        qlabel = QLabel('(to remove files, click cancel then remove \nfrom folder and reopen train window)')
        self.l0.addWidget(qlabel, yoff,0,2,4)

        # click button
        yoff+=2
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda: self.accept(parent))
        self.buttonBox.rejected.connect(self.reject)
        self.l0.addWidget(self.buttonBox, yoff, 0, 1,4)

        
        # list files in folder
        qlabel = QLabel('filenames')
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0,4,1,1)
        qlabel = QLabel('# of masks')
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0,5,1,1)
    
        for i in range(10):
            if i > len(parent.train_files) - 1:
                break
            elif i==9 and len(parent.train_files) > 10:
                label = '...'
                nmasks = '...'
            else:
                label = os.path.split(parent.train_files[i])[-1]
                nmasks = str(parent.train_labels[i].max())
            qlabel = QLabel(label)
            self.l0.addWidget(qlabel, i+1,4,1,1)
            qlabel = QLabel(nmasks)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i+1, 5,1,1)

    def accept(self, parent):
        # set channels
        for i in range(2):
            parent.ChannelChoose[i].setCurrentIndex(self.ChannelChoose[i].currentIndex())
        # set training params
        parent.training_params = {'model_index': self.ModelChoose.currentIndex(),
                                 'learning_rate': float(self.edits[0].text()), 
                                 'weight_decay': float(self.edits[1].text()), 
                                 'n_epochs':  int(self.edits[2].text()),
                                 'model_name': self.edits[3].text()
                                 }
        self.done(1)

def set_norm(parent):
    NW = NormWindow(parent)
    norm = NW.exec_()
