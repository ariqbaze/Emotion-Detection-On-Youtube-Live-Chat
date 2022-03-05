from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow,QLabel
from PyQt5 import uic, QtCore



class ShowMatrix(QMainWindow):
    def __init__(self):
        super(ShowMatrix, self).__init__()
        # Load UI
        uic.loadUi("ui/MatrixGraph.ui", self)
        self.setFixedSize(520, 550)

        self.timestr = ""
        self.pname = ""

        # Define Widget
        self.label = self.findChild(QLabel, "label")
        self.label2 = self.findChild(QLabel, "label_2")

        # Do Something

    def matrix_clicked(self, time):
        self.timestr = time
        self.pname = f'model-data/matrix/matrix-{self.timestr}.png'
        self.pixmap = QPixmap(self.pname)
        self.pixmap_scaled = self.pixmap.scaled(self.label.size(), transformMode=QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(self.pixmap_scaled)


# Initial App
# app = QApplication(sys.argv)
# UIWindow = ShowGraph()