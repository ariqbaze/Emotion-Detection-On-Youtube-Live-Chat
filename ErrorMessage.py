from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QLabel

class ErrorMessage(QDialog):
    def __init__(self):
        super(ErrorMessage, self).__init__()
        # Load UI
        uic.loadUi("ui/Message-1.ui", self)
        self.setFixedSize(411, 111)

        # Define Widget
        self.label = self.findChild(QLabel, "label")

        # Do Something

    def errormess(self, error):
        self.label.setText(error)
        self.exec()

    def closeEvent(self, event):
        event.accept()