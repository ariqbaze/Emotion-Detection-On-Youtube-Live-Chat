import os
import re
import time
from PyQt5.QtWidgets import QMainWindow, QApplication, \
    QLabel, QLineEdit, QProgressBar, \
    QPushButton, QFileDialog, QWidget
from PyQt5 import uic
from InputData import InputData
from MakeModel import MakeModel
from LoadModel import LoadModel
import sys
from LiveChat import LiveChat, LiveChatUI
from contextlib import suppress
from ErrorMessage import ErrorMessage

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # Load UI
        uic.loadUi("ui/UIMain.ui", self)
        self.setFixedSize(460, 360)

        # Define Widget
        # tab1
        self.label = self.findChild(QLabel, "label")
        self.findDataset = self.findChild(QPushButton, "pushButton")
        self.addressDataset = self.findChild(QLabel, "label_2")
        self.lenDataset = self.findChild(QLabel, "label_4")
        self.labelFeature = self.findChild(QLabel, "label_10")
        self.inFeature = self.findChild(QLineEdit, "lineEdit_2")
        self.labelEmbedding = self.findChild(QLabel, "label_11")
        self.inEmbedding = self.findChild(QLineEdit, "lineEdit_3")
        self.labelLSTM = self.findChild(QLabel, "label_12")
        self.inLSTM = self.findChild(QLineEdit, "lineEdit_4")
        self.labelEpoch = self.findChild(QLabel, "label_13")
        self.inEpoch = self.findChild(QLineEdit, "lineEdit_5")
        self.labelBatch = self.findChild(QLabel, "label_14")
        self.inBatch = self.findChild(QLineEdit, "lineEdit_6")
        self.progressTraining = self.findChild(QProgressBar, "progressBar")
        self.makeModel = self.findChild(QPushButton, "pushButton_2")
        self.tab1 = self.findChild(QWidget, "tab_1")

        # tab2
        self.loadModel = self.findChild(QPushButton, "pushButton_3")
        self.model2 = self.findChild(QLabel, "label_3")
        self.testModel = self.findChild(QPushButton, "pushButton_8")
        self.getAccuracy = self.findChild(QLabel, "label_5")
        self.getConfusionMatrix = self.findChild(QPushButton, "pushButton_6")
        self.getLogConfusionMatrix = self.findChild(QPushButton, "pushButton_4")
        self.tab2 = self.findChild(QWidget, "tab_2")

        #tab3
        self.getLinkLiveChat = self.findChild(QLineEdit, "lineEdit")
        self.LiveChat = self.findChild(QPushButton, "pushButton_9")
        self.tab3 = self.findChild(QWidget, "tab_3")

        # Do Something
        # tab1
        self.findDataset.clicked.connect(self.clicker_data)
        self.makeModel.clicked.connect(self.clicker_make)
        # tab2
        self.loadModel.clicked.connect(self.clicker_model)
        self.testModel.clicked.connect(self.clicker_test)
        self.getConfusionMatrix.clicked.connect(self.clicker_table)
        self.getLogConfusionMatrix.clicked.connect(self.clicker_log)
        # tab3
        self.LiveChat.clicked.connect(self.clicker_livechat)

        # init
        self.dataset_model_path = ""



        # Show App
        self.show()

    # live chat play
    def clicker_livechat(self):
        video_url = self.getLinkLiveChat.text()
        if re.search(r"youtube.com/watch\?v=", video_url):
            self.livechatUI = LiveChatUI()
            self.livechatUI.show()
            video_id = re.findall(r'watch\?v=.*[\?]?', video_url)
            video_id = re.sub(r'watch\?v=', '', video_id[0])
            video_id = re.sub(r'\?.*', '', video_id)
            LiveChat(video_id, self.livechatUI, self.load_model.model, self.timestr)
        else:
            self.err = ErrorMessage()
            self.err.errormess("Format Data Tidak Sesuai! Pastikan : \n-Link Valid (youtube.com/watch?v=)")

    # get log matrix
    def clicker_log(self):
        self.load_model.log_matrix()

    # get table
    def clicker_table(self):
        self.load_model.evaluate()

    # test
    def clicker_test(self):
        with suppress(Exception):
            pname = QFileDialog.getOpenFileName(self, "Open Dataset", "Dataset", "CSV File (*.csv)")
            self.load_model.input_test(pname[0])
            self.load_model.test_model()
            self.getAccuracy.setEnabled(True)
            self.getConfusionMatrix.setEnabled(True)
            self.getLogConfusionMatrix.setEnabled(True)
            self.getAccuracy.setText("Akurasi : " + str(self.load_model.get_accuracy()))

    # get model
    def clicker_model(self):
        try:
            pname = QFileDialog.getExistingDirectory(self, "Open Model", "model-data/model")
            self.load_model = LoadModel(pname)
            match = re.search(r"-((\d+)-(\d+))", pname)
            self.timestr = match.group(1)
            self.model2.setText("Model : " + os.path.basename(pname))
            self.model2.setEnabled(True)
            self.testModel.setEnabled(True)
            self.tab3.setEnabled(True)
            self.getConfusionMatrix.setEnabled(False)
            self.getLogConfusionMatrix.setEnabled(False)
            self.getAccuracy.setText("Akurasi :")

        except:
            if pname != '':
                self.err = ErrorMessage()
                self.err.errormess("Format Data Tidak Sesuai! Pastikan : \n-Folder Model Dibuat Dari Aplikasi Ini")

    # get dataset
    def clicker_data(self):
        pname = ""
        try:
            pname = QFileDialog.getOpenFileName(self, "Open Dataset", "Dataset", "CSV File (*.csv)")
            data = InputData(pname[0])
            self.dataset_model_path = pname

            # Get Name
            if pname:
                self.addressDataset.setText("Nama Dataset : " + os.path.basename(pname[0]))
                self.inFeature.setText(data.get_word_length())
                self.lenDataset.setText("Jumlah Dataset : " + data.get_data_length())
                self.inEmbedding.setText("100")
                self.inLSTM.setText("64")
                self.inEpoch.setText("100")
                self.inBatch.setText("64")
                self.enable_tab1(True)
                self.progressTraining.setValue(0)
        except:
            if pname[0] != '':
                self.err = ErrorMessage()
                self.err.errormess("Format Data Tidak Sesuai! Pastikan : \n-Terdapat kolom text dan label")

    # is format for training is legal
    def is_legal(self):
        with suppress(Exception):
            if int(self.inFeature.text()) > 0:
                if int(self.inEmbedding.text()) > 0:
                    if int(self.inLSTM.text()) > 0:
                        if int(self.inEpoch.text()) > 0:
                            return True

    # make model
    def clicker_make(self):
        if self.is_legal():
            self.enable_tab1(False)
            self.tab2.setEnabled(False)
            self.tab3.setEnabled(False)
            self.progressTraining.setValue(0)

            path = self.dataset_model_path[0]
            num_feature = int(self.inFeature.text())
            num_embedding = int(self.inEmbedding.text())
            num_lstm = int(self.inLSTM.text())
            num_epoch = int(self.inEpoch.text())
            num_batch = int(self.inBatch.text())
            self.timestr = time.strftime("%Y%m%d-%H%M%S")

            runnable = MakeModel(path, num_feature, num_embedding, num_lstm, num_epoch, num_batch, UIWindow, self.timestr)
        else:
            self.err = ErrorMessage()
            self.err.errormess("Format Salah! Pastikan Data Inputan Adalah Angka Lebih Dari 0!")

    # enable the disable button tab 1
    def enable_tab1(self, is_enabled):
        self.label.setEnabled(is_enabled)
        self.findDataset.setEnabled(is_enabled)
        self.lenDataset.setEnabled(is_enabled)
        self.addressDataset.setEnabled(is_enabled)
        self.labelFeature.setEnabled(is_enabled)
        self.inFeature.setEnabled(is_enabled)
        self.labelEmbedding.setEnabled(is_enabled)
        self.inEmbedding.setEnabled(is_enabled)
        self.labelLSTM.setEnabled(is_enabled)
        self.inLSTM.setEnabled(is_enabled)
        self.makeModel.setEnabled(is_enabled)
        self.labelEpoch.setEnabled(is_enabled)
        self.inEpoch.setEnabled(is_enabled)
        self.labelBatch.setEnabled(is_enabled)
        self.inBatch.setEnabled(is_enabled)
        self.progressTraining.setEnabled(not is_enabled)

    # close app
    def closeEvent(self, event):
        event.accept()
        sys.exit()

# Initial App
app = QApplication(sys.argv) # your code to init QtCore
UIWindow = UI()
app.exec_()
