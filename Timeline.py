import matplotlib.pyplot as plt
import pandas as pd
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel


class Timeline:
    def __init__(self, video_id):
        self.video_id = video_id

        # load history live chat
        df = pd.read_csv(f"livechat-history/livechat-{self.video_id}.csv")

        # emot
        emot = df["emotion"]

        plot = emot.value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(10, 8),
                                                textprops={'fontsize': 20}, ylabel='')
        fig = plot.get_figure()
        fig.savefig(f"timeline/timeline-{video_id}")
        # ax.clear()
        plt.close()


class ShowTimeline(QMainWindow):
    def __init__(self):
        super(ShowTimeline, self).__init__()
        # Load UI
        self.setFixedSize(650, 500)
        uic.loadUi("ui/Timeline.ui", self)

        self.timestr = ""
        self.pname = ""

        # Define Widget
        self.label = self.findChild(QLabel, "label")

        # Do Something

    def Timeline_clicked(self, video_id):
        self.video_id = video_id
        self.pname = f'timeline/timeline-{self.video_id}.png'
        self.pixmap = QPixmap(self.pname)
        self.pixmap_scaled = self.pixmap.scaled(self.label.size(), transformMode=QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(self.pixmap_scaled)
