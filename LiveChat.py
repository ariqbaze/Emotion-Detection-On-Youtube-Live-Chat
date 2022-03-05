import gc
import os
import pickle
import re
import time
from contextlib import suppress
import numpy as np
import pandas as pd
from ErrorMessage import ErrorMessage
from PyQt5.QtWidgets import QMainWindow, QTextEdit
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pytchat
import csv
from threading import *
from PyQt5 import QtGui, uic

from Timeline import Timeline, ShowTimeline

_FINISH = False


class LiveChat():
    def __init__(self, video_id, LiveChatUI, model, timestr):
        super(LiveChat, self).__init__()
        self.timestr = timestr
        self.model = model
        self.video_id = video_id
        self.UIIII = LiveChatUI
        self.err = ErrorMessage()
        try:
            LiveChatUI.set_video_id(video_id)
            self.chat = pytchat.create(video_id=self.video_id)
        except:
            LiveChatUI.set_video_id("")
            LiveChatUI.close()
            self.err.errormess("Format Data Tidak Sesuai! Pastikan : \n-url benar")

        with open(f'model-data/classes/classes-{self.timestr}.pickle', 'rb') as handle:
            classes_to_index = pickle.load(handle)

        self.index_to_classes = dict((v, k) for k, v in classes_to_index.items())

        with open(f'model-data/token/tokenizer-{self.timestr}.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        global _FINISH
        _FINISH = False
        self.t1 = Thread(target=self.run)
        self.t1.start()

    def run(self):
        with suppress(Exception):
            if os.path.exists(f'livechat-history/livechat-{self.video_id}.csv'):
                os.remove(f'livechat-history/livechat-{self.video_id}.csv')
            with open(f'livechat-history/livechat-{self.video_id}.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(['date', 'emotion'])
                # write multiple rows
                with suppress(Exception):
                    while self.chat.is_alive():
                        if _FINISH:
                            break
                        data = self.chat.get()
                        items = data.items
                        for c in self.chat.get().items:
                            msg = c.message.lower()
                            msg = re.sub(r'@[^\s]+', ' [username] ', msg)
                            msg = re.sub('[^a-zA-z0-9\s\?]', ' ', msg)
                            msg = re.sub(r'[\?]', ' ? ', msg)
                            msg = re.sub(r'(.)\1+\b', r'\1', msg)
                            msg = re.sub(r'[^\s]*wk[^\s]*', " [ketawa] ", msg)
                            msg = re.sub(r'[^\s]*kw[^\s]*', " [ketawa] ", msg)
                            msg = re.sub(r'[^\s]*lmao[^\s]*', " [ketawa] ", msg)
                            msg = re.sub(r'[^\s]*xi[^\s]*', " [ketawa] ", msg)
                            msg = re.sub(r'[^\s]*haha[^\s]*', " [ketawa] ", msg)
                            msg = re.sub(r'[^\s]*hh[^\s]*', " [ketawa] ", msg)
                            kamus = pd.read_csv("kamus.csv")
                            informal = kamus['informal'].astype(str).values.tolist()
                            formal = kamus['formal'].astype(str).values.tolist()
                            for i in range(len(formal)):
                                msg = re.sub(fr"\b{informal[i]}\b", formal[i], msg)
                            msg=[msg]
                            msg = self.tokenizer.texts_to_sequences(msg)
                            msg = pad_sequences(msg, maxlen=100, dtype='int32', value=0)
                            pred = self.model.predict(msg, verbose=0)[0]
                            emot = self.index_to_classes[(np.argmax(pred))]
                            self.UIIII.livechat.append(f"[{emot}]{c.datetime} [{c.author.name}]- {c.message}")
                            self.UIIII.livechat.moveCursor(QtGui.QTextCursor.End)

                            writer.writerow([c.datetime, emot])
                            f.flush()
                            gc.collect()
                    else:
                        time.sleep(5)


class LiveChatUI(QMainWindow):
    def __init__(self):
        super(LiveChatUI, self).__init__()
        self.setFixedSize(500, 500)

        self.video_id = ""
        uic.loadUi("ui/LiveChat.ui", self)

        self.livechat = self.findChild(QTextEdit, "textEdit")

    def set_video_id(self, video_id):
        self.video_id = video_id

    def closeEvent(self, event):
        global _FINISH
        _FINISH = True

        with suppress(Exception):
            Timeline(self.video_id)
            self.ui = ShowTimeline()
            self.ui.Timeline_clicked(self.video_id)
            if self.video_id != "":
                self.ui.show()
        event.accept()
