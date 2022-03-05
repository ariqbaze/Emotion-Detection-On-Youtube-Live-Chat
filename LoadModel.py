import pandas as pd
import pickle
import re
import tensorflow as tf
import numpy as np
import seaborn as sns
from ErrorMessage import ErrorMessage
from matplotlib import pyplot as plt
from Matrix import ShowMatrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import confusion_matrix


class LoadModel:
    def __init__(self, pname):
        self.pname = pname
        self.model = tf.keras.models.load_model(self.pname)
        self.match = re.search(r"-((\d+)-(\d+))", self.pname)
        self.timestr = self.match.group(1)
        self.path = ""

    def input_test(self, path):
        self.path = path

    def test_model(self):
        # test

        self.emotion_test = pd.read_csv(self.path)

        # gettext/label
        try:
            text_test = self.emotion_test['text']
            label_test = self.emotion_test['label']
        except:
            self.err = ErrorMessage()
            self.err.errormess("Format Data Tidak Sesuai! Pastikan : \n-Terdapat kolom text dan label")

        # lower case, get username to [user], delete special character and delete multiply last character text = re.sub(r' .*wk.* '," wkwkwk ",text)
        text_test = text_test.apply(lambda x: x.lower())
        text_test = text_test.apply(lambda x: re.sub(r'@[^\s]+', ' [username] ', x))
        text_test = text_test.apply(lambda x: re.sub('[^a-zA-z0-9\s\?]', ' ', x))
        text_test = text_test.apply(lambda x: re.sub(r'[\?]', ' ? ', x))
        text_test = text_test.apply(lambda x: re.sub(r'(.)\1+\b', r'\1', x))
        text_test = text_test.apply(lambda x: re.sub(r'[^\s]*wk[^\s]*', " [ketawa] ", x))
        text_test = text_test.apply(lambda x: re.sub(r'[^\s]*kwk[^\s]*', " [ketawa] ", x))
        text_test = text_test.apply(lambda x: re.sub(r'[^\s]*lmao[^\s]*', " [ketawa] ", x))
        text_test = text_test.apply(lambda x: re.sub(r'[^\s]*xi[^\s]*', " [ketawa] ", x))
        text_test = text_test.apply(lambda x: re.sub(r'[^\s]*haha[^\s]*', " [ketawa] ", x))
        text_test = text_test.apply(lambda x: re.sub(r'[^\s]*hh[^\s]*', " [ketawa] ", x))


        kamus = pd.read_csv("kamus.csv")
        informal = kamus['informal'].astype(str).values.tolist()
        formal = kamus['formal'].astype(str).values.tolist()
        for i in range(len(formal)):
            text_test = text_test.apply(lambda x: re.sub(fr"\b{informal[i]}\b", formal[i], x))

        # Get Prev Token
        with open(f'model-data/token/tokenizer-{self.timestr}.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Padding
        def get_sequences(token, tweets, max_train=100):
            sequences = token.texts_to_sequences(tweets)
            padded_sequences = pad_sequences(sequences, maxlen=max_train)
            return padded_sequences

        padded_test_sequences = get_sequences(tokenizer, text_test)

        self.preds = np.argmax(self.model.predict(padded_test_sequences, verbose=0), axis=-1)

        with open(f'model-data/classes/classes-{self.timestr}.pickle', 'rb') as handle:
            classes_to_index = pickle.load(handle)

        self.index_to_classes = dict((v, k) for k, v in classes_to_index.items())
        names_to_ids_test = lambda labels: np.array([classes_to_index.get(x) for x in label_test])
        self.test_labels = names_to_ids_test(label_test)

        plt.clf()
        try:
            cf_matrix = confusion_matrix(self.test_labels, self.preds, labels=list(self.index_to_classes.keys()))
        except:
            self.err = ErrorMessage()
            self.err.errormess("Format Data Tidak Sesuai! Pastikan : \n-Pastikan label model sama dengan data uji")
        cmtx = pd.DataFrame(
            cf_matrix,
            index=[format(x) for x in classes_to_index.keys()],
            columns=[format(x) for x in classes_to_index.keys()]
        )
        self.heatmap = sns.heatmap(cmtx, annot=True)

        self.score = self.model.evaluate(padded_test_sequences, self.test_labels, verbose=0)

        fig = self.heatmap.get_figure()
        fig.savefig(f"model-data/matrix/matrix-{self.timestr}.png")
        plt.close()


    def get_accuracy(self):
        return self.score[1]*100

    def get_pname(self):
        return self.pname

    def load_model(self):
        return self.model

    def evaluate(self):
        self.ui = ShowMatrix()
        self.ui.matrix_clicked(self.timestr)
        self.ui.show()

    def log_matrix(self):
        self.log = ConfusionMatrix(self.test_labels, self.preds, str(self.index_to_classes))
