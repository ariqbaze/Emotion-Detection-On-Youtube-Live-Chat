import gc
import re
from threading import Thread
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import time

# path
class MakeModel():
    def __init__(self, path, num_feature, num_embedding, num_lstm, num_epoch, num_batch, UIWindow, timestr):
        super(MakeModel, self).__init__()
        self.value = 0
        self.path = path
        self.num_feature = num_feature
        self.num_embedding = num_embedding
        self.num_lstm = num_lstm
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.UIWindow = UIWindow
        self.timestr = timestr
        t1 = Thread(target=self.run)
        t1.start()

    def run(self):
        tic = time.perf_counter()
        # train
        emotion_train = pd.read_csv(self.path)

        # gettext/label
        text_training = emotion_train['text']
        label_training = emotion_train['label']

        # preprocessing
        text_training = text_training.apply(lambda x: x.lower())
        text_training = text_training.apply(lambda x: re.sub(r'@[^\s]+', ' [username] ', x))
        text_training = text_training.apply(lambda x: re.sub('[^a-zA-z0-9\s\?]', ' ', x))
        text_training = text_training.apply(lambda x: re.sub(r'[^\s]*wk[^\s]*', " [ketawa] ", x))
        text_training = text_training.apply(lambda x: re.sub(r'[^\s]*kwk[^\s]*', " [ketawa] ", x))
        text_training = text_training.apply(lambda x: re.sub(r'[^\s]*lmao[^\s]*', " [ketawa] ", x))
        text_training = text_training.apply(lambda x: re.sub(r'[^\s]*xi[^\s]*', " [ketawa] ", x))
        text_training = text_training.apply(lambda x: re.sub(r'[^\s]*haha[^\s]*', " [ketawa] ", x))
        text_training = text_training.apply(lambda x: re.sub(r'[^\s]*hh[^\s]*', " [ketawa] ", x))
        text_training = text_training.apply(lambda x: re.sub(r'\?', ' ? ', x))
        text_training = text_training.apply(lambda x: re.sub(r'(.)\1+\b', r'\1', x))


        kamus = pd.read_csv("kamus.csv")
        informal = kamus['informal'].astype(str).values.tolist()
        formal = kamus['formal'].astype(str).values.tolist()
        for i in range(len(formal)):
            text_training = text_training.apply(lambda x: re.sub(fr"\b{informal[i]}\b", formal[i], x))

        # Tokenizer
        tokenizer = Tokenizer(num_words=self.num_feature, oov_token='<UNK>', filters='')
        tokenizer.fit_on_texts(text_training)


        # Saving token
        with open(f'model-data/token/tokenizer-{self.timestr}.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Padding
        def get_sequences(token, tweets, max_train=100):
            sequences = token.texts_to_sequences(tweets)
            padded_sequences = pad_sequences(sequences, maxlen=max_train)
            return padded_sequences

        padded_train_sequences = get_sequences(tokenizer, text_training)

        # Label to Int
        classes = set(label_training)
        classes_to_index = dict((c, i) for i, c in enumerate(classes))
        with open(f'model-data/classes/classes-{self.timestr}.pickle', 'wb') as handle:
            pickle.dump(classes_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        names_to_ids_training = lambda labels: np.array([classes_to_index.get(x) for x in label_training])
        train_labels = names_to_ids_training(label_training)

        # model initial
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.num_feature, self.num_embedding, input_length=100),
            tf.keras.layers.LSTM(self.num_lstm),
            tf.keras.layers.Dense(len(classes), activation="softmax")
        ])
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # model fit
        callback_progress_bar = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda self, epoch: update_progress_bar(),
            on_train_batch_end=lambda self, batch: gc.collect()
        )

        self.value = 0

        def update_progress_bar():
            gc.collect()
            # k.clear_session()
            self.value += 1
            update = self.value / self.num_epoch * 99
            self.UIWindow.progressTraining.setValue(update)

        h = model.fit(
            padded_train_sequences, train_labels,
            epochs=self.num_epoch,
            batch_size=self.num_batch,
            callbacks=callback_progress_bar,
            verbose=0
        )
        gc.collect()
        toc = time.perf_counter()
        time_estimate = toc - tic
        with open(f'model-data/time/time-{self.timestr}.txt', 'w') as f:
            f.write(f"time estimate = {time_estimate}")

        # Save model
        model.save(f"model-data/model/model-{self.timestr}")
        gc.collect()

        # Save Data Training History
        hist_df = pd.DataFrame(h.history)
        hist_csv_file = f'model-data/history/history-{self.timestr}.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        gc.collect()
        tf.keras.backend.clear_session()
        gc.collect()

        self.UIWindow.progressTraining.setValue(100)
        self.UIWindow.enable_tab1(True)
        self.UIWindow.tab2.setEnabled(True)
        self.UIWindow.tab3.setEnabled(True)
