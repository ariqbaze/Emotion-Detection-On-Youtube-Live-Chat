import pandas as pd


class InputData:
    def __init__(self, path):
        self._path = path

        # train
        emotion_train = pd.read_csv(self._path)

        # data length
        tweet_training = emotion_train['text']
        self.data_length = len(tweet_training)

        # word length
        result = [i for item in tweet_training for i in item.split()]
        self.word_length = len(set(result))

    def get_address(self):
        return self._path

    def get_data_length(self):
        return str(self.data_length)

    def get_word_length(self):
        return str(self.word_length)


