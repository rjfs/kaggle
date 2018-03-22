"""
Based on Kaggle's user Bongo kernel:
    [For Beginners] Go Even Deeper with Char-Gram+CNN
    https://www.kaggle.com/sbongo/for-beginners-go-even-deeper-with-char-gram-cnn
"""
import pandas as pd
import matplotlib.pyplot as plt

import keras_utils
import load_data

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, GRU, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPool1D, Bidirectional
from keras.models import Model
from keras.models import load_model


class CharGramCNN:

    def __init__(self, n_filters=50, epochs=5, sentences_maxlen=500):
        self.n_filters = n_filters  # 100
        self.sentences_maxlen = sentences_maxlen  # 500
        self.batch_size = 128  # 32
        self.epochs = epochs
        self.last_finished_epoch = 5
        self.tokenizer = None
        self.model = None
        self.output_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.train = None
        self.validation = None
        self.test = None
        self.initial_weights_file = None

    def load_data(self, data_path):
        self.train, self.validation, self.test = load_data.load_train_val_test(
            data_path, clean=True
        )

    def train_predictions(self):
        return self.predictions_df(self.train['comment_text'])

    def validation_predictions(self):
        return self.predictions_df(self.validation['comment_text'])

    def train_predictions(self):
        return self.predictions_df(self.train['comment_text'])

    def test_predictions(self):
        return self.predictions_df(self.test['comment_text'])

    def predictions_df(self, comments):
        """
        Generates predictions DataFrame
        :param comments: pandas.Series
            Comments series, with its IDs as index
        :return: pandas.DataFrame
            Predicted probabilities for each one of the output classes
        """
        x = self.parse_data(comments)
        preds = self.model.predict(x)
        df = pd.DataFrame(preds, index=comments.index, columns=self.output_classes)
        df.index.name = 'id'

        return df

    def fit(self):
        # Initialize Keras tokenizer
        train_comments = self.train['comment_text']
        self.initialize_tokenizer(train_comments.values)
        # Parse data
        print('Parsing data')
        x_train = self.parse_data(train_comments)
        x_val = self.parse_data(self.validation['comment_text'])
        # Initialize neural network
        self.initialize_net()
        # Train model
        y_train = self.train[self.output_classes].values
        y_val = self.validation[self.output_classes].values
        self.train_model(x_train, y_train, x_val, y_val)

    def train_model(self, x_train, y_train, x_val, y_val):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

        # File used to save model checkpoints
        model_filename = 'chargram-cnn.{0:03d}.hdf5'

        if self.last_finished_epoch > 0:
            self.model = load_model(model_filename.format(self.last_finished_epoch-1))
        elif self.initial_weights_file is not None:
            self.model = load_model(self.initial_weights_file)

        print('Fitting model')
        self.model.fit(
            x_train, y_train,
            batch_size=self.batch_size, epochs=self.epochs,
            validation_data=(x_val, y_val),
            callbacks=[
                keras_utils.ModelSaveCallback(model_filename),
                keras_utils.TqdmProgressCallback()
            ],
            verbose=1,
            initial_epoch=self.last_finished_epoch or 0
        )

    def initialize_tokenizer(self, comments):
        max_features = 20000
        self.tokenizer = Tokenizer(num_words=max_features, char_level=True)
        # Update internal vocabulary based on comments
        self.tokenizer.fit_on_texts(comments)

    def initialize_net(self):
        inp = Input(shape=(self.sentences_maxlen,))
        embed_size = 240
        x = Embedding(len(self.tokenizer.word_index) + 1, embed_size)(inp)
        x = Conv1D(filters=self.n_filters, kernel_size=4, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=4)(x)
        gru = GRU(60, return_sequences=True, name='lstm_layer', dropout=0.2, recurrent_dropout=0.2)
        x = Bidirectional(gru)(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(25, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation="sigmoid")(x)

        self.model = Model(inputs=inp, outputs=x)

    def parse_data(self, comments, plot=False):
        # Convert texts to sequences
        tokenized = self.tokenizer.texts_to_sequences(comments)
        # Plot number of words histogram
        if plot:
            plot_words_hist(tokenized)
        # Since there are sentences with varying length of characters, we have to get them on a constant size
        x = pad_sequences(tokenized, maxlen=self.sentences_maxlen)

        return x


def plot_words_hist(tokenized_text):
    n_words = [len(c) for c in tokenized_text]
    plt.hist(n_words, bins='auto')


def save_predictions(df, fname):
    n = fname + '.out'
    df.to_csv(n)
    print('Saved to: %s' % n)


if __name__ == '__main__':
    pass
