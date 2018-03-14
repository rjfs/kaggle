"""
Based on Kaggle's user Bongo kernel:
    [For Beginners] Go Even Deeper with Char-Gram+CNN
    https://www.kaggle.com/sbongo/for-beginners-go-even-deeper-with-char-gram-cnn
"""
import pandas as pd
import matplotlib.pyplot as plt
import click
import time
import re

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D, Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import roc_auc_score
from keras.models import load_model


class CharGramCNN:

    def __init__(self, data_path):
        self.data_path = data_path
        self.n_filters = 30  # 100
        self.sentences_maxlen = 300  # 500
        self.batch_size = 128  # 32
        self.epochs = 2
        self.tokenizer = None
        self.model = None
        self.output_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.stopwords = stopwords.words('english')
        self.porter = PorterStemmer()
        self.tweet_tokenizer = TweetTokenizer()
        self.max_str_len = 30

    def run(self):
        # Load data
        train = pd.read_csv(self.data_path + 'train_train.csv', index_col='id')
        val = pd.read_csv(self.data_path + 'train_validation.csv', index_col='id')
        test = pd.read_csv(self.data_path + 'test.csv', index_col='id')
        # Initialize Keras tokenizer
        train_comments = train['comment_text']
        self.initialize_tokenizer(train_comments.values)
        # Parse data
        print('Parsing data')
        x_train = self.parse_data(train_comments, clean=True)
        x_val = self.parse_data(val['comment_text'], clean=True)
        x_test = self.parse_data(test['comment_text'], clean=True)
        # Build model
        self.build_nn()
        # Run model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        y_train = train[self.output_classes].values
        y_val = val[self.output_classes].values
        print('Fitting model')
        hist = self.model.fit(
            x_train, y_train,
            batch_size=self.batch_size, epochs=self.epochs,
            validation_data=(x_val, y_val)
        )
        print(hist)
        # Print ROC AUC
        y_pred = self.model.predict(x_val)
        print(roc_auc_score(y_val, y_pred))
        # Generate predictions
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.generate_predictions(x_val, ids=list(val.index), fname=timestr+'-val')
        self.generate_predictions(x_test, ids=list(test.index), fname=timestr+'-test')

    def generate_predictions(self, x, ids, fname):
        preds = self.model.predict(x)
        df = pd.DataFrame(preds, index=ids, columns=self.output_classes)
        df.index.name = 'id'
        df.to_csv(fname + '.out')

    def initialize_tokenizer(self, comments):
        max_features = 20000
        self.tokenizer = Tokenizer(num_words=max_features, char_level=True)
        # Update internal vocabulary based on comments
        self.tokenizer.fit_on_texts(comments)

    def build_nn(self):
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

    def parse_data(self, comments, plot=False, clean=False, lower=False):
        if lower:
            comments = comments.apply(lambda c: c.lower())
        if clean:
            print('Cleaning data')
            comments = comments.apply(lambda c: self.clean(c))
        # Convert texts to sequences
        tokenized = self.tokenizer.texts_to_sequences(comments)
        # Plot number of words histogram
        if plot:
            plot_words_hist(tokenized)
        # Since there are sentences with varying length of characters, we have to get them on a constant size
        x = pad_sequences(tokenized, maxlen=self.sentences_maxlen)

        return x

    def clean(self, comment):
        comment = re.sub('[\\n]+', ' ', comment)
        comment = re.sub('[\W]+', ' ', comment)

        words = self.tweet_tokenizer.tokenize(comment)
        words = [self.porter.stem(word[:self.max_str_len]) for word in words]
        words = [w for w in words if w not in self.stopwords]

        return " ".join(words)


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
def main(data_path):
    mod = CharGramCNN(data_path)
    mod.run()


def plot_words_hist(tokenized_text):
    n_words = [len(c) for c in tokenized_text]
    plt.hist(n_words)


if __name__ == '__main__':
    main()
