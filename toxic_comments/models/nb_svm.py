from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.pipeline import Pipeline
import load_data
import click
import pandas as pd
import multiprocessing
import nltk
import string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

WORDS_MAPPING_FILE_PATH = '/home/rafasa/code/kaggle/toxic_comments/data/external/conv.csv'


def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


class NaiveBayesSVM:

    def __init__(self):
        self.output_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.n_processes = 1
        self.models = {}
        self.parser_args = {
            'lower': False,
            'words_map': True,
            'lemmatize': False,
            'rmv_stops': False
        }
        self.train = None
        self.validation = None
        self.test = None

    def load_data(self, data_path):
        self.train, self.validation, self.test = load_data.load_train_val_test(
            data_path, clean=False
        )

    def fit(self):
        print('Parsing comments...')
        comments = self.parse_comments(self.train['comment_text'].values)
        # comments = train['comment_text'].values
        # Run in a pool of processes
        pool = multiprocessing.Pool(processes=self.n_processes)
        manager = multiprocessing.Manager()
        self.models = manager.dict()
        for c in self.output_classes:
            pool.apply_async(
                self._fit,
                args=(comments, self.train[c].values, self.models, c)
            )
        pool.close()
        pool.join()
        self.models = dict(self.models)

    def parse_comments(self, comments):
        assert type(comments) == np.ndarray
        p = TextParser(**self.parser_args)
        p.initialize()
        return [p.parse(c) for c in comments]

    @staticmethod
    def _fit(x, y, res_dict, cat):
        print('Fitting \'%s\'...' % cat)
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), tokenizer=tokenize, min_df=3, max_df=0.9,
            strip_accents='unicode', use_idf=1, smooth_idf=1, sublinear_tf=1
        )
        nbf = NBFeaturer(alpha=10)
        model = LogisticRegression(C=4, dual=True)

        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('nbf', nbf),
            ('lr', model)
        ])

        res_dict[cat] = pipeline.fit(X=x, y=y)
        print('Done')

    def validation_predictions(self):
        return self.predictions_df(self.validation['comment_text'])

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
        preds = {}
        parsed_comms = self.parse_comments(comments.values)
        for c in self.output_classes:
            preds[c] = [
                i[1] for i in self.models[c].predict_proba(parsed_comms)
            ]

        df = pd.DataFrame(preds, index=comments.index, columns=self.output_classes)
        df.index.name = 'id'

        return df


class TextParser:

    def __init__(self, lower=False, words_map=False, lemmatize=False, rmv_stops=False):
        self.lower = lower
        self.words_map = words_map
        self.lemmatize = lemmatize
        self.rmv_stops = rmv_stops
        self.lem = None
        self.tokenizer = None
        self.stopwords = None
        self.words_mapping = None

    def initialize(self):
        self.tokenizer = nltk.tokenize.TweetTokenizer()
        if self.lemmatize:
            self.lem = nltk.stem.wordnet.WordNetLemmatizer()
        if self.rmv_stops:
            self.stopwords = set(nltk.corpus.stopwords.words("english"))
        if self.words_map:
            self.words_mapping = get_words_mapping()

    def parse(self, text):
        # remove \n
        text = re.sub("\\n", " ", text)
        # remove leaky elements like ip,user
        text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", text)
        # removing usernames
        text = re.sub("\[\[.*\]", "", text)
        # to lower
        if self.lower:
            text = text.lower()

        # Split the sentences into words
        words = self.tokenizer.tokenize(text)
        # Replace some words
        if self.words_map:
            words = [
                self.words_mapping[word]
                if word in self.words_mapping else word
                for word in words
            ]
        if self.lemmatize:
            words = [self.lem.lemmatize(word, "v") for word in words]
        if self.rmv_stops:
            words = [w for w in words if w not in self.stopwords]

        text = " ".join(words)

        return text


class NBFeaturer(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha):
        self.alpha = alpha

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb

    def fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)


def get_words_mapping():
    df = pd.read_csv(WORDS_MAPPING_FILE_PATH)
    return {k: v for k, v in df.values}


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
def main(data_path):
    train, val, test = load_data.load_train_val_test(data_path, clean=False)


if __name__ == '__main__':
    main()
