from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import string
from sklearn.pipeline import Pipeline
import load_data
import click
import pandas as pd
import multiprocessing

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


class NaiveBayesSVM:

    def __init__(self):
        self.output_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.n_processes = 3
        self.models = {}

    def fit(self, train, validation):
        # Run in a pool of processes
        pool = multiprocessing.Pool(processes=self.n_processes)
        manager = multiprocessing.Manager()
        self.models = manager.dict()
        for c in self.output_classes:
            pool.apply_async(
                self._fit,
                args=(train['comment_text'].values, train[c].values, self.models, c)
            )
        pool.close()
        pool.join()
        self.models = dict(self.models)

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

    def predictions_df(self, comments):
        """
        Generates predictions DataFrame
        :param comments: pandas.Series
            Comments series, with its IDs as index
        :return: pandas.DataFrame
            Predicted probabilities for each one of the output classes
        """
        preds = {}
        for c in self.output_classes:
            preds[c] = [
                i[1] for i in self.models[c].predict_proba(comments.values)
            ]

        df = pd.DataFrame(preds, index=comments.index, columns=self.output_classes)
        df.index.name = 'id'

        return df


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


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
def main(data_path):
    train, val, test = load_data.load_train_val_test(data_path, clean=False)


if __name__ == '__main__':
    main()
