from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


class ToxicRandomForest:

    def __init__(self):
        self.output_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.train = None
        self.validation = None
        self.test = None
        self.train_features = None
        self.test_features = None
        self.models = {}

    def load_data(self, data_path):
        # Load features data
        self.train_features = self.get_features(data_path + 'train-features.csv')
        self.test_features = self.get_features(data_path + 'test-features.csv')
        # Get ids of train and validation comments
        usecols = ['id'] + self.output_classes
        self.train = pd.read_csv(
            data_path+'train-train.csv', usecols=usecols, index_col='id'
        )
        self.validation = pd.read_csv(
            data_path + 'train-val.csv', usecols=usecols, index_col='id'
        )

        print('Features: %s' % list(self.train_features.columns))

    def get_features(self, fpath):
        # Read features from file
        features = pd.read_csv(fpath, index_col='id')
        # Build extra features
        features['caps_pct'] = features['n_upper'] / (features['n_lower'] + features['n_upper'])
        features['caps_pct'].fillna(0.0, inplace=True)
        features['unique_pct'] = features['n_unique'] / features['n_words']
        features['unique_pct'].fillna(0.0, inplace=True)
        features['n_seps'] = features['n_points'] + features['n_comma']
        features['seps_pct'] = features['n_seps'] / features['n_words']
        features['seps_pct'].fillna(0.0, inplace=True)

        to_drop = ['has_ip', 'n_smilies', 'n_symbols', 'n_stars', 'n_interrogation', 'n_seps']
        features.drop(to_drop, axis=1, inplace=True)

        return features

    def fit(self):
        x_fit = self.train_features.loc[self.train.index].values
        self.models = {}
        for c in self.output_classes:
            print('Fitting \'%s\'...' % c)
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=None, n_jobs=6, oob_score=False,
                max_features='auto', min_samples_leaf=100
            )
            # clf = LogisticRegression(C=100, max_iter=200, tol=1e-5)
            # clf = KNeighborsClassifier(n_neighbors=300, n_jobs=6)
            self.models[c] = clf.fit(x_fit, y=self.train[c].values)
            if hasattr(self.models[c], 'feature_importances_'):
                print(self.models[c].feature_importances_)

    @property
    def validation_features(self):
        return self.train_features.loc[self.validation.index]

    def train_predictions(self):
        return self.predictions_df(self.train_features.loc[self.train.index])

    def validation_predictions(self):
        return self.predictions_df(self.validation_features)

    def test_predictions(self):
        # return self.predictions_df(self.test_features)
        return self.predictions_df(self.test_features.fillna(0.0))

    def predictions_df(self, features):
        """
        Generates predictions DataFrame
        :param features: pandas.DataFrame
            Comments to predict, with respective features
        :return: pandas.DataFrame
            Predicted probabilities for each one of the output classes
        """
        preds = {}
        for c in self.output_classes:
            preds[c] = [i[1] for i in self.models[c].predict_proba(features)]

        df = pd.DataFrame(preds, index=features.index, columns=self.output_classes)
        df.index.name = 'id'

        return df
