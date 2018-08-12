import core
import logging
import sys
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import LinearSVR
from core import MONTH_INT_LABEL, TARGET_LABEL, INDEX_COLUMNS
import evaluate
import utils
sys.path.append(utils.up_dir(utils.get_script_dir(), 1))
from features import build_features
import numpy as np
import tuning


class LinearModel(core.Model):

    def __init__(self, model, drop_first_ohe, standardize, n_evals=2,
                 sample=None):
        super().__init__(
            model, sample=sample, standardize=standardize
            # , train_clip=(0., 40.)
        )
        self.drop_first_ohe = drop_first_ohe
        self.features = None
        self.n_evals = n_evals

    def run(self):
        self.initialize()
        # Get predictions file
        self.get_predictions_file()

    def initialize(self):
        logging.info('loading features')
        self.load_features()
        # Compute features with all available data
        logging.info('computing new features')
        self.parse_features()

    def parse_features(self):
        """
        Features are parsed in the following way:
            - Add OHE month feature
            - Add total sales of last n months, for each pair
            - Encode shop id and item id
            - Compute (last_px - min_px_item) / (max_px_item - min_px_item)
        """
        feats = build_features.Features(
            target_col=TARGET_LABEL, month_col=MONTH_INT_LABEL
        )
        # Add month one hot encoding
        # self.features = feats.one_hot_encode(
        #     self.features, 'month', drop_first=self.drop_first_ohe
        # )
        # self.features.drop('month', axis=1, inplace=True)
        # self.features['month_12'] = self.features.month == 12
        # Add month number (may help with seasonal effects)
        # feats.data['month_n'] = feats.data[MONTH_INT_LABEL].values
        # Fill some missing values
        self.features = feats.fill_lags(self.features, 0)
        self.features = feats.fill_counts(self.features, 0)
        # Add shop and item totals
        totals_lags = {
            'shop_id': [12],
            'item_id': [1, 2],
            # 'item_category_id': [1, 12]
        }
        for c, lags in totals_lags.items():
            self.features = feats.add_totals(
                self.features, c, lags=lags, dtype=np.int16
            )

        # Drop item categories features
        to_drop = ['catname1', 'catname2', 'item_category_id']
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')
        # Parse prices features
        self.features = feats.add_price_features(self.features, fill_value=None)
        c = 'item_price_last_lag1'
        self.features[c] = self.features[c].fillna(self.features[c].mean())
        # Remove some initial rows
        mask = self.features[MONTH_INT_LABEL] >= 12
        self.features = self.features[mask]
        # Add features interactions
        sum12 = self.features['lag1'] + self.features['lag2']
        # self.features['lag1+2'] = sum12
        self.features['lag1+2+3'] = sum12 + self.features['lag3']
        self.features['trend_it'] = (
            self.features['item_id_lag1'] - self.features['item_id_lag2']
        )

        to_drop = [
            'lag2', 'lag3',
            'item_cnt_day_min',
            # 'item_price_last_lag1',
            # 'px_min',
            # 'px_max',
            # 'px_range',
            # 'px_diff_min',
            # 'px_scale',
            # 'px_diff_max',
            # 'n_weekend',
            'month_range',
            'month',
            'n_weekend',
            'item_id_lag2',
            'lag4'
        ]
        self.features.drop(to_drop, axis=1, errors='ignore', inplace=True)
        # self.features = self.features.dropna()
        for c in self.features:
            if c not in INDEX_COLUMNS + [TARGET_LABEL]:
                # TODO: Find better strategy to fill missing values
                self.features[c] = self.features[c].fillna(0.0)

    def get_validation_score(self):
        self.initialize()
        # Fill missing values
        logging.info('running evaluation')
        me = evaluate.ModelEvaluation(model_class=self, n_evals=self.n_evals)
        me.evaluate(self.train_feats)

    def get_months_predictions(self, month_i, month_f):
        self.initialize()
        self.predict_months(month_i, month_f, fname='lasso')


class LinearRegressionModel(LinearModel):

    def __init__(self, sample=None):
        super().__init__(
            LinearRegression(),
            drop_first_ohe=True,
            standardize=False,
            sample=sample
        )


class LinearSVM(LinearModel):

    def __init__(self, sample=None):
        super().__init__(
            LinearSVR(**self.tuned_params),
            drop_first_ohe=False,
            standardize=False,
            n_evals=1,
            sample=sample
        )

    @property
    def tuned_params(self):
        return {
            'C': 0.3, 'loss': 'squared_epsilon_insensitive', 'epsilon': 0.1,
            'intercept_scaling': 1.0
        }

    @property
    def params_lists(self):
        return {
            'C': [0.1, 0.3, 0.5, 1.0, 2.0],
            'loss': ['squared_epsilon_insensitive'],
            'epsilon': [0, 0.1, 0.2],
            # 'dual': [False],
            'intercept_scaling': [0.1, 0.3, 1.0]
        }

    def parameters_search(self):
        self.get_features()
        prs = tuning.ParamsRandomSearch(
            model_class=LinearSVR,
            params_lists=self.params_lists,
            eval_funct=self.get_model_score
        )
        prs.run()

    def get_model_score(self, model):
        # Update model
        self.model = model
        me = evaluate.ModelEvaluation(
            model_class=self, n_evals=self.n_evals
        )
        return me.evaluate(self.train_feats)


class RidgeRegression(LinearModel):

    def __init__(self, sample=None):
        super().__init__(
            Ridge(alpha=1.0, normalize=True),
            drop_first_ohe=False,
            standardize=False,
            n_evals=2,
            sample=sample  # 2**20 ~ 1M
        )

    @property
    def params_lists(self):
        return {
            'alpha': [0.1, 0.3, 0.5, 1.0, 2.0, 3.0],
            'normalize': [False, True]
        }

    def parameters_search(self):
        self.get_features()
        prs = tuning.ParamsRandomSearch(
            model_class=Ridge,
            params_lists=self.params_lists,
            eval_funct=self.get_model_score
        )
        prs.run()

    def get_model_score(self, model):
        # Update model
        self.model = model
        me = evaluate.ModelEvaluation(
            model_class=self, n_evals=self.n_evals
        )
        return me.evaluate(self.train_feats)


class LassoRegression(LinearModel):

    def __init__(self, sample=None):
        # params = {'alpha': 0.5, 'normalize': False}
        params = {'alpha': 2.0, 'fit_intercept': False, 'normalize': False}
        super().__init__(
            Lasso(**params),
            drop_first_ohe=False,
            standardize=False,
            n_evals=2,
            sample=sample
        )

    @property
    def params_lists(self):
        return {
            'alpha': [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 20.0],
            'fit_intercept': [False, True],
            'normalize': [False]
        }

    def parameters_search(self):
        self.get_features()
        prs = tuning.ParamsRandomSearch(
            model_class=Lasso,
            params_lists=self.params_lists,
            eval_funct=self.get_model_score
        )
        prs.run()

    def get_model_score(self, model):
        # Update model
        self.model = model
        me = evaluate.ModelEvaluation(
            model_class=self, n_evals=self.n_evals
        )
        return me.evaluate(self.train_feats)

