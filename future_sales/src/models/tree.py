import logging
from xgboost import XGBRegressor
import core
import evaluate
import tuning
from features import build_features
from core import MONTH_INT_LABEL, TARGET_LABEL


class TreeBased(core.Model):

    def __init__(self, model, n_evals=3, sample=None):
        super().__init__(
            model, sample=sample, standardize=False, train_clip=(0., 40.),
            model_name='gbdt'
        )
        self.features = None
        self.n_evals = n_evals

    @property
    def params_lists(self):
        """
        Used for XGBRegressor best parameters search

        min_child_weight:
        In linear regression mode, this simply corresponds to minimum number of
        instances needed to be in each node.
        The larger, the more conservative the algorithm will be.

        """
        return {
            'booster': ['gbtree'],
            'min_child_weight': [5, 10, 30],
            'colsample_bytree': [0.4, 0.5, 0.6],
            'max_depth': [8, 10, 12, 15, 20],
            'learning_rate': [0.02, 0.03, 0.1, 0.3],
            'n_estimators': [50, 75, 100, 125],
            'lambda': [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
            'subsample': [0.9, 1.0]
        }

    def parameters_search(self):
        self.initialize()
        prs = tuning.ParamsRandomSearch(
            model_class=XGBRegressor,
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

    def initialize(self):
        logging.info('loading features')
        self.load_features()
        # Compute features with all available data
        logging.info('computing new features')
        self.parse_features()

    def get_months_predictions(self, month_i, month_f):
        self.initialize()
        self.predict_months(month_i, month_f, fname='xgb')

    def run(self):
        self.initialize()
        # Get predictions file
        self.get_predictions_file()

    def parse_features(self):

        feats = build_features.Features(
            target_col=TARGET_LABEL,
            month_col=MONTH_INT_LABEL
        )
        # Fill some missing values
        self.features = feats.fill_lags(self.features, 0)
        self.features = feats.fill_counts(self.features, 0)
        # Add shop and item totals
        totals_lags = {
            'item_id': [1, 2],
            'shop_id': [12, 24],
            'item_category_id': [12, 24]
            # 'catname1', 'catname2'
        }
        for c, lags in totals_lags.items():
            print('Adding %s total' % c)
            self.features = feats.add_totals(self.features, c, lags=lags)

        # Add totals interactions
        # self.features['lag_sums'] = (
        #     self.features['lag1'] +
        #     self.features['lag2'] +
        #     self.features['lag3']
        # )
        # self.features['trend'] = self.features['lag1'] - self.features['lag2']
        self.features['trend_it'] = (
            self.features['item_id_lag1'] - self.features['item_id_lag2']
        )
        # Parse prices features
        self.features = feats.add_price_features(self.features, fill_value=-1.)
        c = 'item_price_last_lag1'
        self.features[c] = self.features[c].fillna(-1.0)
        # Remove some initial rows
        self.features = self.features[self.features[MONTH_INT_LABEL] >= 24]
        to_drop = [
            # 'item_category_id',
            'item_cnt_day_min',
            # 'item_price_last_lag1',
            # 'lag12',
            'month',
            # 'px_min',
            # 'px_max',
            # 'px_range',
            # 'px_diff_min',
            'px_diff_max',
            'px_scale',
            'month_range',
            # 'n_weekend',
            'catname1',
            'catname2',
            'lag4',
            'lag5',
            # 'item_id_lag1', 'item_id_lag2'
        ]
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')

    def get_validation_score(self):
        self.initialize()
        # Fill missing values
        logging.info('running evaluation')
        me = evaluate.ModelEvaluation(
            model_class=self, n_evals=self.n_evals
        )
        me.evaluate(self.train_feats)


class TunedGBDT(TreeBased):

    def __init__(self):
        super().__init__(
            XGBRegressor(**self.model_params),
            n_evals=1,
            sample=None
        )

    @property
    def model_params(self):
        """
        return {
            'booster': 'gbtree', 'min_child_weight': 5, 'colsample_bytree': 0.4,
            'max_depth': 10, 'learning_rate': 0.05, 'n_estimators': 40,
            'lambda': 1.0, 'subsample': 1.0
        }
        """

        return {
            'booster': 'gbtree', 'min_child_weight': 5, 'colsample_bytree': 0.5,
            'max_depth': 12, 'learning_rate': 0.03, 'n_estimators': 62,
            'lambda': 1.0, 'subsample': 0.9
        }

        # {
        #     'booster': 'gbtree', 'min_child_weight': 10,
        #     'colsample_bytree': 0.5, 'max_depth': 8, 'learning_rate': 0.1,
        #     'n_estimators': 25, 'lambda': 10.0, 'subsample': 1.0
        # }
