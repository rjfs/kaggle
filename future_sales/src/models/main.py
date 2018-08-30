import pandas as pd
import logging
from sklearn.neighbors import KNeighborsRegressor
import core
import linear
import tree
from core import INDEX_COLUMNS, MONTH_INT_LABEL, TARGET_LABEL
import sys
sys.path.append('../')
from features import build_features
import tuning
import argparse
import numpy as np


pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_MODEL = 'ridge'
DEFAULT_TASK = 'validate'


def main():
    parser = argparse.ArgumentParser(description='Predict Future Sales.')
    parser.add_argument(
        '--model', dest='model', default=DEFAULT_MODEL, type=str, nargs=1,
        help='model name'
    )
    parser.add_argument(
        '--task', dest='task', default=DEFAULT_TASK, type=str, nargs=1,
        help='task to be executed'
    )

    args = parser.parse_args()

    model = args.model if isinstance(args.model, str) else args.model[0]
    task = args.task if isinstance(args.task, str) else args.task[0]

    m = Models(model, task)
    m.run()


class Models:

    def __init__(self, name, task):
        """
        :param name: str
            Either 'gbdt', 'gbdt-lgb', 'knn', 'lasso' or 'previous'
            Model to be used.
            One of the following options:
                'gbdt': XGBRegressor
                'gbdt-lgb': LGBMRegressor
                'dart': LGBMRegressor using 'dart' boosting type
                'knn': KNeighborsRegressor
                'lasso': Lasso linear regression
                'previous': Previous sales model
        :param task: str
            One of the following options:
                'optimize': Search for best hyperparameters
                'validate': Perform model validation
                'predict': Generate file with predictions for test set
                'predict_months': Generate predictions for months interval
        """
        self.name = name
        self.task = task

    def run(self):
        m = self.get_model()
        if self.task == 'optimize':
            m.parameters_search()
        elif self.task == 'validate':
            m.get_validation_score()
        elif self.task == 'predict':
            m.run()
        elif self.task == 'predict_months':
            m.get_months_predictions(31, 34)
        else:
            raise Exception('Invalid task: %s' % self.task)

    def get_model(self):
        if self.name == 'gbdt':
            return tree.TunedGBDT()
        elif self.name == 'gbdt-lgb':
            return tree.LGBM('gbdt')
        elif self.name == 'dart':
            return tree.LGBM('dart')
        elif self.name == 'knn':
            return NeighborsModel(sample=2**16)
        elif self.name == 'lasso':
            return linear.LassoRegression()#sample=2**21)
        elif self.name == 'ridge':
            return linear.RidgeRegression()
        elif self.name == 'previous':
            return LastSales()
        else:
            raise Exception('Invalid name: %s' % self.name)


class NeighborsModel(core.Model):

    def __init__(self, n_evals=1, sample=None):
        super().__init__(
            KNeighborsRegressor(), sample=sample, standardize=False,
            training_range=(0., 40.), n_eval_months=3
        )
        self.features = None
        self.n_evals = n_evals
        self.m0 = 29

    def get_months_predictions(self, month_i, month_f):
        self.initialize()
        self.predict_months(month_i, month_f, fname='knn_test')

    def initialize(self):
        logging.info('loading features')
        self.load_features()
        # Compute features with all available data
        logging.info('computing new features')
        self.parse_features()

    def parameters_search(self):
        self.initialize()
        prs = tuning.ParamsRandomSearch(
            model_class=KNeighborsRegressor,
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

    @property
    def params_lists(self):
        return {
            'n_neighbors': [3, 5, 7, 9, 15, 21],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3],
            'n_jobs': [4]
        }

    def run(self):
        self.initialize()
        # Get predictions file
        self.get_predictions_file()

    def _parse_features(self):
        feats = build_features.Features(
            target_col=TARGET_LABEL,
            month_col=MONTH_INT_LABEL
        )
        # Fill missing values
        self.features = feats.fill_lags_nans(self.features)
        max_v = self.features[MONTH_INT_LABEL].max()
        fill_px = self.features['item_px_mean_L1'].dropna().median()
        fill_vals = {
            'first_sale': (max_v, np.int8),
            'item_first_sale': (max_v, np.int8),
            'last_sale': (max_v, np.int8),
            'item_last_sale': (max_v, np.int8),
            'item_px_mean_L1': (fill_px, np.float16),
            'item_px_min_L1': (fill_px, np.float16),
            'price_mean_trend': (0.0, np.float16),
            'price_min_trend': (0.0, np.float16),
            'month_shopcat_avg_L1': (0.0, np.float16),
            'month_itemcity_avg_L1': (0.0, np.float16)
        }
        for c, (v, dtp) in fill_vals.items():
            self.features[c] = self.features[c].fillna(v).astype(dtp)
        
        # Compute totals
        totals_lags = {
            # 'item_id': [1],
            # 'shop_id': [12]
        }
        for c, lags in totals_lags.items():
            print('Adding %s total' % c)
            self.features = feats.add_totals(self.features, c, lags=lags)

        # Remove some initial rows
        self.features = self.features[self.features[MONTH_INT_LABEL] >= self.m0]
        # Add month n
        self.features['month_n'] = self.features['date_block_num']

    def features_importances(self):
        # TODO: Check why this method is so slow
        self.load_features()
        self.features = self.features.fillna(0.0)
        self._parse_features()
        fixed_cols = INDEX_COLUMNS + [TARGET_LABEL]
        cols = [c for c in self.features.columns if c not in fixed_cols]

        for c in cols:
            print('-------- Evaluating %s --------' % c)
            feats = self.features[fixed_cols + [c]]
            self.evaluate(feats[feats[MONTH_INT_LABEL] < 34])

    def parse_features(self):
        self._parse_features()
        # Drop unnecessary columns
        cols = [
            # 'month_shopcat_avg_L1',
            'item_cnt_month_L1',
            'cum_sales_L1',
            # 'month_itemcity_avg_L1',
            # 'item_id_lag1'
        ]
        self.features = self.features[cols + INDEX_COLUMNS + [TARGET_LABEL]]
        # self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')
        # Divide features by standard deviation
        for c in self.features.columns:
            if c not in INDEX_COLUMNS and c != TARGET_LABEL:
                c_std = self.features[c].std()
                if not np.isnan(c_std):
                    self.features[c] /= c_std
                    
        # Downcast
        for c in self.features.columns:
            if c not in INDEX_COLUMNS:
                self.features[c] = self.features[c].astype(np.float16)
        
        self.features.info()
        
    def get_validation_score(self):
        self.initialize()
        self.evaluate(self.n_evals)


class LastSalesModel:

    def __init__(self):
        self.last_month_label = 'lag1'

    def fit(self, x, y):
        pass

    def predict(self, x):
        return x[self.last_month_label]


class LastSales(core.Model):

    def __init__(self):
        super().__init__(LastSalesModel())
        self.last_sales = None
        self.n_evals = 24

    def run(self):
        self.get_last_sales()
        self.load_test()
        self.get_predictions()
        self.save_preditions()

    def get_validation_score(self):
        logging.info('loading features')
        self.get_last_sales()
        # Fill missing values
        logging.info('running evaluation')
        me = evaluate.ModelEvaluation(
            model_class=self, n_evals=self.n_evals
        )
        me.evaluate(self.train_feats)

    def get_last_sales(self):
        self.load_features()
        feats_cols = INDEX_COLUMNS + ['lag1']
        self.train_feats = self.train_feats[feats_cols + [TARGET_LABEL]]
        self.test_feats = self.test_feats[feats_cols]
        self.train_feats['lag1'] = self.train_feats['lag1'].fillna(0.0)
        self.test_feats['lag1'] = self.test_feats['lag1'].fillna(0.0)

    def get_predictions(self):
        ls = self.last_sales.reset_index()
        preds = self.test.merge(ls, how='left', on=['shop_id', 'item_id'])
        missing = 0
        self.predictions = preds['item_cnt_month'].fillna(missing).clip(0, 20)


if __name__ == '__main__':
    main()
