import logging
from xgboost import XGBRegressor
import core
import tuning
from features import build_features
from core import MONTH_INT_LABEL, TARGET_LABEL
import numpy as np
import lightgbm as lgb


class TreeBased(core.Model):

    def __init__(self, model, model_name, n_evals=3, sample=None):
        super().__init__(
            model, training_range=(0., 40.), sample=sample, standardize=False, 
            model_name=model_name, drop_index=False
        )
        self.features = None
        self.n_evals = n_evals

    def get_model_score(self, model):
        # Update model
        self.model = model
        # Return evaluation score
        return self.evaluate(self.n_evals)

    def get_validation_score(self):
        self.initialize()
        self.evaluate(self.n_evals)

    def get_months_predictions(self, month_i, month_f):
        self.initialize()
        preds, rmse = self.predict_months(month_i, month_f)
        # Save to file
        self.save_predictions(preds)

    def run(self):
        self.initialize()
        # Get predictions file
        self.get_predictions_file()

    def parameters_search(self):
        self.initialize()
        prs = tuning.ParamsRandomSearch(
            model_class=self.model_class,
            params_lists=self.params_lists,
            eval_funct=self.get_model_score
        )
        prs.run()

    def parse_features(self):
        feats = build_features.Features(
            target_col=TARGET_LABEL,
            month_col=MONTH_INT_LABEL
        )
        # Fill lags missing values
        self.features = feats.fill_lags_nans(self.features)
        # Fill other missing values
        fill_vals = {
            'first_sale': (-1, np.int8),
            'item_first_sale': (-1, np.int8),
            'last_sale': (-1, np.int8),
            'item_last_sale': (-1, np.int8),
            'item_px_mean_L1': (-1., np.float16),
            'item_px_min_L1': (-1., np.float16)
        }
        for c, (v, dtp) in fill_vals.items():
            self.features[c] = self.features[c].fillna(v).astype(dtp)

        # Add shop and item totals
        totals_lags = {
            'item_id': [1, 2],
            # 'shop_id': [12],
            # 'item_category_id': [12]
        }
        for c, lags in totals_lags.items():
            print('Adding %s total' % c)
            self.features = feats.add_totals(
                self.features, c, lags=lags, dtype=np.int16
            )

        # Parse prices features
        self.features = feats.add_price_features(self.features)
        c = 'item_price_last_L1'
        self.features[c] = self.features[c].fillna(-1.0)
        for c in ['price_mean_trend', 'price_min_trend']:
            self.features[c] = self.features[c].fillna(0.)
        # Add features interactions
        # self.features['trend'] = self.features['lag1'] - self.features['lag2']
        # self.features['trend_it'] = (
        #     self.features['item_id_lag1'] - self.features['item_id_lag2']
        # )
        # Remove some initial rows
        to_drop = [
            # 'item_category_id',
            'item_cnt_day_min',
            # 'month',
            # 'px_min',
            'px_max',
            'px_range',
            'px_diff_min',
            'px_diff_max',
            'px_scale',
            'month_range',
            'n_weekend',
            'in_version',
            'icn_games',
            # 'item_cnt_day_min_L1',
            'sn_Moscow',
            'type_code',
            'item_cnt_month_L1',
            'item_cnt_month_L2',
            'item_cnt_month_L12',
            'in_PC',
            'month_avg_L12'
            # 'item_id_lag1', 'item_id_lag2'
        ]
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')


class LGBM(TreeBased):

    def __init__(self, boosting_type):
        super().__init__(
            self.get_model(boosting_type),
            model_name='lgbm-' + boosting_type,
            n_evals=1,
            sample=None
        )
        self.boosting_type = boosting_type
        self.model_class = lgb.LGBMRegressor

    def get_model(self, boosting_type):
        if boosting_type == 'gbdt':
            return lgb.LGBMRegressor(**self.gbdt_params)
        elif boosting_type == 'dart':
            return lgb.LGBMRegressor(**self.dart_params)
        else:
            raise Exception('Invalid boosting type')

    @property
    def gbdt_params(self):
        """
        return {
            'boosting_type': 'gbdt', 'min_child_weight': 30,
            'colsample_bytree': 0.4, 'max_depth': 20, 'learning_rate': 0.05,
            'n_estimators': 70, 'reg_lambda': 0.0, 'num_leaves': 3000,
            'subsample': 0.8
        }
        """
        return {
            'boosting_type': 'gbdt', 'min_child_weight': 1000,
            'colsample_bytree': 1.0, 'max_depth': 10, 'learning_rate': 0.1,
            'n_estimators': 150, 'reg_lambda': 30.0, 'num_leaves': 100,
            'subsample': 1.0
        }
        
    @property
    def dart_params(self):
        return {
            'boosting_type': 'dart', 'min_child_weight': 500,
            'colsample_bytree': 0.5, 'max_depth': 15, 'learning_rate': 0.3,
            'n_estimators': 100, 'reg_lambda': 3.0, 'num_leaves': 100,
            'subsample': 0.8
        }

    @property
    def params_lists(self):
        return {
            'boosting_type': [self.boosting_type],
            'min_child_weight': [3, 10, 30, 100, 300, 1000],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.8, 1.0],
            'max_depth': [8, 10, 12, 15, 20, 30],
            'learning_rate': [0.05, 0.1, 0.3],
            'n_estimators': [100, 125, 150],
            'reg_lambda': [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
            'num_leaves': [30, 100, 300, 1000, 3000, 10000],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        }

    def initialize(self):
        logging.info('loading features')
        self.load_features()
        # Compute features with all available data
        logging.info('computing new features')
        self.parse_features()
        # LightGBM parsing
        self.features = self.features[self.features[MONTH_INT_LABEL] >= 12]
        self.features['meta_sold'] = self.features['cum_metacat_sales_L1'] > 0
        self.features['meta_sold'] = self.features['meta_sold'].astype(np.int8)
        self.features['cat_sold'] = self.features['cum_cat_sales_L1'] > 0
        self.features['cat_sold'] = self.features['cat_sold'].astype(np.int8)
        
        to_drop = [
            'px_min', 
            'item_cnt_month_L3', 
            'price_min_trend',
            'item_cnt_month_L6', 
            'month_shop_avg_L6', 
            'month_avg_L1',
            'cum_cat_sales_L1', 'cum_metacat_sales_L1',
            # 'meta_cat',
            # 'item_last_sale'
            # 'item_first_sale'
        ]
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')
        self.features.info()


class TunedGBDT(TreeBased):

    def __init__(self):
        super().__init__(
            XGBRegressor(**self.model_params),
            model_name='xgb',
            n_evals=1,
            sample=None
        )
        self.model_class = XGBRegressor

    @property
    def model_params(self):
        return {
            'min_child_weight': 300, 'colsample_bytree': 0.4, 'max_depth': 20,
            'learning_rate': 0.1, 'n_estimators': 125, 'lambda': 0.3,
            'subsample': 0.8, 'tree_method': 'auto'
        }

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
            'min_child_weight': [10, 30, 50, 100, 300, 500, 1000, 2000],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.8, 1.0],
            'max_depth': [8, 10, 12, 15, 20],
            'learning_rate': [0.05, 0.1, 0.3],
            'n_estimators': [100, 125, 150],
            'lambda': [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
            'subsample': [0.8, 0.9, 1.0],
            'tree_method': ['auto']
        }

    def initialize(self):
        logging.info('loading features')
        self.load_features()
        # Compute features with all available data
        logging.info('computing new features')
        self.parse_features()
        # XGB parsing
        self.features = self.features[self.features[MONTH_INT_LABEL] >= 18]
        to_drop = [
            'item_cnt_month_L3', 'price_min_trend',
            'item_cnt_month_L6', 'month_avg_L1', 'px_min',
            'month_shop_avg_L3', 'month_shop_avg_L6', 'city_code'
        ]
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')
        self.features.info()
