import logging
from xgboost import XGBRegressor
import core
import tuning
from features import build_features
from core import MONTH_INT_LABEL, TARGET_LABEL, INDEX_COLUMNS
import utils
import numpy as np
import lightgbm as lgb
import pandas as pd


class TreeBased(core.Model):

    def __init__(self, model, model_name, n_evals=3, sample=None):
        super().__init__(
            model, training_range=(0., 40.), sample=sample, standardize=False, 
            model_name=model_name, drop_index=True, n_evals=n_evals
        )

    def get_model_score(self, model):
        # Update model
        self.model = model
        # Return evaluation score
        return self.evaluate()

    def get_permutation_importances(self):
        self.initialize()
        self.permutation_importances()

    def get_validation_score(self):
        self.initialize()
        self.evaluate()

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
        # Add special shop-category combinations
        self.features = feats.add_shop_unique(self.features)
        # self.features = feats.add_not_solds(self.features)
        # Fill lags missing values
        self.features = feats.fill_lags_nans(self.features)
        # Clip release months
        c = 'release_months'
        self.features[c] = self.features[c].clip(-1, 1)
        # Fill other missing values
        fill_vals = {
            'first_sale': (-1, np.int8),
            'item_first_sale': (-1, np.int8),
            'last_sale': (-1, np.int8),
            'item_last_sale': (-1, np.int8),
            'item_px_mean_L1': (-1., np.float16),
            'item_px_min_L1': (-1., np.float16),
            'release_months': (9999, np.int16),
            'rel_shopcat': (-1., np.float16),
            'rel_shopmetacat': (-1., np.float16),
            'release_cat': (-1., np.float16),
            'release_metacat': (-1., np.float16),
            'release_shop': (-1., np.float16),
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
        fill_vals = {
            'item_price_last_L1': (-1.0, np.float16),
            'px_min': (-1.0, np.float16),
            'price_mean_trend': (0., np.float16),
            'price_min_trend': (0., np.float16),
        }
        for c, (v, dtp) in fill_vals.items():
            self.features[c] = self.features[c].fillna(v).astype(dtp)

        # Add sold features
        self.features['meta_sold'] = self.features['cum_metacat_sales_L1'] > 0
        self.features['meta_sold'] = self.features['meta_sold'].astype(np.int8)
        self.features['cat_sold'] = self.features['cum_cat_sales_L1'] > 0
        self.features['cat_sold'] = self.features['cat_sold'].astype(np.int8)
        # Remove some initial rows
        to_drop = [
            'item_cnt_day_min',
            'px_max',
            'px_range',
            'px_diff_min',
            'px_diff_max',
            'px_scale',
            'month_range',
            'n_weekend',
            'in_version',
            'icn_games',
            'sn_Moscow',
            'in_PC',
            'month_avg_L12',
            'last_sale',
            'item_first_sale'
        ]
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')
        # Copy item_id and shop_id
        self.features['item_id_cp'] = self.features['item_id']
        self.features['shop_id_cp'] = self.features['shop_id']


class LGBM(TreeBased):

    def __init__(self, boosting_type, sample=None):
        super().__init__(
            self.get_model(boosting_type),
            model_name='lgbm-' + boosting_type,
            n_evals=1,
            sample=sample
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
        """
        return {
            'boosting_type': 'dart', 'min_child_weight': 500,
            'colsample_bytree': 0.5, 'max_depth': 15, 'learning_rate': 0.3,
            'n_estimators': 100, 'reg_lambda': 3.0, 'num_leaves': 100,
            'subsample': 0.8
        }
        """
        return {
            'boosting_type': 'dart', 'min_child_weight': 100,
            'colsample_bytree': 0.6, 'max_depth': 15, 'learning_rate': 0.3,
            'n_estimators': 40, 'reg_lambda': 300.0, 'num_leaves': 3000,
            'subsample': 0.7
        }


    @property
    def params_lists(self):
        return {
            'boosting_type': [self.boosting_type],
            'min_child_weight': [300, 1000, 3000],
            'colsample_bytree': [0.3, 0.4, 0.5],
            'max_depth': [10, 12, 14],
            'learning_rate': [0.03, 0.1, 0.3],
            'n_estimators': [100, 125, 150],
            'reg_lambda': [0.1, 0.3, 1.0],
            'num_leaves': [10, 30, 100],
            'subsample': [0.6, 0.7, 0.8]
        }

    def initialize(self, lgbm_parse=True):
        logging.info('loading features')
        self.load_features()
        # Compute features with all available data
        logging.info('computing new features')
        self.parse_features()
        # LightGBM parsing
        self.features = self.features[self.features[MONTH_INT_LABEL] >= 12]
        if lgbm_parse:
            self.lgbm_parsing()

    def features_search(self):
        self.initialize(lgbm_parse=False)

        def print_iteration_info(i, params, score, best_score, best_params):
            print('[---------- Iteration %d ----------]' % i)
            print(params)
            print('Score: %.4f' % score)
            print('Best parameters (Score=%.4f):' % best_score)
            print(best_params)

        n_runs = 1000
        best_score = np.inf
        goal = 'minimize'
        best_params = None
        fname = 'all_feats_temp.h5'
        self.features.to_hdf(fname, 'df')
        # all_feats = self.features.copy()
        for i in range(n_runs):
            # Get iteration features
            all_feats = pd.read_hdf(fname, 'df')
            all_cols = list(all_feats.columns)
            params = utils.random_subset(all_cols, min_n=10, max_n=30)
            cols = list(set(params) | set(INDEX_COLUMNS + [TARGET_LABEL]))
            self.features = all_feats[cols]
            del all_feats
            # Compute score
            score = self.evaluate()

            max_cond = score > best_score and goal == 'maximize'
            min_cond = score < best_score and goal == 'minimize'
            if max_cond or min_cond:
                best_score = score
                best_params = params

            # Print iteration info to console
            print_iteration_info(i, params, score, best_score, best_params)

    def lgbm_parsing(self):
        # Clip features
        n_shops = len(set(self.features.shop_id.values))
        clip_feats = {
            'item_cnt_month_L1': (-1., 100.),
            'item_cnt_month_L2': (-1., 100.),
            'item_id_lag1': (-1., 40. * n_shops),
            'item_id_lag2': (-1., 40. * n_shops)
        }
        for c, vals in clip_feats.items():
            self.features[c] = self.features[c].clip(vals[0], vals[1])
            self.features[c] = self.features[c].astype(np.float16)
            
        to_drop = [ 
            'cat_sold',
            # 'city_code',
            'cum_cat_sales_L1', 
            'cum_metacat_sales_L1',
            # 'cum_sales_L1',
            'item_cnt_day_min_L1',
            # 'item_cnt_month_L1',
            # 'item_cnt_month_L2',
            # 'item_cnt_month_L3', 
            # 'item_cnt_month_L6', 
            'item_cnt_month_L12',
            'item_id_lag2',
            'item_first_sale',
            # 'item_last_sale'
            'item_price_last_L1',
            # 'item_px_mean_L1',
            'item_px_min_L1',
            # 'meta_cat',
            'meta_sold',
            # 'month',
            # 'month_avg_L1',
            # 'month_cat_avg_L1',
            'month_city_avg_L1',
            'month_item_avg_L3',
            'month_item_avg_L6',
            'month_itemcity_avg_L1',
            # 'month_shop_avg_L1',
            # 'month_shop_avg_L2',
            'month_shop_avg_L3',
            'month_shop_avg_L6',
            'price_mean_trend', 
            # 'price_min_trend',
            'px_min',
            'rel_shopcat',
            # 'rel_shopmetacat',
            'release_cat',
            'release_metacat',
            # 'release_months', 
            'release_shop',
            # 'subtype_code',
            'type_code',
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
