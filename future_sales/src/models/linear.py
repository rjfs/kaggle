import core
import logging
import sys
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import LinearSVR
from core import MONTH_INT_LABEL, TARGET_LABEL, INDEX_COLUMNS
import utils
sys.path.append(utils.up_dir(utils.get_script_dir(), 1))
from features import build_features
import numpy as np
import tuning


class LinearModel(core.Model):

    def __init__(self, model, drop_first_ohe, standardize, model_class,
                 model_name, n_evals=2, n_eval_months=2, sample=None):
        super().__init__(
            model, sample=sample, standardize=standardize,
            training_range=(0., 30.),
            n_eval_months=n_eval_months,
            model_name=model_name
        )
        self.drop_first_ohe = drop_first_ohe
        self.features = None
        self.n_evals = n_evals
        self.model_class = model_class
        self.add_prices_features = False

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

    def parameters_search(self):
        self.initialize()
        prs = tuning.ParamsRandomSearch(
            model_class=self.model_class,
            params_lists=self.params_lists,
            eval_funct=self.get_model_score
        )
        prs.run()

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
        # Fill lags missing values
        self.features = feats.fill_lags_nans(self.features)
        # Fill other missing values
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
            'release_cat': (6.0, np.float16),
            'release_metacat': (6.0, np.float16),
            'release_months': (6.0, np.float16),
            'release_shop': (6.0, np.float16),
        }
        for c, (v, dtp) in fill_vals.items():
            self.features[c] = self.features[c].fillna(v).astype(dtp)
        # Add shop and item totals
        totals_lags = {
            'shop_id': [12],
            'item_id': [1],
            # 'item_category_id': [1, 12]
        }
        for c, lags in totals_lags.items():
            print('Computing %s totals' % c)
            self.features = feats.add_totals(
                self.features, c, lags=lags, dtype=np.int16
            )

        # Drop categorical features
        to_drop = [
            'city_code', 'item_category_id', 'meta_cat', 'subtype_code',
            'type_code'
        ]
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')
        # Add prices features
        if self.add_prices_features:
            print('adding prices features')
            self.features = feats.add_price_features(self.features)
            last_price_med = self.features['item_price_last_L1'].median()
            fill_vals = {
                'item_price_last_L1': (last_price_med, np.float16),
                'px_max': (self.features['px_max'].median(), np.float16),
                'px_min': (self.features['px_min'].median(), np.float16),
                'px_range': (0.0, np.float16),
                'px_diff_min': (0.0, np.float16),
                'px_diff_max': (0.0, np.float16),
            }
            for c, (v, dtp) in fill_vals.items():
                self.features[c] = self.features[c].fillna(v).astype(dtp)
        # Parse cumulative sales
        for c in ['cum_cat_sales_L1', 'cum_metacat_sales_L1']:
            self.features[c] = np.log1p(self.features[c]).astype(np.float16)
        
        # Remove some initial rows
        mask = self.features[MONTH_INT_LABEL] >= 12
        self.features = self.features[mask]
        # Add features interactions
        print('adding features interactions')
        # self.features['old_last_sale'] = self.features['last_sale'] > 24
        self.features['old_item_sale'] = self.features['item_last_sale'] > 24
        self.features['has_sold'] = self.features['cum_sales_L1'] > 0.0
        self.features['cat_sold'] = self.features['cum_cat_sales_L1'] > 0.0
        int_cols = [
            # 'old_last_sale',  
            'old_item_sale', 'has_sold', 'cat_sold'
        ]
        for c in int_cols:
            self.features[c] = self.features[c].astype(np.int8)
        self.features['px_mean_min_diff'] = (
            self.features['item_px_mean_L1'] - self.features['item_px_min_L1']
        )
        self.features['px_last_min_diff'] = (
            self.features['item_price_last_L1'] -
            self.features['item_px_min_L1']
        ).fillna(0.0)
        
        to_drop = [
            'cum_cat_sales_L1', 'cum_metacat_sales_L1', 'cum_sales_L1',
            'item_cnt_day_min_L1', 'item_px_min_L1', 'month',
            'rel_shopcat', 'rel_shopmetacat', 'month_item_avg_L2',
            'month_item_avg_L6'
        ]
        self.features.drop(to_drop, axis=1, inplace=True, errors='ignore')
        
        self.select_features()
        self.features.info()

    def select_features(self):
        """
        Select features based on correlations:
            - Compute correlations dropping test and validation data in order
            to avoid leakage.
            - Drop features with low correlation with target
            - Drop some features in order to avoid having too correlated
            features.
        """
        low_threshold = 0.05
        high_threshold = 0.8
        # Drop uncorrelated features
        print('dropping uncorrelated')
        last_m = last_correlation_month(self.n_evals, self.n_eval_months)
        mask = self.features[MONTH_INT_LABEL] <= last_m
        corrs = self.features[mask].drop(INDEX_COLUMNS, axis=1).corr()
        to_drop = [
            c for c in self.features.columns if c not in INDEX_COLUMNS
            and abs(corrs.loc[c, TARGET_LABEL]) < low_threshold
        ]
        self.features.drop(to_drop, axis=1, errors='ignore', inplace=True)
        corrs = corrs.drop(to_drop, axis=0).drop(to_drop, axis=1)
        # Drop too correlated features (between themselves)
        selected = get_low_correlated(corrs, TARGET_LABEL, high_threshold)
        self.features = self.features[selected + INDEX_COLUMNS]

    def get_validation_score(self):
        self.initialize()
        self.evaluate(self.n_evals)

    def get_months_predictions(self, month_i, month_f):
        self.initialize()
        preds, rmse = self.predict_months(month_i, month_f)
        # Save to file
        self.save_predictions(preds)

    def get_model_score(self, model):
        # Update model
        self.model = model
        # Return evaluation score
        return self.evaluate(self.n_evals)


def get_low_correlated(corrs_r, target_label, high_thresh):
    print('Removing high correlated features')
    corrs = corrs_r.abs()
    target_corrs = corrs[target_label]
    corrs = corrs.drop(target_label, axis=0).drop(target_label, axis=1)
    # Set upper triangle to NaN
    mask = np.zeros(corrs.shape, dtype='bool')
    mask[np.triu_indices(len(corrs))] = True
    corrs[mask] = np.nan

    to_rem = []
    to_keep = []
    while corrs.max().max() > high_thresh:
        max_corr_idx = list(corrs.stack().sort_values().index[-1])
        corr_a = target_corrs[max_corr_idx[0]]
        corr_b = target_corrs[max_corr_idx[1]]
        rem, keep = max_corr_idx if corr_a < corr_b else max_corr_idx[::-1]
        print('Removing %s %s' % (rem, max_corr_idx))
        to_keep.append(keep)
        to_rem.append(rem)

        corrs = corrs.drop(max_corr_idx, axis=0).drop(max_corr_idx, axis=1)

    return [c for c in corrs_r.columns if c not in to_rem]


class LinearRegressionModel(LinearModel):

    def __init__(self, sample=None):
        super().__init__(
            LinearRegression(),
            model_name='lin_reg',
            drop_first_ohe=True,
            standardize=False,
            sample=sample,
            model_class=LinearRegression
        )


class LinearSVM(LinearModel):

    def __init__(self, sample=None):
        super().__init__(
            LinearSVR(**self.tuned_params),
            drop_first_ohe=False,
            model_name='svm',
            standardize=False,
            n_evals=1,
            sample=sample,
            model_class=LinearSVR
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


class RidgeRegression(LinearModel):

    def __init__(self, sample=None):
        super().__init__(
            Ridge(alpha=3.0, normalize=False),
            drop_first_ohe=False,
            standardize=False,
            n_evals=1,
            n_eval_months=3,
            sample=sample,
            model_class=Ridge,
            model_name='ridge',
        )

    @property
    def params_lists(self):
        return {
            'alpha': [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            'normalize': [False, True]
        }


class LassoRegression(LinearModel):

    def __init__(self, sample=None):
        # params = {'alpha': 2.0, 'fit_intercept': False, 'normalize': True}
        params = {'alpha': 0.01, 'fit_intercept': True, 'normalize': False}
        super().__init__(
            Lasso(**params),
            drop_first_ohe=False,
            standardize=False,
            n_evals=1,
            n_eval_months=3,
            sample=sample,
            model_name='lasso',
            model_class=Lasso
        )

    @property
    def params_lists(self):
        return {
            'alpha': [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 20.0],
            'fit_intercept': [False, True],
            'normalize': [True, False]
        }


def last_correlation_month(n_evals, n_eval_m, test_m=34):
    return test_m - n_evals * n_eval_m - 1


def test_last_corr():
    assert last_correlation_month(1, 2) == 31
    assert last_correlation_month(1, 1) == 32
    assert last_correlation_month(2, 1) == 31
    assert last_correlation_month(2, 2) == 29
    assert last_correlation_month(1, 2, 32) == 29
