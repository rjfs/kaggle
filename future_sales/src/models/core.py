import utils
import random
import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.externals import joblib


INDEX_COLUMNS = ['date_block_num', 'shop_id', 'item_id']
MONTH_INT_LABEL = 'date_block_num'
TARGET_LABEL = 'item_cnt_month'


class Model:
    """ Use last sales values of each product """
    def __init__(self, model, training_range, standardize=False, sample=None,
                 model_name=None, drop_index=True, n_eval_months=3):
        self.model = model
        self.training_range = training_range
        self.standardize = standardize
        self.sample = sample
        self.model_name = model_name
        self.drop_index = drop_index
        self.n_eval_months = n_eval_months
        self.features = None
        self.scaler = None

    @property
    def train(self):
        return self.features[self.features[MONTH_INT_LABEL] < self.test_m]

    @property
    def test(self):
        return self.months_features([self.test_m])

    def months_features(self, m):
        return self.features[self.features[MONTH_INT_LABEL].isin(m)]

    @property
    def test_feats(self):
        return self.test.drop(INDEX_COLUMNS + [TARGET_LABEL], axis=1)

    @property
    def test_m(self):
        return self.features[MONTH_INT_LABEL].max()

    def load_features(self):
        data_path = utils.get_data_dir() + '/processed/'
        train = pd.read_hdf(data_path + 'train_features.h5')
        test = pd.read_hdf(data_path + 'test_features.h5')
        self.features = train.append(test, sort=True)

    def predict(self, x):
        if self.standardize:
            x = self.scaler.transform(x)

        # Predict in chunks to save memory
        preds = np.array([])
        for x_ch in utils.chunker(x, 2**19):
            preds = np.append(preds, self.model.predict(x_ch))

        return pd.Series(preds, index=x.index).clip(0., 20.)

    def get_predictions_file(self):
        # Get predictions
        test_x = features_target_split(self.test, self.drop_index)[0]
        preds = self.train_and_predict(self.train, test_x)
        # Compute predictions DataFrame
        preds_df = self.test[INDEX_COLUMNS]
        preds_df[TARGET_LABEL] = preds
        test_raw = utils.fix_shop_id(utils.load_raw_data('test.csv.gz'))
        preds_df = test_raw.merge(preds_df, on=['shop_id', 'item_id'])
        # Save predictions to file
        logging.info('saving predictions')
        fpath = get_file_path()
        preds_save = preds_df.set_index('ID')[TARGET_LABEL]
        preds_save.to_csv(fpath, header=True)
        logging.info('%s predictions saved to %s' % (len(preds_save), fpath))

    def predict_months(self, month_i, month_f, block_size=1):
        month_preds = []
        rmse = {}
        for m in range(month_i, month_f + 1, block_size):
            months = list(range(m, m + block_size))
            logging.info('Predicting months %s' % months)
            test_x, test_y = features_target_split(
                df=self.months_features(months),
                drop_index=self.drop_index
            )
            # Get predictions
            preds = self.train_and_predict(
                self.features[self.features[MONTH_INT_LABEL] < m],
                test_x=test_x, test_y=test_y
            )
            df = self.months_features(months)[INDEX_COLUMNS]
            df['item_cnt_month'] = preds
            month_preds.append(df)
            # Evaluate
            err_t, err_v = self.get_rmses(
                train=self.features[self.features[MONTH_INT_LABEL] < m],
                preds=preds,
                trues=test_y
            )
            print('Train: %.4f' % err_t)
            print('Validation: %.4f' % err_v)
            rmse[m] = (err_t, err_v)

        app_df = month_preds.pop(0)
        while len(month_preds) > 0:
            app_df = app_df.append(month_preds.pop(0))

        return app_df, rmse

    def get_rmses(self, train, preds, trues):
        train_x, train_y = features_target_split(
            df=train,
            drop_index=self.drop_index
        )
        err_t = utils.compute_score(self.predict(train_x), train_y)
        err_v = utils.compute_score(preds, trues)
        return err_t, err_v
        
    def training_split(self, train_df, validation_df):
        x, y = features_target_split(train_df, self.drop_index)
        xv, yv = None, None
        if validation_df is not None:
            xv, yv = features_target_split(validation_df, self.drop_index)

        return x, y, xv, yv

    def train_and_predict(self, train, test_x, test_y=None):
        # Train model
        self.train_model(
            train,
            validation_df=test_x.join(test_y) if test_y is not None else None
        )
        return self.predict(test_x)
        
    def train_model(self, train_df, validation_df=None):
        logging.info('training model')
        # Remove values either too high or too low
        train_df[TARGET_LABEL] = train_df[TARGET_LABEL].astype(np.float16)
        train_df = train_df[train_df[TARGET_LABEL] >= self.training_range[0]]
        train_df = train_df[train_df[TARGET_LABEL] <= self.training_range[1]]
        # Split data into features and targets
        x, y, xv, yv = self.training_split(train_df, validation_df)
        has_validation = validation_df is not None and not yv.dropna().empty
        # Free memory
        del train_df
        del validation_df
        # Parse targets
        y = y.clip(0., 20.).astype(np.int16)
        if yv is not None:
            yv = yv.clip(0., 20.)
            if yv.isnull().sum() == 0:
                yv = yv.astype(np.int16)
        # Standardize data
        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(x)
            x_transf = self.scaler.transform(x)
            x = pd.DataFrame(x_transf, index=x.index, columns=x.columns)
            del x_transf
            if has_validation:
                xv_t = self.scaler.transform(xv)
                xv = pd.DataFrame(xv_t, index=xv.index, columns=xv.columns)
                del xv_t

        if self.sample is not None and self.sample < len(x):
            # Takes sample of given size from features
            lst = range(x.shape[0])
            sample_idx = sorted(random.sample(lst, self.sample))
            x = x.iloc[sample_idx]
            y = y.iloc[sample_idx]

        # Print missing values
        print_missing_values(x)

        print(
            'Training with %d features (%d rows):\n%s' % 
            (x.shape[1], x.shape[0], list(x.columns))
        )
        fit_params = {}
        xgb_model = isinstance(self.model, XGBRegressor)
        lgbm_model = isinstance(self.model, LGBMRegressor)
        if xgb_model or lgbm_model:
            fit_params['verbose'] = True
            if has_validation:
                fit_params['eval_metric'] = 'rmse'
                fit_params['eval_set'] = [(x, y), (xv, yv)]
                fit_params['early_stopping_rounds'] = 10

        # Fit model
        self.model.fit(x, y, **fit_params)

        if hasattr(self.model, 'coef_'):
            print('Coefficients:')
            importances = self.model.coef_ * np.std(x.values, 0)
            d = dict(zip(x.columns, importances))
            sorted_dict_print(d)
        if hasattr(self.model, 'feature_importances_'):
            print('Features Importances:')
            d = dict(zip(x.columns, self.model.feature_importances_))
            sorted_dict_print(d)

    def save_model(self):
        logging.info('saving model')
        filename = 'model.sav' if self.model_name is None else self.model_name
        fpath = utils.get_script_dir()
        joblib.dump(self.model, fpath + filename)
        logging.info('done')

    def save_predictions(self, preds):
        out_dir = utils.get_script_dir()
        fpath = '%s/outputs/predictions/%s.csv' % (out_dir, self.model_name)
        preds.set_index(MONTH_INT_LABEL).to_csv(fpath, header=True)
        logging.info('predictions saved to %s' % fpath)

    def evaluate(self, n_evals):
        logging.info('running evaluation')
        month_f = self.test_m - 1
        month_i = month_f - n_evals * self.n_eval_months + 1
        preds, rmse = self.predict_months(
            month_i, month_f, block_size=self.n_eval_months
        )
        print('Validation Results:')
        print(rmse)
        train_errs = np.array([i[0] for i in rmse.values()])
        val_errs = np.array([i[1] for i in rmse.values()])
        print('Mean Train: %.5f' % np.mean(train_errs))
        print('Mean Validation: %.5f' % np.mean(val_errs))

        return np.mean(val_errs)


def get_file_path():
    out_dir = utils.get_script_dir()
    fpath = '%s/outputs/%s.csv' % (out_dir, utils.get_timestamp())
    # Create outputs dir if it does not exist yet
    base_dir = os.path.dirname(fpath)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    return fpath


def features_target_split(df, drop_index):
    x = df.drop(TARGET_LABEL, axis=1)
    y = df[TARGET_LABEL]
    if drop_index:
        x = x.drop(INDEX_COLUMNS, axis=1, errors='ignore')

    return x, y


def sorted_dict_print(d):
    d_view = [(v, k) for k, v in d.items()]
    d_view.sort(reverse=True)
    for v in d_view:
        print(v[0], v[1])


def print_missing_values(x):
    nans = x.isnull().sum()[x.isnull().sum() > 0]
    if len(nans) > 0:
        print('Number of missing values')
        print(nans)
