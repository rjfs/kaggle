import utils
import random
import os
import logging
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBRegressor
from sklearn.externals import joblib


INDEX_COLUMNS = ['date_block_num', 'shop_id', 'item_id']
MONTH_INT_LABEL = 'date_block_num'
TARGET_LABEL = 'item_cnt_month'


class Model:
    """ Use last sales values of each product """
    def __init__(self, model, standardize=False, sample=None, train_clip=None,
                 model_name=None):
        self.model = model
        self.standardize = standardize
        self.sample = sample
        self.train_clip = train_clip
        self.model_name = model_name
        self.predictions = None
        self.test = None
        self.features = None
        self.scaler = None

    @property
    def train_feats(self):
        return self.features[self.features[MONTH_INT_LABEL] < self.test_m]

    @property
    def test_feats(self):
        test_data = self.features[self.features[MONTH_INT_LABEL] == self.test_m]
        return test_data.drop(TARGET_LABEL, axis=1)

    @property
    def test_m(self):
        return self.features[MONTH_INT_LABEL].max()

    def get_predictions_file(self):
        # Train model
        logging.info('training model')
        self.train_model(self.train_feats)
        # Save model to disk
        logging.info('Saving model')
        filename = 'model.sav' if self.model_name is None else self.model_name
        fpath = utils.get_script_dir()
        joblib.dump(self.model, fpath + filename)
        # Make predictions and save
        self.predictions = self.get_predictions()
        logging.info('saving predictions')
        self.save_preditions()

    def load_features(self):
        data_path = utils.get_data_dir() + '/processed/'
        train = pd.read_hdf(data_path + 'train_features.h5')
        test = pd.read_hdf(data_path + 'test_features.h5')
        self.features = train.append(test, sort=True)

    def save_preditions(self):
        out_dir = utils.get_script_dir()
        fname = time.strftime("%Y%m%d-%H%M%S")
        fpath = '%s/outputs/%s.csv' % (out_dir, fname)
        # Create outputs dir if it does not exist yet
        base_dir = os.path.dirname(fpath)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.predictions.to_csv(fpath, header=True, index_label='ID')
        logging.info('predictions saved to %s' % fpath)

    def load_test(self):
        self.test = utils.load_raw_data('test.csv.gz', index_col='ID')

    def train_model(self, train_df, validation_df=None):
        x, y = features_target_split(train_df)
        xv, yv = None, None
        if validation_df is not None:
            xv, yv = features_target_split(validation_df)

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(x)
            x_transf = self.scaler.transform(x)
            x = pd.DataFrame(x_transf, index=x.index, columns=x.columns)
            del x_transf
            if validation_df is not None:
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
        nans = x.isnull().sum()[x.isnull().sum() > 0]
        if len(nans) > 0:
            print('Number of missing values')
            print(nans)

        if self.train_clip is not None:
            l, u = self.train_clip
            y = y.clip(l, u)
            if validation_df is not None:
                yv = yv.clip(l, u)

        print('Training with features:\n%s' % list(x.columns))
        xgb_model = isinstance(self.model, XGBRegressor)
        if validation_df is not None and xgb_model:
            self.model.fit(
                x, y,
                eval_metric='rmse',
                eval_set=[
                    (x, np.clip(y, 0., 20.)), (xv, np.clip(yv, 0., 20.))
                ],
                verbose=True,
                early_stopping_rounds=10
            )
        else:
            self.model.fit(x, y)

        if hasattr(self.model, 'coef_'):
            importances = self.model.coef_ * np.std(x.values, 0)
            print(dict(zip(x.columns, importances)))
        if hasattr(self.model, 'feature_importances_'):
            d = dict(zip(x.columns, self.model.feature_importances_))
            sorted_dict_print(d)

    def predict(self, x):
        if self.standardize:
            x = self.scaler.transform(x)

        # Predict in chunks to save memory
        preds = np.array([])
        for x_ch in utils.chunker(x, 2**19):
            preds = np.append(preds, self.model.predict(x_ch))

        return pd.Series(preds, index=x.index).clip(0., 20.)

    def get_predictions(self):
        # Load raw test file
        self.load_test()
        # Predict model
        shop_item_labels = ['shop_id', 'item_id']
        preds_df = self.test_feats[shop_item_labels]
        test_x = self.test_feats.drop(INDEX_COLUMNS, axis=1)
        preds_df[TARGET_LABEL] = self.predict(test_x)
        preds_df = self.test.merge(preds_df, on=shop_item_labels, how='outer')
        preds_df.index.name = 'ID'
        return preds_df[TARGET_LABEL].fillna(0.0).clip(0., 20.)

    def predict_months(self, month_i, month_f, fname=None):
        month_preds = []
        for m in range(month_i, month_f + 1):
            logging.info('Predicting month %d' % m)
            train = self.features[self.features[MONTH_INT_LABEL] < m]
            test = self.features[self.features[MONTH_INT_LABEL] == m]
            # Train model
            self.train_model(train)
            # Generate predictions
            test_x, test_y = features_target_split(test)
            df = test[INDEX_COLUMNS]
            preds = self.predict(test_x)
            df['item_cnt_month'] = preds
            month_preds.append(df)
            # Evaluate
            train_x, train_y = features_target_split(train)
            err_t = utils.compute_score(self.predict(train_x), train_y)
            err_v = utils.compute_score(preds, test_y)
            print('Train: %.4f' % err_t)
            print('Validation: %.4f' % err_v)

        app_df = month_preds.pop(0)
        while len(month_preds) > 0:
            app_df = app_df.append(month_preds.pop(0))

        if fname is not None:
            # Save to file
            out_dir = utils.get_script_dir()
            fpath = '%s/outputs/predictions/%s.csv' % (out_dir, fname)
            app_df.set_index(MONTH_INT_LABEL).to_csv(fpath, header=True)
            logging.info('predictions saved to %s' % fpath)

        return app_df


def features_target_split(df):
    x = df.drop(INDEX_COLUMNS + [TARGET_LABEL], axis=1)
    y = df[TARGET_LABEL]
    return x, y


def sorted_dict_print(d):
    d_view = [(v, k) for k, v in d.items()]
    d_view.sort(reverse=True)
    print(d_view)
