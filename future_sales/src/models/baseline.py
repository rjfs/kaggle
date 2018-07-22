"""

TODO:
    - Week 3
        - Decrease memory usage
        - Debug validation implementation
        - Try different models: RandomForest, ...
    - Explore item price relative to other shops
    - Moving averages of target variations
    - Explore negative item_cnt_day

"""
import utils
import time
import os
import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

SORTING_ORDER = ['date_block_num', 'shop_id', 'item_id']
MONTH_INT_LABEL = 'date_block_num'
TARGET_LABEL = 'item_cnt_month'


def main():
    m = MeanEncodingsGBM()
    # m.run()
    m.find_best_model()


class Model:
    """ Use last sales values of each product """
    def __init__(self):
        self.data = None
        self.monthly_data = None
        self.predictions = None
        self.test = None

    def load_data(self, merge_test=False, **load_args):
        self.data = utils.load_training(**load_args)
        # Aggregate monthly data
        cnt_label = 'item_cnt_day'
        self.monthly_data = self.data.groupby(SORTING_ORDER)[cnt_label].sum()
        self.monthly_data.name = TARGET_LABEL
        self.monthly_data = self.monthly_data.reset_index()
        # Remove pairs with zero sales
        mask = self.monthly_data[TARGET_LABEL] != 0.0
        self.monthly_data = self.monthly_data[mask]

        if merge_test:
            self.load_test()
            last_num = self.monthly_data[MONTH_INT_LABEL].max() + 1
            to_add = self.test.copy()
            to_add[MONTH_INT_LABEL] = last_num
            to_add[TARGET_LABEL] = np.nan
            self.monthly_data = self.monthly_data.append(to_add, sort=True)

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


class MeanEncodingsGBM(Model):

    def __init__(self):
        super().__init__()
        # model = LinearRegression()
        self.model = XGBRegressor()
        self.features = None

    def run(self):
        logging.info('loading data')
        self.load_data(load_pct=1.0, parse_dts=False, merge_test=True)
        # Compute features with all available data
        logging.info('computing features')
        self.compute_features(self.monthly_data)
        # Train linear regression
        logging.info('training model')
        self.train_model()
        # Make predictions and save
        self.predictions = self.get_predictions()
        logging.info('saving predictions')
        self.save_preditions()

    def find_best_model(self):
        self.load_data(load_pct=1.0, parse_dts=False, merge_test=False)
        model = XGBRegressor()
        me = ModelEvaluation(model, data=self.monthly_data, n_evals=2)
        me.evaluate()

    def train_model(self):
        # Fit model
        train_df = self.features.dropna().merge(
            self.monthly_data.dropna(), how='left', on=SORTING_ORDER
        )
        train_df = train_df.sample(2**20)  # 2**20 ~ 1M
        x = train_df.drop(SORTING_ORDER + [TARGET_LABEL], axis=1)
        y = train_df[TARGET_LABEL].fillna(0.0)
        self.model.fit(x, y)

    def get_predictions(self):
        # Predict model
        m = self.monthly_data[MONTH_INT_LABEL].max()
        test_month = self.monthly_data[self.monthly_data[MONTH_INT_LABEL] == m]
        test_df = self.features.merge(test_month, on=SORTING_ORDER)
        test_x = test_df.drop(SORTING_ORDER + [TARGET_LABEL], axis=1)
        test_df[TARGET_LABEL] = self.model.predict(test_x)
        cols = ['shop_id', 'item_id', TARGET_LABEL]
        on = ['shop_id', 'item_id']
        # test_month instead of self.test on next line?
        preds_df = self.test.merge(test_df[cols], on=on, how='outer')
        preds_df.index.name = 'ID'
        return preds_df[TARGET_LABEL].fillna(0.0).clip(0., 20.)

    def compute_features(self, data):
        fts = Features(data)
        self.features = fts.compute_features()


class ModelEvaluation:

    def __init__(self, model, data, n_evals=3):
        self.model = model
        self.data = data
        self.n_evals = n_evals

    def evaluate(self):
        max_m = self.data[MONTH_INT_LABEL].max()
        for i in range(max_m - self.n_evals + 1, max_m + 1):
            curr_f = i - max_m + self.n_evals
            logging.info('Fold %d/%d' % (curr_f, self.n_evals))
            # Compute fold data
            fold_data = self.data[self.data[MONTH_INT_LABEL] <= i]
            row_idx = fold_data[MONTH_INT_LABEL] == i
            fold_data.loc[row_idx, TARGET_LABEL] = np.nan
            # Compute fold features
            fts = Features(fold_data)
            fold_features = fts.compute_features()
            # Fit model
            train_df = fold_features.dropna().merge(
                fold_data.dropna(), how='left', on=SORTING_ORDER
            )
            train_df = train_df.sample(2 ** 20)  # 2**20 ~ 1M
            x = train_df.drop(SORTING_ORDER + [TARGET_LABEL], axis=1)
            y = train_df[TARGET_LABEL].fillna(0.0)
            self.model.fit(x, y)
            # Compute score
            preds = self.get_predictions(fold_data, fold_features)
            eval_month = self.data[self.data[MONTH_INT_LABEL] == i]
            preds_arr = preds[TARGET_LABEL].values
            trues_arr = eval_month[TARGET_LABEL].clip(0., 20.).values
            rmse = ((preds_arr - trues_arr)**2).mean() ** .5
            print(rmse)

    def get_predictions(self, data, features):
        # Compute predictions
        m = data[MONTH_INT_LABEL].max()
        test_month = data[data[MONTH_INT_LABEL] == m].dropna(axis=1, how='all')
        test_df = features.merge(test_month, on=SORTING_ORDER)
        test_x = test_df.drop(SORTING_ORDER, axis=1)
        test_df[TARGET_LABEL] = self.model.predict(test_x)
        test_df[TARGET_LABEL] = test_df[TARGET_LABEL]
        # Compute scores
        cols = ['shop_id', 'item_id', TARGET_LABEL]
        on = ['shop_id', 'item_id']
        preds_df = test_month.merge(test_df[cols], on=on, how='outer')
        preds_df[TARGET_LABEL].fillna(0.0, inplace=True)
        preds_df[TARGET_LABEL].clip(0., 20., inplace=True)
        return preds_df


class Features:

    def __init__(self, data):
        self.data = data
        self.windows = [1, 2]

    def compute_features(self):
        # Initialize with shop/item pair moving averages
        features = self.rolling_means()
        # Add shop id moving averages
        s_ma = self.column_means('shop_id')
        features = features.merge(
            s_ma, how='left', on=[MONTH_INT_LABEL, 'shop_id']
        )
        # Add item id moving averages
        i_ma = self.column_means('item_id')
        features = features.merge(
            i_ma, how='left', on=[MONTH_INT_LABEL, 'item_id']
        )
        # Convert some columns to int32 to save memory
        for c in [MONTH_INT_LABEL, 'item_id', 'shop_id']:
            features[c] = features[c].astype('int32')

        return features

    def column_means(self, column):
        gb = self.data.groupby([MONTH_INT_LABEL, column])
        totals = gb[TARGET_LABEL].sum().unstack()
        ma = moving_averages(totals.fillna(0.0), self.windows).stack(level=0)
        # Rename columns
        ma = ma.rename(columns={c: '%s-%s' % (column, c) for c in ma.columns})
        return ma.reset_index()

    def rolling_means(self):
        md_reshaped = self.data.pivot_table(
            index=MONTH_INT_LABEL,
            columns=['shop_id', 'item_id'],
            values='item_cnt_month'
        )
        df = moving_averages(md_reshaped.fillna(0.0), self.windows)
        # Stack using for loop (faster than using pandas stack directly)
        shop_dfs = []
        for shop in sorted(set(df.columns.get_level_values(0))):
            sdf = df[shop].stack(level=0).reset_index()
            sdf['shop_id'] = shop
            shop_dfs.append(sdf)

        return pd.concat(shop_dfs)


def moving_averages(df, windows):
    """
    Computes moving averages with given windows
    :param df: pandas.DataFrame
    :param windows: list of ints
    :return: pandas.DataFrame
    """
    rmeans = []
    for w in windows:
        ma = df.shift(1) if w == 1 else df.rolling(w).mean().shift(1)
        # Change type to save memory
        rmeans.append(ma.astype('float32'))

    names = ['RM%s' % w for w in windows]
    return pd.concat(rmeans, keys=names).unstack(level=0)


class LastSales(Model):

    def __init__(self):
        super().__init__()
        self.last_sales = None

    def run(self):
        self.load_data()
        self.add_features()
        self.get_last_sales()
        self.load_test()
        self.get_predictions()
        self.save_preditions()

    def add_features(self):
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month

    def get_last_sales(self):
        prev_month = (self.data['month'] == 10) & (self.data['year'] == 2015)
        group_idxs = ['shop_id', 'item_id']
        df = self.data[prev_month].groupby(group_idxs)['item_cnt_day'].sum()
        self.last_sales = df.to_frame('item_cnt_month')

    def get_predictions(self):
        ls = self.last_sales.reset_index()
        preds = self.test.merge(ls, how='left', on=['shop_id', 'item_id'])
        missing = 0
        self.predictions = preds['item_cnt_month'].fillna(missing).clip(0, 20)


if __name__ == '__main__':
    main()
