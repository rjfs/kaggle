import pandas as pd
import numpy as np


class Features:

    def __init__(self, month_col, target_col):
        # self.data = features
        self.month_col = month_col
        self.target_col = target_col
        self.windows = [1, 2]
        self.lags_columns = ['lag1', 'lag2', 'lag3', 'lag12']

    def add_price_features(self, data, fill_value=None):
        # TODO: Fill with most common price for item
        needed = [self.month_col, 'item_id', 'item_price_last_lag1', 'lag1']
        px_data = data[needed]
        # [month_col, 'item_id', 'item_price_last_lag1']
        gb = px_data.groupby([self.month_col, 'item_id'])
        px_label = 'item_price_last_lag1'
        minmax = gb.agg({px_label: ['min', 'max']})
        minmax.columns = ['px_min', 'px_max']
        px_feats = minmax.reset_index()
        px_data = px_data.merge(px_feats, on=[self.month_col, 'item_id'])
        px_data.index = data.index
        # Fill missing values
        prev_null_sales = px_data[px_data['lag1'] == 0]
        f_min, f_max = prev_null_sales[['px_min', 'px_max']].mean().values
        px_data['px_min'] = px_data['px_min'].fillna(f_min)
        px_data['px_max'] = px_data['px_max'].fillna(f_max)
        px_data[px_label] = px_data[px_label].fillna(px_data['px_max'])
        # Compute more features
        px_data['px_range'] = px_data['px_max'] - px_data['px_min']
        px_data['px_diff_min'] = px_data[px_label] - px_data['px_min']
        px_data['px_diff_max'] = px_data['px_max'] - px_data[px_label]
        px_data['px_scale'] = px_data['px_diff_min'] / px_data['px_range']
        px_data['px_scale'] = px_data['px_scale'].fillna(1.0)
        # Select output columns
        cols = [
            'px_min', 'px_max', 'px_range', 'px_diff_min', 'px_scale',
            'px_diff_max'
        ]
        # TODO: Better fill NaN strategy
        if fill_value is not None:
            # Fill NaNs
            for c in cols:
                px_data[c] = px_data[c].fillna(fill_value)

        return data.join(px_data[cols])

    def add_totals(self, data, label, lags, fill_value=0.0, dtype=None):
        # Adds total sales of given column label for given list of lags
        on = [self.month_col, label]
        gb = data.groupby(on)
        totals = gb[self.target_col].sum().unstack()
        lags_s = []
        for l in lags:
            s = totals.shift(l).stack()
            s.name = '%s_lag%s' % (label, l)
            lags_s.append(s)
            del s

        lags_df = pd.concat(lags_s, axis=1).reset_index()
        # free memory
        del totals
        del lags_s
        # Merge dfs
        out = data.merge(lags_df, on=on, how='left')
        # Fill missing values and convert type
        cols = list(lags_df.columns)
        cols.remove(self.month_col)
        cols.remove(label)
        for c in cols:
            out[c] = out[c].fillna(fill_value)
            if dtype is not None:
                out[c] = out[c].astype(dtype)

        return out

    def add_totals_interactions(self, data):
        for f in ['item_id', 'shop_id']:
            l1_lab = f + '_lag1'
            l2_lab = f + '_lag2'
            l3_lab = f + '_lag3'
            sum12_lab = f + '_sum12'
            sum123_lab = f + '_sum123'
            data[sum12_lab] = data[l1_lab] + data[l2_lab]
            if l3_lab in data.columns:
                data[sum123_lab] = data[sum12_lab] + data[l3_lab]

        return data

    def fill_lags(self, data, value):
        assert isinstance(value, int)
        for c in self.lags_columns:
            # After filling missing values, column can be converted back to int
            data[c] = data[c].fillna(value).astype(np.int32)

        return data

    def fill_counts(self, data, value):
        assert isinstance(value, int)
        c = 'item_cnt_day_min'
        if c in data.columns:
            # After filling missing values, column can be converted back to int
            data[c] = data[c].fillna(value).astype(np.int32)

        return data

    @property
    def raw_prices_columns(self):
        return [
            'item_price_%s_lag1' % f
            for f in ['min', 'max', 'last', 'mean']
        ]

    def fill_prices(self, data, value):
        for c in self.raw_prices_columns:
            if c in data.columns:
                data[c] = data[c].fillna(value)

        return data

    def one_hot_encode(self, data, label, drop_first):
        month_ohe = pd.get_dummies(
            data[label], prefix=label, drop_first=drop_first
        )
        return data.join(month_ohe)

    def add_last_sales(self, data):
        data['sum12'] = data['lag1'] + data['lag2']
        data['sum123'] = data['sum12'] + data['lag3']
        return data

    def remove_previous_months(self, data, first_m):
        # Removes months with numbers less than first_m
        return data[data[self.month_col] >= first_m]

    def remove_month(self, data, m):
        # Removes months with numbers less than first_m
        return data[data[self.month_col] != m]

    def compute_features(self, features):
        # Initialize with shop/item pair moving averages
        features = self.rolling_means(features)
        # Add shop id moving averages
        s_ma = self.column_means(features, 'shop_id')
        features = features.merge(
            s_ma, how='left', on=[self.month_col, 'shop_id']
        )
        # Add item id moving averages
        i_ma = self.column_means(features, 'item_id')
        features = features.merge(
            i_ma, how='left', on=[self.month_col, 'item_id']
        )
        # Convert some columns to int32 to save memory
        for c in [self.month_col, 'item_id', 'shop_id']:
            features[c] = features[c].astype('int32')

        return features

    def column_means(self, data, column):
        gb = data.groupby([self.month_col, column])
        totals = gb[self.target_col].sum().unstack()
        ma = moving_averages(totals.fillna(0.0), self.windows).stack(level=0)
        # Rename columns
        ma = ma.rename(columns={c: '%s-%s' % (column, c) for c in ma.columns})
        return ma.reset_index()

    def rolling_means(self, data):
        md_reshaped = data.pivot_table(
            index=self.month_col,
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


def chunks_merge(left, right, on, how, chunksize=2**18):
    # Left is the DataFrame to be chunked
    n_rows = left.shape[0]
    list_df = [left[i:i + chunksize] for i in range(0, n_rows, chunksize)]
    merged_dfs = [left_c.merge(right, on=on, how=how) for left_c in list_df]
    out = merged_dfs[0]
    while len(merged_dfs) > 0:
        out = out.append(merged_dfs.pop(0))

    print(left)
    print(out)
    aaa
    chunks_merge(data, lags_df, on=on, how='left').fillna(fill_value)

    return data.merge(lags_df, on=on, how='left').fillna(fill_value)
