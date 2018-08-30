import pandas as pd
import numpy as np


class Features:

    def __init__(self, month_col, target_col):
        # self.data = features
        self.month_col = month_col
        self.target_col = target_col

    def add_price_features(self, data):
        px_label = 'item_price_last_L1'
        lag1_label = 'item_cnt_month_L1'
        needed = [self.month_col, 'item_id', px_label, lag1_label]
        px_data = data[needed]
        # [month_col, 'item_id', 'item_price_last_lag1']
        gb = px_data.groupby([self.month_col, 'item_id'])
        minmax = gb.agg({px_label: ['min', 'max']})
        minmax.columns = ['px_min', 'px_max']
        px_feats = minmax.reset_index()
        px_data = px_data.merge(px_feats, on=[self.month_col, 'item_id'])
        px_data.index = data.index
        # Compute more features
        px_data['px_range'] = px_data['px_max'] - px_data['px_min']
        px_data['px_diff_min'] = px_data[px_label] - px_data['px_min']
        px_data['px_diff_max'] = px_data['px_max'] - px_data[px_label]
        px_data['px_scale'] = px_data['px_diff_min'] / px_data['px_range']
        infs = [-np.nan, np.nan]
        px_data['px_scale'] = px_data['px_scale'].replace(infs, np.nan)
        # Select output columns
        cols = [
            'px_min', 'px_max', 'px_range', 'px_diff_min', 'px_scale',
            'px_diff_max'
        ]

        return data.join(px_data[cols])

    def add_totals(self, data, label, lags, fill_value=0.0, dtype=None):
        # Adds total sales of given column label for given list of lags
        data[self.target_col] = data[self.target_col].astype(np.float32)
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

    def fill_lags_nans(self, data):
        lags_columns = {
            'item_cnt_month': np.int16,
            'item_cnt_day_min': np.int8,
            'month_avg': np.float16,
            'month_cat_avg': np.float16,
            'month_city_avg': np.float16,
            'month_item_avg': np.float16,
            'month_itemcity_avg': np.float16,
            'month_shop_avg': np.float16,
            'month_shopcat_avg': np.float16
        }
        for l, dtp in lags_columns.items():
            # self.features = feats.fill_lags(self.features, 0, l, dtype=dtp)
            lags_cols = ['%s_L%s' % (l, i) for i in range(1, 13)]
            for c in lags_cols:
                if c in data.columns:
                    data[c] = data[c].fillna(0)
                    if dtp is not None:
                        data[c] = data[c].astype(dtp)

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
