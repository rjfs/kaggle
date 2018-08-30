import os
import pandas as pd
import datetime
import numpy as np
import time


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def downcast_dtypes(df):
    """
    Changes column types in the DataFrame:
        `float64` type to `float32`
        `int64`   type to `int32`
    """
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df


def fix_shop_id(df):
    # Якутск Орджоникидзе, 56
    df.loc[df.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    df.loc[df.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    df.loc[df.shop_id == 10, 'shop_id'] = 11
    return df


def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))


def up_dir(path, n):
    return '/'.join(path.split('/')[:-n])


def get_data_dir():
    return up_dir(get_script_dir(), 2) + '/data'


def load_raw_data(fname, **read_csv_args):
    return pd.read_csv(get_data_dir() + '/raw/' + fname, **read_csv_args)


def load_training(load_pct=1.0, parse_dts=True):
    dtype = {
        'date': 'object',
        'date_block_num': 'int8',
        'shop_id': 'int8',
        'item_id': 'int16',
        'item_price': 'float32',
        'item_cnt_day': 'int32'
    }
    df = load_raw_data('sales_train.csv.gz', dtype=dtype)
    df = df.iloc[:int(load_pct*len(df))]
    if parse_dts:
        dts_conv = {
            dt: datetime.datetime.strptime(dt, '%d.%m.%Y')
            for dt in df['date'].unique()
        }
        df['date'] = df['date'].apply(lambda x: dts_conv[x])

    return fix_shop_id(df)
    
    
def load_test():
    dtype = {'shop_id': 'int8', 'item_id': 'int16', 'ID': 'int32'}
    df = load_raw_data('test.csv.gz', dtype=dtype)
    return fix_shop_id(df)


def load_monthly_data(target_label='item_cnt_month', **load_args):
    data = load_training(**load_args)
    # Aggregate monthly data
    gb_list = ['date_block_num', 'shop_id', 'item_id']
    monthly_data = data.groupby(gb_list)['item_cnt_day'].sum()
    monthly_data.name = target_label
    monthly_data = monthly_data.reset_index()
    # Remove pairs with zero sales
    mask = monthly_data[target_label] != 0.0
    monthly_data = monthly_data[mask]

    return monthly_data


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def compute_score(y1, y2):
    return ((y1.clip(0., 20.) - y2.clip(0., 20.)) ** 2).mean() ** .5
