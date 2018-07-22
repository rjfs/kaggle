import os
import pandas as pd
import datetime


def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_data_dir():
    dir_lst = get_script_dir().split('/')[:-2] + ['data']
    return '/'.join(dir_lst)


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

    return df
