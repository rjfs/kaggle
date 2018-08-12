import pandas as pd
import numpy as np
from itertools import product


def unseen_pairs_split(past, present):
    subset = ['shop_id', 'item_id']
    past_un = past.drop_duplicates(subset=subset)
    conc = pd.concat([past_un, present], keys=['train', 'test'], sort=True)
    known_pairs = conc.duplicated(subset=subset, keep=False).loc['test']
    return present[~known_pairs], present[known_pairs]


def drop_unseen_pairs(df):
    first_m = df['date_block_num'].min()
    out = df[df['date_block_num'] == first_m]
    for m in df['date_block_num'].unique():
        if m != first_m:
            past = df[df['date_block_num'] < m]
            curr = df[df['date_block_num'] == m]
            out = out.append(unseen_pairs_split(past, curr)[1])

    return out


def remove_zero_sales(df, label, target_label):
    sales = df.groupby(label)[target_label].sum()
    return df[~df[label].isin(sales[sales == 0].index)]


def add_known_pairs(df, target_label):
    subset = ['shop_id', 'item_id']
    month_label = 'date_block_num'
    first_m = df[month_label].min()
    out = df[df[month_label] == first_m]
    known_pairs = out[subset]
    for m in df[month_label].unique():
        if m != first_m:
            past = df[df[month_label] < m]
            curr = df[df[month_label] == m]
            to_concat = known_pairs.copy()
            to_append = curr.merge(to_concat, on=subset, how='outer', sort=True)
            to_append[month_label].fillna(m, inplace=True)
            to_append[month_label] = to_append[month_label].astype(np.int32)
            to_append[target_label].fillna(0.0, inplace=True)
            # Remove shops and items with no sales
            shop_sales = to_append.groupby('shop_id')[target_label].sum()
            zero_sales = shop_sales[shop_sales == 0].index
            to_append = to_append[~to_append['shop_id'].isin(zero_sales)]
            # Remove items with no sales
            to_append = remove_zero_sales(to_append, 'shop_id', target_label)
            to_append = remove_zero_sales(to_append, 'item_id', target_label)
            # Append
            out = out.append(to_append, sort=True)
            # Update known pairs
            unseen = unseen_pairs_split(past, curr)[0]
            new_pairs = unseen[subset].drop_duplicates()
            known_pairs = known_pairs.append(new_pairs)

    return out


def fix_train_distribution(train, target_label):
    """
    Fixes train distribution to be similar to test's

    For every month, create a grid from all shops/items combinations from that
    month.
    """
    index_cols = ['date_block_num', 'shop_id', 'item_id']
    grid = []
    for block_num in train['date_block_num'].unique():
        block_mask = train['date_block_num'] == block_num
        cur_shops = train.loc[block_mask, 'shop_id'].unique()
        cur_items = train.loc[block_mask, 'item_id'].unique()
        prod = product(*[[block_num], cur_shops, cur_items])
        grid.append(np.array(list(prod), dtype='int32'))

    grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)
    train_p = train.merge(grid, on=index_cols, how='outer')
    train_p = train_p.sort_values(index_cols)
    train_p[target_label] = train_p[target_label].fillna(0)

    return train_p


def fix_distributions(train, test, target_label):
    """ Fixed distributions by changing both train and test sets """
    # Drop unseen pairs
    train_red = drop_unseen_pairs(train)
    test_red = unseen_pairs_split(train, test)[1]
    # Add known pairs with target zero
    train_red = add_known_pairs(train_red, target_label=target_label)

    return train_red, test_red
