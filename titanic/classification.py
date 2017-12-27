import datetime
import pandas as pd
import random

LABEL_UNKNOWN = 'Unknown'
LABEL_OTHER = 'Other'


class ClassificationModel:

    def __init__(self, train_df, test_df, output_label):
        self.train_df = train_df
        self.test_df = test_df
        self.output_label = output_label
        self.category_encode = None
        self.seed = 28

    @property
    def features(self):
        return self.train_df[[c for c in self.train_df.columns if c != self.output_label]]

    @property
    def output(self):
        return self.train_df[self.output_label]

    def get_type_features(self, dtype):
        return [f for f, t in self.features.dtypes.items() if t == dtype]

    def drop_feature(self, feature):
        self.train_df.drop(feature, axis=1, inplace=True)

    def get_uncorrelated_features(self, candidates):
        timestamp_print('Computing correlations')
        corr_df = self.train_df[candidates].corr()
        corrs_info = get_correlated_columns_info(corr_df)
        while len(corrs_info) > 0:
            corrs_info = corrs_info.sort_values(['n_corr', 'avg_corr'], ascending=False)
            to_remove = corrs_info.index[0]
            timestamp_print('Removing %s' % to_remove)
            corr_df = corr_df.drop(to_remove, axis=0).drop(to_remove, axis=1)
            candidates.remove(to_remove)
            corrs_info = get_correlated_columns_info(corr_df)

        return candidates

    def split_train(self, validation_pct=0.2, how='random', sort=True):
        if how != 'random':
            raise NotImplementedError()

        n_val = int(len(self.train_df) * validation_pct)
        random.seed(self.seed)
        val_idx = random.sample(self.train_df.index, n_val)
        train_idx = [i for i in self.train_df.index if i not in val_idx]

        train_df = self.train_df.loc[train_idx]
        val_df = self.train_df.loc[val_idx]
        if sort:
            train_df = train_df.sort_index()
            val_df = val_df.sort_index()

        train_out = train_df[self.output_label]
        train = train_df.drop(self.output_label, axis=1)
        val_out = val_df[self.output_label]
        val = val_df.drop(self.output_label, axis=1)

        return train, train_out, val, val_out

    @property
    def x_train(self):
        train_features = [c for c in self.train_df.columns if c != self.output_label]
        return self.train_df[train_features].as_matrix()

    @property
    def y(self):
        return self.train_df[self.output_label].as_matrix()

    @property
    def x_test(self):
        return self.test_df.as_matrix()


def timestamp_print(msg):
    print ('[%s] %s' % (get_current_timestamp().replace(microsecond=0), msg))


def get_current_timestamp():
    return datetime.datetime.now()


def one_hot_encode(df, category_encode):
    timestamp_print('Encoding features')
    for c, enc in category_encode.items():
        df[c] = df[c].apply(lambda x: enc.get(x, LABEL_OTHER))
    new_features = pd.get_dummies(df)
    # Remove Other columns
    cols = [c for c in new_features.columns if '_'+LABEL_OTHER not in c]
    # Join to DataFrame
    df = new_features[cols]

    return df


def get_correlated_columns_info(corr_df):
    corr_df_s = abs(corr_df.stack())
    # Get correlated features
    corr_threshold = 0.9
    corrs = corr_df_s.loc[[i for i in corr_df_s.index if i[0] != i[1]]]
    corrs = corrs[corrs > corr_threshold]
    corrs_info = {
        f: {'n_corr': len(corrs.loc[f]), 'avg_corr': corrs.loc[f].mean()}
        for f in corrs.index.levels[0]
    }
    corrs_info = pd.DataFrame.from_dict(corrs_info, orient='index')

    return corrs_info[corrs_info['n_corr'] > 0]
