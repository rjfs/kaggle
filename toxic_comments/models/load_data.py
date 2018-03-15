import pandas as pd


def load_train_val_test(data_path, clean=True):
    # Load data
    train_fname = 'train-train.csv'
    test_fname = 'test.csv'
    if clean:
        train_fname = 'clean-' + train_fname
        test_fname = 'clean-' + test_fname

    train = pd.read_csv(data_path + train_fname, index_col='id')
    val_path = get_validation_path(data_path, clean)
    val = pd.read_csv(val_path, index_col='id')
    test = pd.read_csv(data_path + test_fname, index_col='id')
    train['comment_text'] = train['comment_text'].astype(str)
    val['comment_text'] = val['comment_text'].astype(str)
    test['comment_text'] = test['comment_text'].astype(str)

    return train, val, test


def get_validation_path(data_path, clean):
    val_fname = 'train-val.csv'
    if clean:
        val_fname = 'clean-' + val_fname

    return data_path + val_fname
