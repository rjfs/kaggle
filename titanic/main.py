import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import neighbors, model_selection
import seaborn

TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
SUBMISSION_FILE_NAME = 'submission.csv'

UNKNOWN_LABEL = 'Unknown'
output_label = 'Survived'


def discretize_vector(v):
    keys = set(v)
    translate_d = dict(zip(keys, range(len(keys))))
    return [translate_d[i] for i in v]


def scatter_plot(x, y):
    df_plt = pd.concat([x, y], axis=1)
    df_plt.plot(kind='scatter', x=x.name, y=y.name)
    plt.show()


def plot_confusion_matrix(y_pred, y_test, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_df = confusion_matrix_df(y_pred=y_pred, y_test=y_test)
    cm = cm_df.as_matrix()
    classes = list(cm_df.index)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm.astype(np.int64), interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Model')
    plt.xlabel('Real')


def confusion_matrix_df(y_pred, y_test):
    """
    Confusion matrix DataFrame with model prediction labels in rows and test labels as columns.

    :param y_pred:
    :param y_test:
    :return:
    """
    assert len(y_pred) == len(y_test)
    labels = list(set(y_pred) | set(y_test))
    cnf_df = pd.DataFrame(index=labels, columns=labels)
    for p_l in labels:
        for t_l in labels:
            eq_lst = [(y_pred[i] == p_l and y_test[i] == t_l) for i in range(len(y_pred))]
            cnf_df.loc[p_l, t_l] = len([i for i in eq_lst if i])

    return cnf_df


def get_name_common_title(name):
    common_titles = ['Miss.', 'Mrs.', 'Mr.']
    in_name = [t for t in common_titles if t in name]
    if len(in_name) == 1:
        return in_name[0]
    else:
        print 'NOT FOUND FOR', name


def get_name_title(name):
    # Assumes that title ends with a '.'
    words = name.split(' ')
    titles_in_name = [i for i in words if len(i) > 0 and i[-1] == '.']
    return titles_in_name[0] if len(titles_in_name) == 1 else get_name_common_title(name)


def get_names_title(names):
    return [get_name_title(n) for n in names]


def count_list_values(l):
    return {i: l.count(i) for i in l}


def discretize_series(s):
    cat_count = count_list_values(list(s.values))
    other_threshold = 1
    categories = [i  if cat_count[i] > other_threshold else 'Other' for i in s.values]
    return pd.Series(discretize_vector(categories), index=s.index)


def pearson_correlation(x, y):
    df_j = pd.DataFrame([x, y]).T
    return df_j.corr().iloc[0, 1]


def get_cabin_division(cabin):
    d = UNKNOWN_LABEL
    if isinstance(cabin, str):
        s_split = cabin.split(' ')
        first_l = set([i[0] for i in s_split])
        if len(first_l) == 1:
            d = cabin[0]

    return d


def get_cabin_list_divisions(cabins):
    return [get_cabin_division(c) for c in cabins]


def confusion_matrix_df(y_pred, y_test):
    """
    Confusion matrix DataFrame with model prediction labels in rows and test labels as columns.

    :param y_pred:
    :param y_test:
    :return:
    """
    assert len(y_pred) == len(y_test)
    labels = list(set(y_pred) | set(y_test))
    cnf_df = pd.DataFrame(index=labels, columns=labels)
    for p_l in labels:
        for t_l in labels:
            eq_lst = [(y_pred[i] == p_l and y_test[i] == t_l) for i in range(len(y_pred))]
            cnf_df.loc[p_l, t_l] = len([i for i in eq_lst if i])

    return cnf_df


def get_fare_category(fare):
    def classify(f):
        if f < limit1:
            return 1
        elif f < limit2:
            return 2
        else:
            return 3

    limit1 = fare.mean()
    limit2 = limit1 + fare.std()

    return [classify(f) for f in fare]


def get_features_list(df):
    # Remove unnecessary features
    features = [c for c in df.columns if c != output_label]
    # Remove features with more than half unknown data points
    features = [k for k, c in df[features].count().to_dict().items() if c >= len(df) * 0.5]
    # Remove features that are an object
    features = [k for k, t in df[features].dtypes.to_dict().items() if t != 'object']
    # Remove features with low standard deviation
    features = [k for k, std in df[features].std().to_dict().items() if std > 0.0]
    # Remove features with low correlation with the output
    out_corrs = abs(df[features+[output_label]].corr()[output_label]).to_dict()
    corr_t = 0.3
    features = [f for f in features if abs(out_corrs[f]) > corr_t]
    # Remove features from manual list
    to_remove = ['Cabin_d', 'Fare']
    features = [f for f in features if f not in to_remove]
    print 'Final features:', features

    return features


def get_null_values(df):
    return df.isnull().sum()


def plot_categories(data):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data.value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title(data.name)
    ax[0].set_ylabel('')
    data_df = data.to_frame()
    seaborn.countplot(data.name, data=data_df, ax=ax[1])
    ax[1].set_title(data.name)
    plt.show()


def main():
    # ------------ Read data ----------------
    df_train = pd.read_csv(TRAIN_DATA_FILE).set_index('PassengerId')
    df_test = pd.read_csv(TEST_DATA_FILE).set_index('PassengerId')
    df_comp = df_train.append(df_test)
    train_idx = df_train.index
    test_idx = df_test.index
    # Fill missing values
    # Missing values on Age, Cabin, Embarked
    df_comp['Age'] = df_comp['Age'].fillna(df_comp['Age'].mean())
    df_comp['Cabin'] = df_comp['Cabin'].fillna(UNKNOWN_LABEL)
    df_comp['Embarked'] = df_comp['Embarked'].fillna(UNKNOWN_LABEL)
    # ------------ Feature engineering ------------
    # Add title feature
    df_comp['title'] = get_names_title(df_comp['Name'].values)
    # Add cabin division
    df_comp['cabin_division'] = get_cabin_list_divisions(df_comp['Cabin'].values)
    # Add fare category
    df_comp['fare_category'] = get_fare_category(df_comp['Fare'])
    # Add (woman | child) feature
    children_t = 16
    df_comp['child_or_women'] = [r['Age'] < children_t or r['Sex'] == 'female' for _, r in df_comp.iterrows()]
    # Add has family
    df_comp['has_family'] = [r['SibSp'] + r['Parch'] > 0 for _, r in df_comp.iterrows()]
    # Discretize variables
    obj_variables = [k for k, v in df_comp.dtypes.to_dict().items() if v == 'object']
    for c in obj_variables:
        df_comp[c + '_d'] = discretize_series(df_comp[c])
    # Get features
    features = get_features_list(df_comp)
    # ----------------- Train Model -----------------------
    # KNN
    # Get final training DataFrame
    cols = features + [output_label]
    df_f = df_comp.loc[train_idx, cols]
    x = df_f[features].as_matrix()
    y = df_f[output_label].as_matrix()
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X=x_train, y=y_train)
    y_pred = knn.predict(x_val)
    plot_confusion_matrix(y_pred=y_pred, y_test=y_val)
    print 'Score: %.1f' % (knn.score(X=x_val, y=y_val) * 100)
    plt.show()
    # Train model with whole training set
    final_model = knn.fit(X=x, y=y)
    generate_predictions_file(model=final_model, test_df=df_comp.loc[test_idx, features])


def generate_predictions_file(model, test_df):
    pred = model.predict(test_df.values)
    df = pd.Series(pred, index=test_df.index).astype(int)
    df.index.name = 'PassengerId'
    df.name = 'Survived'
    df.to_csv(SUBMISSION_FILE_NAME, header=True)


if __name__ == '__main__':
    main()
