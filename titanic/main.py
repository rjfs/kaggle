import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import neighbors, linear_model, model_selection, ensemble, svm, metrics, naive_bayes
import seaborn
import classification

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
    titles = [get_name_title(n) for n in names]
    translations = {
        'Capt.': 'Mr.', 'Countess.': 'Mrs.', 'Lady.': 'Mrs.', 'Mme.': 'Mrs.', 'Mlle.': 'Mrs.', 'Sir.': 'Mr.',
        'Major.': 'Mr.', 'Rev.': 'Mr.', 'Don.': 'Mr.', 'Col.': 'Mr.'
    }
    titles = [translations.get(t, t) for t in titles]
    return titles


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
    # correlations_heatmap(df[features + [output_label]])
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


def plot_feature_vs_output(feature, output):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data = pd.concat([feature, output], axis=1)
    data.groupby([feature.name]).mean().plot.bar(ax=ax[0])
    ax[0].set_title('%s vs %s' % (output.name, feature.name))
    seaborn.countplot(feature.name, hue=output.name, data=data, ax=ax[1])
    ax[1].set_title('%s: %s vs not %s' % (feature.name, output.name, output.name))
    plt.show()


def plot_count_table(x, y):
    return pd.crosstab(x, y, margins=True)  # .style.background_gradient(cmap='summer_r')


def factor_plot(x, y, hue):
    data = pd.concat([x, y, hue], axis=1)
    seaborn.factorplot(x.name, y.name, hue=hue.name, data=data)
    plt.show()


def violin_plot(x, y, hue):
    f, ax = plt.subplots(1, 1, figsize=(18, 8))
    data = pd.concat([x, y, hue], axis=1)
    seaborn.violinplot(x.name, y.name, hue=hue.name, data=data, split=True, ax=ax)
    ax.set_title('%s and %s vs %s' % (x.name, y.name, hue.name))
    ax.set_yticks(range(0, 110, 10))
    plt.show()


def correlations_heatmap(data):
    seaborn.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)  # data.corr()-->correlation matrix
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


class Classifier:
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


def one_hot_encoding(df):
    ohe_labels = ['Embarked', 'cabin_division', 'title']
    return df.join(pd.get_dummies(df[ohe_labels]))


class TitanicClassifier:

    def __init__(self, model, output_correlation_th, features_correlation_th):
        self.model = model
        self.output_correlation_th = output_correlation_th
        self.features_correlation_th = features_correlation_th
        self.train = None
        self.test = None
        self.train_out = None
        self.test_out = None

    def add_data(self, train, train_out, test, test_out=None):
        p_train, p_test = self.parse_data(train, train_out, test)
        self.train = p_train
        self.test = p_test
        self.train_out = train_out
        self.test_out = test_out

    def fit(self):
        self.model.fit(X=self.train.values, y=self.train_out.values)

    def get_model_score(self):
        print list(self.train.columns)
        print 'Train Score:', self.model.score(X=self.train.values, y=self.train_out.values)
        print 'Validation Score:', self.model.score(X=self.test.values, y=self.test_out.values)

    def predict_probabilities(self):
        return [x[1] for x in self.model.predict_proba(self.test.values)]

    def parse_data(self, train, train_out, test):
        df = train.append(test)
        # Missing values on Age, Cabin, Embarked
        # Age will be filled later
        df['Cabin'] = df['Cabin'].fillna(UNKNOWN_LABEL)
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
        # ------------ Feature engineering ------------
        # Add title feature
        df['title'] = get_names_title(df['Name'].values)
        # Replace Nan ages with mean title age
        title_ages = df.groupby('title')['Age'].mean().to_dict()
        for t, a in title_ages.items():
            df.loc[(df['Age'].isnull()) & (df['title'] == t), 'Age'] = a
        # Add cabin division
        df['cabin_division'] = get_cabin_list_divisions(df['Cabin'].values)
        # Add fare category
        df['fare_category'] = get_fare_category(df['Fare'])
        # Add age category
        df['age_group'] = [int(np.round(i/10.0)) for i in df['Age'].values]
        # Add is men
        df['is_men'] = df['Sex'] == 'male'
        # Add (woman | child) feature
        children_t = 10
        df['child_or_women'] = [r['Age'] <= children_t or r['Sex'] == 'female' for _, r in df.iterrows()]
        # Add is_chid
        df['is_child'] = [r['Age'] <= children_t for _, r in df.iterrows()]
        # Add has family
        df['has_family'] = [r['SibSp'] + r['Parch'] > 0 for _, r in df.iterrows()]
        # Add family
        df['family_n'] = [r['SibSp'] + r['Parch'] for _, r in df.iterrows()]
        # Apply one hot encoding
        df = one_hot_encoding(df)
        # Drop object columns
        obj_cols = [c for c in df.columns if df[c].dtype == 'object']
        df = df.drop(obj_cols, axis=1)
        # Remove correlated features
        df = self.remove_correlated_features(df)
        # Remove features with low correlation with the output
        correlations = df.corrwith(train_out).to_dict()
        features = [f for f in df.columns if abs(correlations[f]) > self.output_correlation_th]
        df = df[features]

        return df.loc[train.index], df.loc[test.index]

    def remove_correlated_features(self, df):
        corr_df = df[[c for c in df.columns if c != output_label]].corr()
        corrs_info = self.get_correlated_columns_info(corr_df)
        rmv_list = []
        while len(corrs_info) > 0:
            corrs_info = corrs_info.sort_values(['n_corr', 'avg_corr'], ascending=False)
            to_remove = corrs_info.index[0]
            print('Removing %s' % to_remove)
            corr_df = corr_df.drop(to_remove, axis=0).drop(to_remove, axis=1)
            rmv_list.append(to_remove)
            corrs_info = self.get_correlated_columns_info(corr_df)

        df = df.drop(rmv_list, axis=1)

        return df

    def get_correlated_columns_info(self, corr_df):
        corr_df_s = abs(corr_df.stack())
        # Get correlated features
        corrs = corr_df_s.loc[[i for i in corr_df_s.index if i[0] != i[1]]]
        corrs = corrs[corrs > self.features_correlation_th]
        corrs_info = {
            f: {'n_corr': len(corrs.loc[f]), 'avg_corr': corrs.loc[f].mean()}
            for f in corrs.index.levels[0]
        }
        corrs_info = pd.DataFrame.from_dict(corrs_info, orient='index')

        return corrs_info[corrs_info['n_corr'] > 0]


def get_knn_score(train_df, test_df):
    features = [f for f in train_df.columns if f != output_label]
    x_train = train_df[features].as_matrix()
    y_train = train_df[output_label].as_matrix()
    x_test = test_df[features].as_matrix()
    y_test = test_df[output_label].as_matrix()
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X=x_train, y=y_train)

    return knn.score(X=x_test, y=y_test)


class ModelEnsemble:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.prob_threshold = 0.5
        self.validation_pct = 0.4

    def get_validation_score(self):
        m = classification.ClassificationModel(
            train_df=self.train_df, test_df=self.test_df, output_label=output_label
        )
        train, train_out, validation, validation_out = m.split_train(validation_pct=self.validation_pct)
        preds = self.get_predictions(train=train, train_out=train_out, test=validation)
        hr_array = [not (preds[i] ^ validation_out.values[i]) for i in range(len(preds))]

        return float(sum(hr_array)) / len(hr_array)

    def get_predictions(self, train, train_out, test):
        knn_model = TitanicClassifier(
            model=neighbors.KNeighborsClassifier(),
            features_correlation_th=0.8,
            output_correlation_th=0.0
        )
        logreg_model = TitanicClassifier(
            model=linear_model.LogisticRegression(),
            features_correlation_th=0.8,
            output_correlation_th=0.0
        )
        models = {'knn': knn_model, 'logreg': logreg_model}
        probs = {}
        for model_name, model in models.items():
            print '>>', model_name
            model.add_data(train=train, train_out=train_out, test=test)
            model.fit()
            probs[model_name] = model.predict_probabilities()

        probs_means = np.mean(np.array(probs.values()), axis=0)

        return [x > self.prob_threshold for x in probs_means]

    def get_test_predictions(self):
        train_out = self.train_df[output_label]
        train = self.train_df.drop(output_label, axis=1)
        return self.get_predictions(train=train, train_out=train_out, test=self.test_df)


def main():
    # ------------ Read data ----------------
    df_train = pd.read_csv(TRAIN_DATA_FILE).set_index('PassengerId')
    df_test = pd.read_csv(TEST_DATA_FILE).set_index('PassengerId')
    m = ModelEnsemble(train_df=df_train, test_df=df_test)
    print m.get_validation_score()
    predictions = m.get_test_predictions()
    print predictions
    generate_predictions_file(test_df=df_test, predictions=predictions)


def generate_predictions_file(test_df, predictions):
    df = pd.Series(predictions, index=test_df.index).astype(int)
    df.index.name = 'PassengerId'
    df.name = 'Survived'
    print df
    df.to_csv(SUBMISSION_FILE_NAME, header=True)


if __name__ == '__main__':
    main()
