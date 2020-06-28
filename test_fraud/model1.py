# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import lightgbm as lgb
from sklearn import metrics

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
columns = [
    'TransactionDT',
    'card1',
    'addr1',
    'D2', 'D15',
    'V127', 'V159', 'V160', 'V203', 'V274', 'V275', 'V307', 'V308', 'V314', 'V331',
    'id_02', 'id_20',
    'TransactionAmt',
    'ProductCD',
    'card3', 'card4',
    'addr2',
    'dist1', 'dist2',
    'R_emaildomain',
    'C1', 'C2', 'C3', 'C7', 'C9', 'C10', 'C14',
    'D1', 'D3', 'D4', 'D8', 'D11', 'D13', 'D14',
    'M4', 'M5',
    'V5', 'V10', 'V29', 'V31', 'V33', 'V35', 'V42', 'V48', 'V49', 'V53', 'V60', 'V84', 'V97', 'V98',
    'V101', 'V126', 'V129', 'V134', 'V143', 'V144', 'V145', 'V150', 'V164', 'V166', 'V167', 'V172', 'V173',
    'V174', 'V175', 'V176', 'V178', 'V180', 'V187',
    'V202',
    'V204',
    'V206',
    'V218',
    'V220',
    'V223',
    'V228',
    'V231',
    'V240',
    'V245',
    'V246',
    'V253',
    'V264',
    'V267',
    'V268',
    'V270',
    'V271',
    'V277',
    'V279',
    'V284',
    'V286',
    'V291',
    'V296',
    'V298',
    'V306',
    'V310',
    'V311',
    'V313',
    'V315',
    'V316',
    'V317',
    'V321',
    'V324',
    'V327',
    'V328',
    'V330',
    'V334',
    'V335',
    'V336',
    'id_06',
    'id_11',
    'id_19',
    'id_32',
    'id_33',
    'id_37',
    'id_07_null',
    'id_08_null',
    'id_21_null',
    'id_22_null',
    'id_23_null',
    'id_26_null',
    'id_27_null',
    'D2_null', 'D10_null', 'D12_null', 'D13_null'
]

# Load data
target = 'isFraud'
train = pd.read_csv("../input/data-processing/train_filled.csv", usecols=columns + [target])
validation = pd.read_csv("../input/data-processing/validation_filled.csv", usecols=columns + [target])


def get_smoothing_encoding_mapping(series, target, alpha=100, global_mean=None):
    df = pd.concat([series, target], axis=1)
    if global_mean is None:
        global_mean = target.mean()
    means = df.groupby(series.name)[target.name].mean()
    counts = df[series.name].value_counts().sort_index()
    return(means * counts + global_mean * alpha) / (counts + alpha)

global_mean = train[target].mean()
for c in train.columns:
    if train[c].dtype == 'object':
        c_encode = get_smoothing_encoding_mapping(train[c], train[target], alpha=10, global_mean=global_mean)
        train[c] = train[c].map(c_encode)
        validation[c] = validation[c].map(c_encode).fillna(global_mean)

X_train = train.drop(target, axis=1)
y_train = train[target]
X_test = validation.drop(target, axis=1)
y_test = validation[target]

params = {
    'boosting_type': 'gbdt', 'min_child_weight': 1000,
    'colsample_bytree': 1.0, 'max_depth': 10, 'learning_rate': 0.1,
    'n_estimators': 1500, 'reg_lambda': 1.0, 'num_leaves': 100,
    'subsample': 1.0
}


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={
    'metric': 'auc',
    'num_leaves': sp_randint(6, 50), 
    'min_child_samples': sp_randint(100, 500), 
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': sp_uniform(loc=0.2, scale=0.8), 
    'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# n_estimators is set to a "large value"
# The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(
    max_depth=-1, 
    random_state=314, 
    silent=True, 
    metric='None', 
    n_jobs=4, 
    n_estimators=5000
)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=3,  # 100,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True
)

fit_params={
    "early_stopping_rounds":30, 
    "eval_metric" : 'auc', 
    "eval_set" : [(X_test, y_test)],
    'eval_names': ['valid'],
    #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
    'verbose': 100,
    'categorical_feature': 'auto'
}
gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

params = gs.best_params_a

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)
print(metrics.roc_auc_score(y_train, model.predict(X_train)))
print(metrics.roc_auc_score(y_test, model.predict(X_test)))