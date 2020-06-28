Future Sales Predictions
==============================

Predict total sales for every product and store.

Requirements
------------
You should run this code using Python 3.

The needed packages and corresponding versions are in *requirements.txt* file.
I recommend to use this file with *pip* or *conda* to install the packages in a separate environment.

Run Project
-----------
To generate the final predictions, the following steps should be executed in the following order.

1. Place raw data in 'data/raw' folder and extract the zip files.
2. Run notebook 'notebooks/text_eda.ipynb' to generate text features.
3. Run notebook 'notebooks/features.ipynb' to generate more raw features.
4. Run models predictions using 'src/models/main.py'.
    
    Beyond other functionalities, the following commands are mandatory to generate files used by ensemble model:
    
        python src/models/main.py --model dart --task predict_months
        python src/models/main.py --model gbdt-lgb --task predict_months
5. Run notebook 'notebooks/ensemble.ipynb' to generate final submission file.
        

Project Organization
-------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
