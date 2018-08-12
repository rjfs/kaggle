import logging
import core
import numpy as np
from core import MONTH_INT_LABEL, TARGET_LABEL, INDEX_COLUMNS
import utils


class ModelEvaluation:

    def __init__(self, model_class, n_evals=3):
        self.model_class = model_class
        self.n_evals = n_evals

    def evaluate(self, data):
        max_m = data[MONTH_INT_LABEL].max()
        rmse = {}
        for i in range(max_m - self.n_evals + 1, max_m + 1):
            curr_f = i - max_m + self.n_evals
            logging.info('Fold %d/%d' % (curr_f, self.n_evals))
            # Compute fold data
            fold_train = data[data[MONTH_INT_LABEL] < i]
            # Fit model
            fold_test = data[data[MONTH_INT_LABEL] == i]
            self.model_class.train_model(
                fold_train, validation_df=fold_test,
            )
            # Compute errors in train and validation
            x_t, y_t = core.features_target_split(fold_train)
            fold_test_x, fold_test_y = core.features_target_split(fold_test)
            del fold_train
            # Get RMSE for train and validation
            err_t = self.get_rmse(x_t, y_t)
            del x_t
            err_v = self.get_rmse(fold_test_x, fold_test_y)
            rmse[i] = (err_t, err_v)
            print(rmse[i])

        print('Validation Results:')
        print(rmse)
        train_errs = np.array([i[0] for i in rmse.values()])
        val_errs = np.array([i[1] for i in rmse.values()])
        print('Mean Train: %.5f' % np.mean(train_errs))
        print('Mean Validation: %.5f' % np.mean(val_errs))

        return np.mean(val_errs)

    def get_rmse(self, x, y):
        return utils.compute_score(self.model_class.predict(x), y)

