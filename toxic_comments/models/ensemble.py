import click
import evaluate
import pandas as pd
import numpy as np
import utils
import os

mix_file = 'outputs/preprocessed_blend.csv'


@click.command()
@click.argument('files_dir', type=click.Path(exists=True))
@click.argument('trues_file', type=click.Path(exists=True))
def main(files_dir, trues_file):
    files_names = [
        '20180319-180049-nb-svm-val.out',
        '20180320-142913-char-gram-val.out',
        # '20180320-184149-char-gram-val.out',
        # '20180320-222222-random-forest-val.out'
        # '20180320-230610-random-forest-val.out'
    ]
    ens = Ensemble(files_names, files_dir, trues_file)
    ens.initialize()
    print('=== ENSEMBLE ===')
    ens.average_score()
    ens.weighted_average_score()
    ens.category_weighted_average_score()
    ens.best_in_category()

    ens.save_files('category_weighted_average')


class Ensemble:

    def __init__(self, files_names, files_dir, trues_file):
        self.files_names = files_names
        self.files_dir = files_dir
        self.trues_file = trues_file
        self.trues = None
        self.models = []
        self.output_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def initialize(self):
        self.trues = pd.read_csv(self.trues_file, index_col='id')
        self.models = [Model(self.files_dir + f) for f in self.files_names]
        for m in self.models:
            m.load()
            m.get_scores(self.trues)
            print('=========')
            print(m.fpath)
            print(m.scores)
            print(m.mean_score)

    def save_files(self, method):
        if method == 'category_weighted_average':
            val_preds = {m.fpath: m.predictions for m in self.models}
            val = self.category_weighted_average(val_preds)
            file_paths = [self.files_dir + f for f in self.files_names]
            test_preds = {
                fp: read_file(fp.replace('-val.out', '-test.out'))
                for fp in file_paths
            }
            test = self.category_weighted_average(test_preds)
            blend = pd.read_csv(mix_file, index_col='id')
            test = 0.6 * test + 0.4 * blend
        else:
            raise NotImplementedError('Not implemented for \'%s\'' % method)

        utils.save_predictions(
            os.path.dirname(self.trues_file) + '/',
            train_preds=None, val_preds=val, test_preds=test,
            model='ensemble'
        )

    def average_score(self):
        """ Prints average ensemble score """
        avg_preds = sum([m.predictions for m in self.models]) / len(self.models)
        self.print_global_score(avg_preds)

    def category_weighted_average_score(self):
        """ Prints weighted average, by category, ensemble score """
        predictions = {m.fpath: m.predictions for m in self.models}
        preds = self.category_weighted_average(predictions)
        self.print_global_score(preds)

    def category_weighted_average(self, predictions):
        n_m = len(self.models)
        preds_lst = []
        for c in self.output_classes:
            weights = softmax_weights([(m.fpath, m.scores[c]) for m in self.models])
            s_preds = sum([p[c] * weights[fpath] for fpath, p in predictions.items()])
            preds_lst.append(s_preds / n_m)

        return pd.concat(preds_lst, axis=1)

    def best_in_category(self):
        preds_lst = []
        for c in self.output_classes:
            c_scores = sorted([(m, m.scores[c]) for m in self.models], key=lambda x: x[1])
            m = c_scores[-1][0]
            preds_lst.append(m.predictions[c])

        preds = pd.concat(preds_lst, axis=1)
        self.print_global_score(preds)

    def weighted_average_score(self):
        """ Prints weighted average ensemble score """
        weights = softmax_weights([(m.fpath, m.mean_score) for m in self.models])
        sum_preds = sum([m.predictions * weights[m.fpath] for m in self.models])
        preds = sum_preds / len(self.models)
        self.print_global_score(preds)

    def print_global_score(self, preds):
        scores = evaluate.get_scores_dict(preds, self.trues)
        print('> %.4f' % np.mean(list(scores.values())))


def softmax_weights(scores):
    scores_vals = [i[1] for i in scores]
    rng = max(scores_vals) - min(scores_vals)
    rng = max(rng, 0.002)  # to avoid overflows
    d = {f: np.exp(s/rng) for f, s in scores}
    sum_vals = sum(list(d.values()))
    return {f: v/sum_vals for f, v in d.items()}


def read_file(fpath):
    return pd.read_csv(fpath, index_col='id')


class Model:

    def __init__(self, fpath):
        self.fpath = fpath
        self.predictions = None
        self.scores = {}

    @property
    def mean_score(self):
        return np.mean(list(self.scores.values()))

    def load(self):
        self.predictions = pd.read_csv(self.fpath, index_col='id')

    def get_scores(self, trues):
        self.scores = evaluate.get_scores_dict(self.predictions, trues)


if __name__ == '__main__':
    main()
