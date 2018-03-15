import click
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


@click.command()
@click.argument('pred_path', type=click.Path(exists=True))
@click.argument('true_path', type=click.Path(exists=True))
def main(pred_path, true_path):
    eval_validation(pred_path, true_path)


def eval_validation(pred_path, true_path):
    preds = pd.read_csv(pred_path, index_col='id')
    trues = pd.read_csv(true_path, index_col='id')
    scores = {}
    for c in preds.columns:
        y_true = trues[c].values
        y_score = preds[c].values
        score = roc_auc_score(y_true, y_score)
        scores[c] = score
        print('ROC AUC for %s: %.4f' % (c, score))

    print(scores)
    print('Final Score: %.4f' % np.mean([i for i in scores.values()]))


if __name__ == '__main__':
    main()
