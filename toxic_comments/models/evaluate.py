import click
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

out_categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@click.command()
@click.argument('pred_path', type=click.Path(exists=True))
@click.argument('true_path', type=click.Path(exists=True))
def main(pred_path, true_path):
    predictions = pd.read_csv(pred_path, index_col='id')
    trues = pd.read_csv(true_path, index_col='id')
    show_misclassified(predictions, trues)


def eval_validation(pred_path, true_path):
    evaluate_predictions(pred_path, true_path)


def evaluate_predictions(pred_path, true_path):
    predictions = pd.read_csv(pred_path, index_col='id')
    trues = pd.read_csv(true_path, index_col='id')
    # Get scores dictionary
    scores = get_scores_dict(predictions=predictions, trues=trues)
    # Print results
    print(scores)
    print('Final Score: %.4f' % np.mean([i for i in scores.values()]))


def get_scores_dict(predictions, trues):

    scores = {}
    for c in predictions.columns:
        y_true = trues[c].values
        y_score = predictions[c].values
        score = roc_auc_score(y_true, y_score)
        scores[c] = score

    return scores


def show_misclassified(predictions, trues, threshold=0.5):
    m = len(predictions.iloc[0])
    preds_mat = predictions[out_categories].values
    trues_mat = trues[out_categories].values
    for idx, i in enumerate(predictions.index):
        mse = sum((preds_mat[idx] - trues_mat[idx]) ** 2) / m
        if mse >= threshold:
            print('MSE: %.3f' % mse)
            print('Predictions:')
            print(predictions.loc[i, out_categories])
            print('True Values:')
            print(trues.loc[i, out_categories])
            print('------')
            print(trues.loc[i, 'comment_text'])
            print('===================================================')


if __name__ == '__main__':
    main()
