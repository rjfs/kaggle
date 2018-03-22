import evaluate
import time


def save_predictions(data_path, train_preds, val_preds, test_preds, model):
    val_path = data_path + 'train-val.csv'
    train_path = data_path + 'train-train.csv'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    train_fname = '%s-%s-train.out' % (timestr, model)
    val_fname = '%s-%s-val.out' % (timestr, model)
    test_fname = '%s-%s-test.out' % (timestr, model)
    # Save to files
    if train_preds is not None:
        save_df(train_preds, fname=train_fname)
    save_df(val_preds, fname=val_fname)
    save_df(test_preds, fname=test_fname)
    # Evaluate
    if train_preds is not None:
        print('Evaluating train...')
        evaluate.evaluate_predictions(pred_path=train_fname, true_path=train_path)
    print('Evaluating validation...')
    evaluate.evaluate_predictions(pred_path=val_fname, true_path=val_path)


def save_df(df, fname):
    df.to_csv(fname)
    print('Saved to: %s' % fname)
