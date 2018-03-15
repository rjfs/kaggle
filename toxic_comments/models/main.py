import click
import chargram_cnn
import load_data
import evaluate
import time
import nb_svm
import logging

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('model')
def main(data_path, model):
    if model == 'char-gram':
        clean_data = True
        train, val, test = load_data.load_train_val_test(data_path, clean=clean_data)
        mod = chargram_cnn.CharGramCNN(epochs=1, sentences_maxlen=50)
    elif model == 'nb-svm':
        clean_data = False
        train, val, test = load_data.load_train_val_test(data_path, clean=clean_data)
        mod = nb_svm.NaiveBayesSVM()
    else:
        raise Exception('Unknown model: \'%s\'' % model)

    # Fit model
    logger.info('Fitting model...')
    mod.fit(train, val)
    # Generate predictions df
    logger.info('Generating validation predictions...')
    val_preds = mod.predictions_df(val['comment_text'])
    logger.info('Generating test predictions...')
    test_preds = mod.predictions_df(test['comment_text'])
    # Save predictions to files
    logger.info('Saving to files')
    val_path = load_data.get_validation_path(data_path, clean=clean_data)
    save_predictions(
        val_preds, test_preds, model, val_path=val_path
    )


def save_predictions(validation, test, model, val_path):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    val_fname = '%s-%s-val.out' % (timestr, model)
    test_fname = '%s-%s-test.out' % (timestr, model)
    save_df(validation, fname=val_fname)
    # Evaluate validation set
    evaluate.eval_validation(pred_path=val_fname, true_path=val_path)
    # Generate test set predictions
    save_df(test, fname=test_fname)


def save_df(df, fname):
    df.to_csv(fname)
    print('Saved to: %s' % fname)


if __name__ == '__main__':
    main()
