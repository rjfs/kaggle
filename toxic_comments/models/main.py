import click
import chargram_cnn
import utils
import nb_svm
import logging
import random_forest
import time
import evaluate

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('model')
def main(data_path, model):
    if model == 'char-gram':
        mod = chargram_cnn.CharGramCNN()
    elif model == 'nb-svm':
        mod = nb_svm.NaiveBayesSVM()
    elif model == 'random-forest':
        mod = random_forest.ToxicRandomForest()
    else:
        raise Exception('Unknown model: \'%s\'' % model)

    # Load data
    logger.info('Loading data...')
    mod.load_data(data_path)
    # Fit model
    logger.info('Fitting model...')
    mod.fit()
    # Generate predictions df
    logger.info('Generating training predictions...')
    train_preds = mod.train_predictions()
    logger.info('Generating validation predictions...')
    val_preds = mod.validation_predictions()
    logger.info('Generating test predictions...')
    test_preds = mod.test_predictions()
    # Save predictions to files
    logger.info('Saving to files...')
    utils.save_predictions(data_path, train_preds, val_preds, test_preds, model)


if __name__ == '__main__':
    main()
