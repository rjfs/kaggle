# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import clean_data
import os
import sys


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--clean', is_flag=True)
def cmd_main(input_filepath, output_filepath, clean=False):
    """ Runs main function with arguments taken from command line """
    main(input_filepath, output_filepath, clean=clean)


def run():
    """ Runs main with hard coded arguments """
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    mod_path = '/'.join(script_path.split('/')[:-2]) + '/'
    input_filepath = mod_path + 'data/raw/'
    output_filepath = mod_path + 'data/processed/'
    # Generate processed dataset
    main(input_filepath, output_filepath, clean=False)
    # Generate cleaned text processed dataset
    main(input_filepath, output_filepath, clean=True)


def main(input_filepath, output_filepath, clean=False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    raw_train = pd.read_csv(input_filepath + 'train.csv', index_col='id')
    raw_test = pd.read_csv(input_filepath + 'test.csv', index_col='id')
    # Output file names
    train_fname = 'train-train.csv'
    val_fname = 'train-val.csv'
    test_fname = 'test.csv'
    if clean:
        data_dir = '/'.join(input_filepath.split('/')[:-2]) + '/'
        mapping_file = data_dir + 'external/conv.csv'
        cleaner = clean_data.DataCleaner(mapping_file)
        train_val = cleaner.clean(raw_train)
        test = cleaner.clean(raw_test)
        train_fname = 'clean-' + train_fname
        val_fname = 'clean-' + val_fname
        test_fname = 'clean-' + test_fname
    else:
        train_val = raw_train
        test = raw_test
    # Split train data in train and validation set
    train, val = train_validation_split(train_val)
    # Save to output files
    logger.info('saving output')
    train.to_csv(output_filepath + train_fname)
    val.to_csv(output_filepath + val_fname)
    test.to_csv(output_filepath + test_fname)
    logger.info('done')


def train_validation_split(data, train_pct=0.8):
    """
    Split data into train and validation sets
    :param data: pandas.DataFrame
        Whole data
    :param train_pct: float, default 0.8
        Training data percentage
    :return: tuple of pandas.DataFrame
        (training_set, validation_set)
    """
    np.random.seed(28)
    shuffled = data.iloc[np.random.permutation(len(data))]
    split_point = int(np.ceil(len(shuffled) * train_pct))
    train = shuffled.iloc[:split_point].sort_index()
    val = shuffled.iloc[split_point:].sort_index()
    return train, val


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    run()
