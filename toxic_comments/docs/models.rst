=================
Model Explanation
=================

This document briefly explains the implemented models and the final submitted models
ensemble.

======
Models
======

The three main models are:

#. Random Forest
    Random Forest model applied to features extracted from comments: capital letters percentage, unique words percentage, punctuation features, etc.

#. Naive Bayes Logistic Regression using Bigrams
    This model is implemented using a pipeline with the following elements:

    1) TF-IDF vectorizer using n-grams with n = 1 and n = 2 (single words and bigrams).
    2) Computation of Naive Bayes log-count ratios.
    3) Apply logistic regression to the features obtained in 2).

    The model is based on this `notebook <https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/>`_ and this `paper <https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf/>`_.

#. Convolutional Neural Network with Char n-gram
    Character n-grams are used instead of word-tokens to deal with the problem of misspelled words. Because of the high dimensionality of the data introduced by char n-gram, a Convolutional Neural Network is used in order to have less parameters to train. The design of the network is based on `this notebook <https://www.kaggle.com/sbongo/for-beginners-go-even-deeper-with-char-gram-cnn/>`_ and includes a Bidirectional GRU (Gated Recurrent Unit) besides the Convolutional and Pooling Layers.

========
Ensemble
========

The models ensemble was done using an output category weighted average, i.e., for each output label, a weighted average in the models outputs was performed based on the performance of the model in that same output label.

The three models ensemble (ME) was then combined with this `blend <https://www.kaggle.com/antmarakis/another-cleaned-data-blend-with-low-correlation/code/>`_ (BL) in the following way: OUTPUT = 0.6 * ME + 0.4 * BL, with the goal of obtaining an ensemble with more different models.

