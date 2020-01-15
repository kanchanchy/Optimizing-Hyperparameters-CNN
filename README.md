# Optimizing Hyperparameters of CNN

Performance of a multi-layer neural network always depends on hyper-parameters such as learning rate, mini batch size, dropout rate, starting learning rate, and learning rate etc. Optimizing hyper-parameters of a multi-layer neural network is always a challenging task. This repository implements two ways of optimizing hyper-parameters of a convolutional neural network and compares their performances: 1) Grid Search and 2) Bayesian Optimization.

The implementation is done using pytorch framework. grid_search.py file implements grid search based approach while bayesian_search.py file implements bayesian optimization based approach. train_test_cnn.py implements training and testing of a CNN based on some static hyper-parameters.
