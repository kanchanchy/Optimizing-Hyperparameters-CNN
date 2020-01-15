# Optimizing Hyperparameters of CNN

Performance of a multi-layer neural network always depends on hyper-parameters such as learning rate, mini batch size, dropout rate, starting learning rate, and learning rate etc. Optimizing hyper-parameters of a multi-layer neural network is always a challenging task. This repository implements two ways of optimizing hyper-parameters of a convolutional neural network and compares their performances: 1) Grid Search and 2) Bayesian Optimization. This repository optimizes three particular hyper-parameters: learning rate, dropout for first fully connected layer and dropout for second fully connected layer.

The implementation is done using pytorch framework. grid_search.py file implements grid search based approach while bayesian_search.py file implements bayesian optimization based approach. train_test_cnn.py implements training and testing of a CNN based on some static hyper-parameters.

# How to run:
Download SVHN dataset (both train and test) from http://ufldl.stanford.edu/housenumbers/ and place both dataset inside data/ directory. After that run grid_search.py, bayesian_search.py, and train_test_cnn.py separately. Note that running grid_search.py might take a very long time because it trains the model for all hyper-parameter combinations.
