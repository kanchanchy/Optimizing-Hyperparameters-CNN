import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import torch.nn.functional as F
import torch.utils.data as torchUtils
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.dataset import CVSplit
import time
import multiprocessing
from utils import *


if __name__== "__main__":

	train_data_path = 'data/train_32x32.mat'
	test_data_path = 'data/test_32x32.mat'
	trainX, trainY, valX, valY, testX, testY = load_dataset(train_data_path, test_data_path)

	#converting to grayscale images
	trainX = rgb2gray(trainX).astype(np.float32)
	valX = rgb2gray(valX).astype(np.float32)
	testX = rgb2gray(testX).astype(np.float32)

	trainY = trainY.astype(int)
	valY = valY.astype(int)
	testY = testY.astype(int)

	# Calculating the mean on the training data
	train_mean = np.mean(trainX, axis=0)
	# Calculating the std on the training data
	train_std = np.std(trainX, axis=0)

	# normalization of the data
	trainX = (trainX - train_mean) / train_std
	valX = (valX - train_mean)  / train_std
	testX = (testX - train_mean) / train_std

	#reshaping channel number
	trainX = trainX.transpose(0, 3, 1, 2)
	valX = valX.transpose(0, 3, 1, 2)
	testX = testX.transpose(0, 3, 1, 2)

	trainY = trainY.transpose()[0]
	valY = valY.transpose()[0]
	testY = testY.transpose()[0]

	trainY[trainY == 10] = 0
	valY[valY == 10] = 0
	testY[testY == 10] = 0

	params = {
	'lr': [0.001, 0.01, 0.05],
    'module__dropout1': [0, 0.2, 0.4],
    'module__dropout2': [0, 0.3, 0.5],
	}

	net = NeuralNetClassifier(
		ConvNet,
		max_epochs=15,
		batch_size = 100,
		iterator_train__shuffle=True,
		verbose=False,
		train_split=None,
		)

	number_jobs = multiprocessing.cpu_count()-1
	gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', n_jobs = number_jobs)

	start_time = time.time()
	gs.fit(trainX, trainY)
	end_time = time.time()

	# print results
	print("Required Time: " + str(end_time - start_time) + " ms")
	print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
	means = gs.cv_results_['mean_test_score']
	stds = gs.cv_results_['std_test_score']
	params = gs.cv_results_['params']


	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

	print("\n\n")
	print("means scores:")
	print(means)
	print("Standard Deviations")
	print(stds)
	print("Parameters:")
	print(params)
	
