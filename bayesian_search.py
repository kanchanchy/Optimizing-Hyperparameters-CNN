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
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.dataset import CVSplit
import time
from functools import partial
from bayes_opt import BayesianOptimization
from utils import *


def fit_with(dropout1_rate, dropout2_rate, learning_rate, train_data_loader, val_data_loader):
	# Create the model using a specified hyperparameters.
	model = ConvNet(dropout1_rate, dropout2_rate)
	if torch.cuda.is_available():
		model = model.cuda()
	num_epochs = 15

	# Train the model
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	criterion = nn.CrossEntropyLoss()
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_data_loader):
			outputs = model(images)
			loss = criterion(outputs, labels)

			# Backpropagation and Adam optimisation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	#Evaluation with validation data
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in val_data_loader:
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	score = correct/total
	print('Validation accuracy:', score)
	return score



if __name__== "__main__":
	mini_batch_size = 100
		
	train_data_path = 'data/train_32x32.mat'
	test_data_path = 'data/test_32x32.mat'
	trainX, trainY, valX, valY, testX, testY = load_dataset(train_data_path, test_data_path)

	#converting to grayscale images
	trainX = rgb2gray(trainX).astype(np.float32)
	valX = rgb2gray(valX).astype(np.float32)
	testX = rgb2gray(testX).astype(np.float32)

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

	# changing the labels of 10 to 0
	trainY[trainY == 10] = 0
	valY[valY == 10] = 0
	testY[testY == 10] = 0

	#converting numpy to tensor
	tensor_trainX = torch.from_numpy(trainX)
	tensor_trainY = torch.from_numpy(trainY).long()
	tensor_valX = torch.from_numpy(valX)
	tensor_valY = torch.from_numpy(valY).long()
	tensor_testX = torch.from_numpy(testX)
	tensor_testY = torch.from_numpy(testY).long()

	#running with GPU
	if torch.cuda.is_available():
		tensor_trainX = tensor_trainX.cuda()
		tensor_trainY = tensor_trainY.cuda()
		tensor_valX = tensor_valX.cuda()
		tensor_valY = tensor_valY.cuda()
		tensor_testX = tensor_testX.cuda()
		tensor_testY = tensor_testY.cuda()

	# creating datset for dataloader
	train_dataset = torchUtils.TensorDataset(tensor_trainX, tensor_trainY)
	val_dataset = torchUtils.TensorDataset(tensor_valX, tensor_valY)
	test_dataset = torchUtils.TensorDataset(tensor_testX, tensor_testY)

	# Data loader
	train_loader = torchUtils.DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
	val_loader = torchUtils.DataLoader(dataset=val_dataset, batch_size=mini_batch_size, shuffle=False)

	#Bayesian Optimization
	start_time = time.time()

	fit_with_partial = partial(fit_with, train_data_loader = train_loader, val_data_loader = val_loader)

	# Bounded region of parameter space
	pbounds = {'dropout1_rate': (0.0, 0.5), 'dropout2_rate': (0.0, 0.5), 'learning_rate': (0.001, 0.01)}

	optimizer = BayesianOptimization(
		f=fit_with_partial,
		pbounds=pbounds,
		random_state=1,
		)

	optimizer.maximize(init_points=10, n_iter=10,)

	for i, res in enumerate(optimizer.res):
		print("Iteration {}: \n\t{}".format(i, res))

	print(optimizer.max)
	end_time = time.time()
	print("Required Time: " + str(end_time - start_time) + " s")


