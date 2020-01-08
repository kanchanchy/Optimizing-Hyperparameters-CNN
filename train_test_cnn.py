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

# Hyperparameters
learning_rate = 0.001
dropout1 = 0.36
dropout2 = 0.36
num_epochs = 15
mini_batch_size = 100


if __name__== "__main__":
		
	train_data_path = 'train_32x32.mat'
	test_data_path = 'test_32x32.mat'
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
	test_loader = torchUtils.DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=False)

	model = ConvNet(dropout1, dropout2)
	if torch.cuda.is_available():
		model = model.cuda()

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

	# Train the model
	total_step = len(train_loader)
	training_loss_list = []
	val_loss_list = []
	training_acc_list = []
	val_acc_list = []
	epoc_list = []

	for epoch in range(num_epochs):
		training_loss = 0
		total_train_data = 0
		training_correct = 0
		train_count = 0
		
		for i, (images, labels) in enumerate(train_loader):
			#Run the forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)
			training_loss += loss
			train_count += 1

			# Backprop and perform Adam optimisation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Track the accuracy
			total = labels.size(0)
			_, predicted = torch.max(outputs.data, 1)
			correct = (predicted == labels).sum().item()
			
			total_train_data += total
			training_correct += correct

			if (i + 1) % 100 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
					.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

		training_loss_list.append(training_loss/train_count)
		training_acc_list.append(training_correct/total_train_data)

		#validation
		val_loss = 0
		correct = 0
		total = 0
		for images, labels in val_loader:
			outputs = model(images)
			val_loss += criterion(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		val_loss_list.append(val_loss/len(val_loader))
		val_acc_list.append(correct/total)

		epoc_list.append(epoch + 1)

	#Evaluation with test data
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		print('Test Accuracy of the model on the 2000 test images: {} %'.format((correct / total) * 100))


	#Plot results

	plt.plot(epoc_list, training_loss_list, lw=1, color = "#FF0000", label='Training Loss')
	plt.plot(epoc_list, val_loss_list, lw=1, color = "#008000", label='Validation Loss')
	plt.title("Loss vs Epoch Numbers")
	plt.show()

	plt.plot(epoc_list, training_acc_list, lw=1, color = "#FF0000", label='Training Accuracy')
	plt.plot(epoc_list, val_acc_list, lw=1, color = "#008000", label='Validation Accuracy')
	plt.title("Accuracy vs Epoch Numbers")
	plt.show()

