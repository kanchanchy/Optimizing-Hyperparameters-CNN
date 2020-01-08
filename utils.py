import scipy.io
import numpy as np

#function for loading train and test data
def load_dataset(train_data_path, test_datapath):
	#loading dataset
	trainData= scipy.io.loadmat(train_data_path)
	testData= scipy.io.loadmat(test_datapath)
	trainVec = trainData['X']
	trainLabel = trainData['y']
	testVec = testData['X']
	testLabel = testData['y']

	train_size = 10000
	val_size = 500
	train_total = {}
	val_total = {}
	trainX = []
	trainY = []
	valX = []
	valY = []

	i = 0
	while len(trainX) < train_size or len(valX) < val_size:
		label = trainLabel[i][0]
		trainCount = 0
		valCount = 0
		if label in train_total:
			trainCount = train_total[label]
		if label in val_total:
			valCount = val_total[label]

		if trainCount < 1000:
			trainX.append(trainVec[:,:,:,i])
			trainY.append(trainLabel[i])
			trainCount += 1
			train_total[label] = trainCount
		elif valCount < 50:
			valX.append(trainVec[:,:,:,i])
			valY.append(trainLabel[i])
			valCount += 1
			val_total[label] = valCount
		i += 1

	test_size = 2000
	test_total = {}
	testX = []
	testY = []

	i = 0
	while len(testX) < test_size:
		label = testLabel[i][0]
		testCount = 0
		if label in test_total:
			testCount = test_total[label]

		if testCount < 200:
			testX.append(testVec[:,:,:,i])
			testY.append(testLabel[i])
			testCount += 1
			test_total[label] = testCount
		i += 1

	trainX = np.array(trainX)
	trainY = np.array(trainY)
	valX = np.array(valX)
	valY = np.array(valY)
	testX = np.array(testX)
	testY = np.array(testY)

	return trainX, trainY, valX, valY, testX, testY


#function for converting RGB image to grayscale image
def rgb2gray(images):
	return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, dropout1 = 0.1, dropout2 = 0.1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.drop_out1 = nn.Dropout(p = dropout1)
        self.drop_out2 = nn.Dropout(p = dropout2)
        self.fc1 = nn.Linear(24 * 24 * 128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
    	out = F.relu(self.conv1(x))
    	out = F.relu(self.conv2(out))
    	out = F.relu(self.conv3(out))
    	out = out.reshape(out.size(0), -1)
    	out = self.drop_out1(F.relu(self.fc1(out)))
    	out = self.drop_out2(F.relu(self.fc2(out)))
    	out = F.softmax(self.fc3(out))
    	return out



