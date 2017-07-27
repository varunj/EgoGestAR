import matplotlib as mpl
mpl.use('Agg')
import numpy as np
np.random.seed(123)
import glob, os
import pandas as pd
from scipy.spatial.distance import euclidean
from keras.layers.core import Dropout, Activation, Flatten
import matplotlib.pyplot as plt
from math import sqrt
from pprint import pprint
import pdb
np.set_printoptions(threshold=np.nan)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D
from keras.layers import LSTM
from keras.utils import plot_model
import random
import cv2
from keras.utils import np_utils

'''
https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-7
'''
CLASSES = ('up','down','left','right','star','del','square','carret','tick','circlecc')
NOS_CLASSES = len(CLASSES)
PATH_TRAIN = '../train_images/'
PATH_TEST = '../test_images/'
NOS_INP_TRAIN = 2000
NOS_INP_TEST = 500
EPOCHS_NO = 3
BATCH_SIZE = 8

# make list of all files to be read. then shuffle it
fileNamesArr = []
for fileName in glob.glob(PATH_TRAIN + "*.png"):
	fileNamesArr.append(os.path.basename(fileName))
fileNamesArr = random.sample(fileNamesArr, len(fileNamesArr))

# read images to imageStacks. make nos channels second dimen. scale input to 0-1
c = 0
trainInpStack = np.zeros((NOS_INP_TRAIN, 1, 28, 28))
trainGrndStack = np.zeros((NOS_INP_TRAIN))
for eachImgName in fileNamesArr[:NOS_INP_TRAIN]:
	trainInpPath = PATH_TRAIN + eachImgName
	trainInpStack[c] = cv2.resize(cv2.imread(trainInpPath,0), (28, 28)).reshape(1, 28, 28)
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==trainInpPath.split('_')[2]):
			trainGrndStack[c] = i
	c = c+1
trainGrndStack = np_utils.to_categorical(trainGrndStack, 10)


# make list of all files to be read. then shuffle it
fileNamesArr = []
for fileName in glob.glob(PATH_TEST + "*.png"):
	fileNamesArr.append(os.path.basename(fileName))
fileNamesArr = random.sample(fileNamesArr, len(fileNamesArr))

# read images to imageStacks. make nos channels second dimen. scale input to 0-1
c = 0
testInpStack = np.zeros((NOS_INP_TEST, 1, 28, 28))
testGrndStack = np.zeros((NOS_INP_TEST))
for eachImgName in fileNamesArr[:NOS_INP_TEST]:
	testInpPath = PATH_TEST + eachImgName
	testInpStack[c] = cv2.resize(cv2.imread(testInpPath,0), (28, 28)).reshape(1, 28, 28)
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==testInpPath.split('_')[2]):
			testGrndStack[c] = i
	c = c+1
testGrndStack = np_utils.to_categorical(testGrndStack, 10)

# save images with class names
# c = 0
# for x, y in zip(trainInpStack, trainGrndStack):
# 	c = c + 1
# 	print('./zzz/' + CLASSES[int(y)] + '_' + str(c) + '_' + '.png')
# 	cv2.imwrite('./zzz/' + CLASSES[int(y)] + '_' + str(c) + '_' + '.png', x.reshape(480,640,1))

print(trainInpStack.shape , trainGrndStack.shape)
print(testInpStack.shape , testGrndStack.shape)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(trainInpStack, trainGrndStack, batch_size=32, nb_epoch=30, verbose=1)

model.save('my_model.h5')
model.save_weights('my_model_weights.h5')

plt.plot(history.history['acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('accuracy.png')
plt.close()
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('loss.png')

score = model.evaluate(testInpStack, testGrndStack, verbose=0)
print(score)