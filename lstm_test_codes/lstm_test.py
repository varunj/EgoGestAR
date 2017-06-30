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
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import time
import scipy as sp
import scipy.interpolate
from sklearn.metrics import confusion_matrix

CLASSES = ('up','down','left','right','star','del','square','carret','tick','circlecc')
NOS_CLASSES = len(CLASSES)

model = load_model('my_model6.h5')
model.load_weights('my_model_weights6.h5')

dataSeq = []
targetSeq = []
for fileName in glob.glob("../test1_resampled_200/*.txt"):
	file = pd.read_csv(fileName, delim_whitespace=True, header=None)
	fname = fileName.split('_')[-2]
	arr = np.array(file.ix[:, :])
	arr1 = np.transpose(arr)
	targetarr = np.zeros(NOS_CLASSES)
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==fname):
			targetarr[i] = 1
	arrstack = arr1.reshape(1,2,200)
	dataSeq.append(arrstack) 
	targetSeq.append(targetarr)

print(len(dataSeq))
data = np.vstack(dataSeq)
target = np.vstack(targetSeq)

c = 0
truePositive = 0
predArr = []
trueArr = []
startTime = time.clock()
for eachData in data:
	eachData = eachData.reshape(1, 2, 200)
	result = model.predict(eachData)

	predArr.append(np.argmax(result[0]))
	trueArr.append(np.argmax(target[c]))
	if (np.argmax(target[c]) == np.argmax(result[0])):
		truePositive = truePositive+1

	# print('t: ' + CLASSES[np.argmax(target[c])] + ' p: ' + CLASSES[np.argmax(result[0])] + ' with prob: ' + str(max(result[0])))
	c = c+1

print(CLASSES)
print(confusion_matrix(trueArr, predArr))
print('acc: ' + str(truePositive) + '/500')

print('time taken: ' + str(time.clock() - startTime))