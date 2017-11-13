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
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import time
import scipy as sp
import scipy.interpolate
# from sklearn.metrics import confusion_matrix
import pdb
from shutil import copyfile

CLASSES1 = ('up','down','left','right','star','del','square','carret','tick','circlecc')
NOS_CLASSES1 = len(CLASSES1)

# model = load_model('20avMy_model.hdf5')
# model = load_model('20concatMy_model.hdf5')
model = load_model('30testdrop2testMy_model.hdf5')
# model = load_model('20sumMy_model.hdf5')

print(model.summary())
# pdb.set_trace()
dataSeq = []
targetSeq = []
filenameArr = []
for fileName in glob.glob("../test_resampled_200/*.txt"):
	file = pd.read_csv(fileName, delim_whitespace=True, header=None)
	fname = fileName.split('_')[-2]
	filenameArr.append(fileName)
	arr = np.array(file.ix[:, :])
	#arr1 = np.transpose(arr)
	targetarr = np.zeros(NOS_CLASSES1)
	for i in range(0, NOS_CLASSES1):
		if (CLASSES1[i]==fname):
			targetarr[i] = 1
	arrstack = arr.reshape(1,200,2)
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
	eachData = eachData.reshape(1, 200, 2)
	result = model.predict(eachData)
	#pdb.set_trace()
	predArr.append(np.argmax(result[0]))
	trueArr.append(np.argmax(target[c]))
	if (np.argmax(target[c]) == np.argmax(result[0])):
		truePositive = truePositive+1

	# print('t: ' + CLASSES1[np.argmax(target[c])] + ' p: ' + CLASSES1[np.argmax(result[0])] + ' with prob: ' + str(max(result[0])))
	c = c+1

print(CLASSES1)
# print(confusion_matrix(trueArr, predArr))
print('acc: ' + str(truePositive) + '/500')

print('time taken: ' + str(time.clock() - startTime))


# copy source files of misclassified gestures
c = 0
for x in range(0, len(predArr)):
	c = c + 1
	if (CLASSES1[predArr[x]] != CLASSES1[np.argmax(target[x])]):
		copyfile(filenameArr[x], './misclassified/' + str(c) + '_' + CLASSES1[predArr[x]] + '.txt')
		print(CLASSES1[predArr[x]], filenameArr[x])