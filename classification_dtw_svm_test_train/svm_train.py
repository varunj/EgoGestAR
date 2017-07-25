import matplotlib as mpl
mpl.use('Agg')
import numpy as np
np.random.seed(123)
np.set_printoptions(threshold=np.nan)
import glob, os
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time

'''
http://scikit-learn.org/stable/modules/svm.html
'''

CLASSES = ('up','down','left','right','star','del','square','carret','tick','circlecc')
NOS_CLASSES = len(CLASSES)

def shuffle_data(labels, seq):
	temp = 0
	for k in seq:
		if (temp ==0):
			labels2 = labels[k:k+1,:]
			temp =temp+1
		else:
			labels2 = np.vstack((labels2, labels[k:k+1,:]))
			temp = temp+1	
	return labels2

dataSeqTrain = []
targetSeqTrain = []
for fileName in glob.glob("../train_resampled_200/*.txt"):
	file = pd.read_csv(fileName, delim_whitespace=True, header=None)
	fname = fileName.split('_')[-2]
	arr = np.array(file.ix[:, :])
	arr1 = np.transpose(arr)
	targetarr = 0
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==fname):
			targetarr = i
	# arrstack = arr1.reshape(1,2,200)
	arrstack = arr1.reshape(1,400)
	dataSeqTrain.append(arrstack) 
	targetSeqTrain.append(targetarr)
dataTrain = np.vstack(dataSeqTrain)
targetTrain = np.vstack(targetSeqTrain)
seq = np.arange(len(dataSeqTrain))
np.random.shuffle(seq)
dataTrain = shuffle_data(dataTrain, seq)
targetTrain = shuffle_data(targetTrain, seq)
print('train set', dataTrain.shape, targetTrain.shape)

dataSeqTest = []
targetSeqTest = []
for fileName in glob.glob("../test_resampled_200/*.txt"):
	file = pd.read_csv(fileName, delim_whitespace=True, header=None)
	fname = fileName.split('_')[-2]
	arr = np.array(file.ix[:, :])
	arr1 = np.transpose(arr)
	targetarr = 0
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==fname):
			targetarr = i
	# arrstack = arr1.reshape(1,2,200)
	arrstack = arr1.reshape(1,400)
	dataSeqTest.append(arrstack) 
	targetSeqTest.append(targetarr)
dataTest = np.vstack(dataSeqTest)
targetTest = np.vstack(targetSeqTest)
seq = np.arange(len(dataSeqTest))
np.random.shuffle(seq)
dataTest = shuffle_data(dataTest, seq)
targetTest = shuffle_data(targetTest, seq)
print('test set ', dataTest.shape, targetTest.shape)

# train SVM
lin_clf = svm.LinearSVC()
lin_clf.fit(dataTrain, targetTrain) 

t = time.time()

# test SVM
y_true = []
y_pred = []
for x, y in zip(dataTest, targetTest):
	y_true.append(y)
	y_pred.append(lin_clf.predict(x))

print(time.time()-t)
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print(precision_score(y_true, y_pred, average='weighted'))
print(recall_score(y_true, y_pred, average='weighted'))
print(f1_score(y_true, y_pred, average='weighted'))

'''
[[48  0  0  0  0  0  0  0  0  2]
 [ 0 50  0  0  0  0  0  0  0  0]
 [ 0  0 50  0  0  0  0  0  0  0]
 [ 0  0  0 34  0  0  0  0 10  6]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  0 48  0  0  1  1]
 [ 0  0  0  0  0  0 50  0  0  0]
 [ 0  0  0  0  0  0  0 50  0  0]
 [ 0  0  0  0  0  1  0  0 49  0]
 [ 0  0  0  0  0  0  0  0  1 49]]
accuracy_score 	= (48+50+50+34+50+48+50+50+49+49)/500 = 0.956
precision_score	= avg(48.0/48+50.0/50+50.0/50+34.0/34+50.0/50+48.0/49+50.0/50+50.0/50+49.0/61+49.0/58) = 0.9627698111466181
recall_score 	= avg(48.0/50+50.0/50+50.0/50+34.0/50+50.0/50+48.0/50+50.0/50+50.0/50+49.0/50+49.0/50) = 0.956
f1_score		= 2pr/p+r = 0.9593729629362364
time			= 0.0614831447601/500
'''