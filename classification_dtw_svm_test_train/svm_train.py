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
dataTrain = {x1, x2, .... x200, y1, ... y200}
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
	arr = arr[1::2]
	arr1 = np.transpose(arr)
	targetarr = 0
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==fname):
			targetarr = i
	# arrstack = arr1.reshape(1,2,200)
	arrstack = arr1.reshape(1,200)
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
	arr = arr[1::2]
	arr1 = np.transpose(arr)
	targetarr = 0
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==fname):
			targetarr = i
	# arrstack = arr1.reshape(1,2,200)
	arrstack = arr1.reshape(1,200)
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

# test SVM
y_true = []
y_pred = []
for x, y in zip(dataTest, targetTest):
	y_true.append(y)
	y_pred.append(lin_clf.predict(x))

print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print(precision_score(y_true, y_pred, average='weighted'))
print(recall_score(y_true, y_pred, average='weighted'))
print(f1_score(y_true, y_pred, average='weighted'))

'''
('train set', (2000, 400), (2000, 1))
('test set ', (500, 400), (500, 1))
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

-----------------------------------
only x
('train set', (2000, 200), (2000, 1))
('test set ', (500, 200), (500, 1))
[[50  0  0  0  0  0  0  0  0  0]
 [50  0  0  0  0  0  0  0  0  0]
 [ 0  0 50  0  0  0  0  0  0  0]
 [10  0  0 39  0  0  0  1  0  0]
 [ 1  0  0  0 14 11 24  0  0  0]
 [ 3  0  0  0  0 33 13  1  0  0]
 [ 0  0  0  0  1  0 49  0  0  0]
 [45  0  0  0  0  0  0  5  0  0]
 [10  0  0  3  0  7  0 16 14  0]
 [ 0  0  0  1  0  0  0  0  0 49]]
0.606
0.657038563542
0.606
0.566453878585

-----------------------------------
only y
('train set', (2000, 200), (2000, 1))
('test set ', (500, 200), (500, 1))
[[50  0  0  0  0  0  0  0  0  0]
 [ 0 50  0  0  0  0  0  0  0  0]
 [ 0  0 29  0  0  0  0  0 21  0]
 [ 0  0 33  0  0  0  0  0 17  0]
 [ 1  0  0  0 49  0  0  0  0  0]
 [ 0  0  0  0  0 44  4  0  1  1]
 [ 0  0  0  0  0  0 27  0  6 17]
 [ 0  0  0  0  0  0  0 50  0  0]
 [ 0  0  0  0  0  2  3  0 45  0]
 [ 0  0  0  0  0  2 15  0 15 18]]
0.724
0.684439259575
0.724
0.694051868424

-----------------------------------
remove alt
('train set', (2000, 200), (2000, 1))
('test set ', (500, 200), (500, 1))
[[50  0  0  0  0  0  0  0  0  0]
 [ 0 50  0  0  0  0  0  0  0  0]
 [ 0  0 50  0  0  0  0  0  0  0]
 [ 0  0  0 47  0  0  0  0  0  3]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  0 49  0  0  0  1]
 [ 0  0  0  0  0  0 50  0  0  0]
 [ 0  0  0  0  0  0  0 50  0  0]
 [ 0  0  0 12  0  4  0  0 32  2]
 [ 0  0  0  0  0  0  0  0  0 50]]
0.956
0.961399561424
0.956
0.953772566307

-----------------------------------
remove alt alt
('train set', (2000, 100), (2000, 1))
('test set ', (500, 100), (500, 1))
[[50  0  0  0  0  0  0  0  0  0]
 [ 0 50  0  0  0  0  0  0  0  0]
 [ 0  0 50  0  0  0  0  0  0  0]
 [ 0  0  0 47  0  0  0  0  0  3]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  0 49  0  0  0  1]
 [ 0  0  0  0  0  0 50  0  0  0]
 [ 0  0  0  0  0  0  0 50  0  0]
 [ 0  0  0  1  0  6  0  0 42  1]
 [ 0  0  0  0  0  0  0  0  0 50]]
0.976
0.977916666667
0.976
0.975794143744

'''