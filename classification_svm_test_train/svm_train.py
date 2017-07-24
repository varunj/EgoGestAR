import matplotlib as mpl
mpl.use('Agg')
import numpy as np
np.random.seed(123)
np.set_printoptions(threshold=np.nan)
import glob, os
import pandas as pd
from sklearn import svm

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

dataSeq = []
targetSeq = []
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
	dataSeq.append(arrstack) 
	targetSeq.append(targetarr)

data = np.vstack(dataSeq)
target = np.vstack(targetSeq)
seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = shuffle_data(data, seq)
target = shuffle_data(target, seq)
print(len(dataSeq))
print(data.shape, target.shape)

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]


lin_clf = svm.LinearSVC()
lin_clf.fit(data, target) 
# print(lin_clf.fit(data, target))
dec = lin_clf.decision_function([[400]])
print(dec.shape[1])