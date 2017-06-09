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

CLASSES = ('up','del','down','left', 'ques','star','tick','right', 'carret','square','circlec','circlecc')
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
for fileName in glob.glob("../train2_resampled_200_10/*.txt"):
	file = pd.read_csv(fileName, delim_whitespace=True, header=None)
	fname = fileName.split('_')[4]
	arr = np.array(file.ix[:, :])
	arr1 = np.transpose(arr)
	targetarr = np.zeros(NOS_CLASSES)
	for i in xrange(0, NOS_CLASSES):
		if (CLASSES[i]==fname):
			targetarr[i] = 1
	arrstack = arr1.reshape(1,2,200)
	dataSeq.append(arrstack) 
	targetSeq.append(targetarr)
	print fname, fileName

print len(dataSeq)
data = np.vstack(dataSeq)
target = np.vstack(targetSeq)
seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = shuffle_data(data, seq)
target = shuffle_data(target, seq)

# pdb.set_trace()

model = Sequential()  
model.add(LSTM(200, input_shape=(2,200), return_sequences=True))
print(model.layers[-1].output_shape)

model.add(Flatten())
print(model.layers[-1].output_shape)

model.add(Dense(NOS_CLASSES, activation='softmax'))
print(model.layers[-1].output_shape)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, target, epochs=1000, batch_size=10, verbose=2, validation_split=0.30)

scores = model.predict(tq)
print scores