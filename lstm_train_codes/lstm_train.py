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
from keras.utils import plot_model

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
for fileName in glob.glob("../zcomb/*.txt"):
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
	print(fname, fileName)

print(len(dataSeq))
data = np.vstack(dataSeq)
target = np.vstack(targetSeq)
seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = shuffle_data(data, seq)
target = shuffle_data(target, seq)

# pdb.set_trace()

model = Sequential()  
model.add(LSTM(200, input_shape=(2,200), return_sequences=True))
model.add(Flatten())
model.add(Dense(NOS_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(data, target, epochs=300, batch_size=10, verbose=2, validation_split=0.30)

model.save('my_model.h5')
model.save_weights('my_model_weights.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')