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
from keras.layers import LSTM, Bidirectional
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, Callback
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


name="20concat"
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
	#arr1 = np.transpose(arr)
	targetarr = np.zeros(NOS_CLASSES)
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==fname):
			targetarr[i] = 1
	arrstack = arr.reshape(1,200,2)
	dataSeq.append(arrstack) 
	targetSeq.append(targetarr)
	#print(fname, fileName)

#print(len(dataSeq))
data = np.vstack(dataSeq)
target = np.vstack(targetSeq)
seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = shuffle_data(data, seq)
target = shuffle_data(target, seq)

#pdb.set_trace()


model = Sequential()  

model.add(Bidirectional(LSTM(20, input_dim=2,input_length=200, return_sequences=True), input_shape=(200,2), merge_mode='concat'))
# model.add(LSTM(20, input_dim=2,input_length=200, return_sequences=True))

model.add(Flatten())
model.add(Dense(NOS_CLASSES, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# pdb.set_trace()
# plot_model(model, to_file='model'+name+'.png')

print (model.get_config())
print(data.shape, target.shape)

#model.layers[0].w.get_value()
#output_layer=model.layers[0].get_output()
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
filepath=name+"My_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]
#history = model.fit(img_data, labels2, nb_epoch=150, batch_size=100, validation_split=0.30, callbacks=callbacks_list, verbose = 1)
history = model.fit(data, target, epochs=600, batch_size=10, verbose=2, callbacks=callbacks_list,validation_split=0.30)


#pdb.set_trace()
#model.save('my_model20.h5')
#model.save_weights('my_model_weights20.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('accuracy'+name+'.png')
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('loss'+name+'.png')
print "Done"