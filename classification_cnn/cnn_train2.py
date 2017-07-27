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
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras.utils import plot_model
import random
import cv2
from keras.utils import np_utils
import pdb
from keras.callbacks import ModelCheckpoint, Callback

img_rows, img_cols = 224, 224

def make_chan_first(path):
	img=cv2.resize(cv2.imread(path), (img_rows, img_cols))
	#pdb.set_trace()
	a = np.array(img[:,:,0])
	b = np.array(img[:,:,1])
	c = np.array(img[:,:,2])

	d = np.reshape(a, (1,img_rows,img_cols))
	e = np.reshape(b, (1,img_rows,img_cols))
	f = np.reshape(c, (1,img_rows,img_cols))
	img1 = np.append(d,e, axis=0)
	chan_fst_img = np.append(img1, f, axis =0)
	return chan_fst_img	



name="test1"
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
pdb.set_trace()
count=0
c = 0
trainInpStack =[]
trainGrndStack = np.zeros((NOS_INP_TRAIN))
for eachImgName in fileNamesArr[:NOS_INP_TRAIN]:
	trainInpPath = PATH_TRAIN + eachImgName
	#trainInpStack[c] = cv2.resize(cv2.imread(trainInpPath,0), (img_rows, img_cols)).reshape(1, img_rows, img_cols)
	if (count ==0):
			im1=make_chan_first(trainInpPath)
			trainInpStack=im1.reshape(1,3,img_rows,img_cols)
			#scipy.misc.imsave('npary.png',imgstack[0,:,:,:].transpose(1,2,0))
			count =count+1
	else:
			im2 = make_chan_first(trainInpPath)
			im3=im2.reshape(1,3,img_rows,img_cols)
			#print im2
			trainInpStack= np.vstack((trainInpStack, im3))

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
testInpStack = np.zeros((NOS_INP_TEST, 1, img_rows, img_cols))
testGrndStack = np.zeros((NOS_INP_TEST))
for eachImgName in fileNamesArr[:NOS_INP_TEST]:
	testInpPath = PATH_TEST + eachImgName
	testInpStack[c] = cv2.resize(cv2.imread(testInpPath,0), (img_rows, img_cols)).reshape(1, img_rows, img_cols)
	#cv2.imshow('iansadj', cv2.resize(cv2.imread(testInpPath,0), (img_rows, img_cols)))
	#cv2.waitKey(0)
	for i in range(0, NOS_CLASSES):
		if (CLASSES[i]==testInpPath.split('_')[2]):
			testGrndStack[c] = i
	c = c+1
testGrndStack = np_utils.to_categorical(testGrndStack, 10)



print(trainInpStack.shape , trainGrndStack.shape)
print(testInpStack.shape , testGrndStack.shape)
pdb.set_trace()
model = Sequential()
# Block 1
model.add(Conv2D(96, (7, 7), strides = (2,2), init='uniform', data_format= 'channels_first', activation='relu', padding='same', name='block1_conv1',input_shape=(3,img_rows, img_cols)))
print(model.layers[-1].output_shape)
model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool'))
print(model.layers[-1].output_shape)
model.add(BatchNormalization((96,256,256)))
print(model.layers[-1].output_shape)
# Block 2
model.add(Conv2D(256, (7, 7), strides = (2,2), init='uniform', activation='relu', padding='same', name='block2_conv'))
print(model.layers[-1].output_shape)
model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool'))
print(model.layers[-1].output_shape)
# Block 3
model.add(Conv2D(384, (7, 7), strides = (1,1), init='uniform', activation='relu', padding='same', name='block3_conv'))
print(model.layers[-1].output_shape)
# Block 4
model.add(Conv2D(384, (7, 7), strides = (1,1), init='uniform', activation='relu', padding='same', name='block4_conv'))
print(model.layers[-1].output_shape)
# Block 5
model.add(Conv2D(256, (7, 7), strides = (1,1), init='uniform', activation='relu', padding='same', name='block5_conv'))
print(model.layers[-1].output_shape)
model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block5_pool'))
print(model.layers[-1].output_shape)

pdb.set_trace()

model.add(Flatten(name='flatten'))

model.add(Dense(4096, activation='relu', name='fc1', init='uniform'))
model.add(Dense(4096, activation='relu', name='fc2', init='uniform'))

model.add(Dense(NOS_CLASSES, activation='softmax', name='predictions', init='uniform'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model(model, to_file='model'+'trial'+'.png')
print(model.summary())

pdb.set_trace()

filepath=name+"My_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit(trainInpStack, trainGrndStack, epochs=50, batch_size=10, verbose=1, callbacks=callbacks_list,validation_split=0.30)
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