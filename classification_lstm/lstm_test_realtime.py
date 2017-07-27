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
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line
from kivy.gesture import Gesture, GestureDatabase
import time
import scipy as sp
import scipy.interpolate


CLASSES = ('up','down','left','right','star','del','square','carret','tick','circlecc')
NOS_CLASSES = len(CLASSES)
TARGET_LEN = 200
# canvas size M*N
# tab
# M = 1280
# N = 800
# pc
M = 800
N = 600

model = load_model('my_model20.hdf5')

def addBetween(inpList, x):
	#pdb.set_trace()
	a = np.zeros(shape=(1,2))
	a[0,0] = (inpList[x-1][0]+inpList[x][0])/2.0
	a[0,1] = (inpList[x-1][1]+inpList[x][1])/2.0
	toReturn = np.append(inpList[:x], a, axis=0)
	toReturn = np.append(toReturn, inpList[x:], axis=0)
	return toReturn

class GestureBoard(FloatLayout):
	"""
	Our application main widget, derived from touchtracer example, use data
	constructed from touches to match symboles loaded from my_gestures.
	"""
	def __init__(self, *args, **kwargs):
		super(GestureBoard, self).__init__()
		self.gdb = GestureDatabase()

	def on_touch_down(self, touch):
		# start collecting points in touch.ud create a line to display the points
		userdata = touch.ud
		with self.canvas:
			Color(1, 1, 0)
			d = 30.
			Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
			userdata['line'] = Line(points=(touch.x, touch.y))
		return True

	def on_touch_move(self, touch):
		# store points of the touch movement
		try:
			touch.ud['line'].points += [touch.x, touch.y]
			return True
		except (KeyError) as e:
			pass

	def on_touch_up(self, touch):
		arr = np.array([[0,0]])
		pointList = touch.ud['line'].points
		for x in range(0, len(pointList)-1, 2):
			arr = np.append(arr, np.array([[float(pointList[x]*640.0/M), float((N-pointList[x+1])*480.0/N)]]), axis=0)
		arr = np.delete(arr, 0, 0)
		inplen = len(arr)
		if (inplen < TARGET_LEN):
			x = 1
			while (inplen < TARGET_LEN):
				arr = addBetween(arr, x)
				x = x + 2
				if (x == len(arr)):
					x = 1
				inplen = len(arr)
		else:
			listRem = []
			for x in range(1, len(arr)):
				if (x%2 == 0):
					listRem.append(x)
			listRem = listRem[:min(len(listRem), abs(TARGET_LEN-len(arr)))]
			arr = np.delete(arr, listRem, ax)
		# arr = np.transpose(arr)
		arr = arr.reshape(1,200,2)
		result = model.predict(arr)
		
		ans = sorted(zip(CLASSES, result[0]), key=lambda x: x[1])
		print(ans)

		# erase the lines on the screen, this is a bit quick&dirty, since we can have another touch event on the way...
		self.canvas.clear()

class DemoGesture(App):
	def build(self):
		return GestureBoard()

if __name__ == '__main__':
	DemoGesture().run()
