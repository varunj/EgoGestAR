import glob, os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from dtw import dtw
from math import sqrt
from pprint import pprint

PATHH = 'train2_resampled_200'

dic_grndFiles = {}
for fileName in glob.glob("./" + PATHH + "/*.txt"):
	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
	arr = np.array(file.ix[:, :])
	fileName = fileName.split('_')[-2]
	if (fileName not in dic_grndFiles):
		dic_grndFiles[fileName] = [arr]
	else:
		dic_grndFiles[fileName].append(arr)
		

for grnd_name, arr_grnd in dic_grndFiles.items():
	n = 1
	for i in range(0, len(arr_grnd[0]), int(len(arr_grnd[0])/8)):
		plt.subplot(3,3,n)
		n = n+1
		for each in arr_grnd:
			plt.plot([x[0] for x in each[:i]], [-y[1] for y in each[:i]])
		plt.xlim([0,640])
		plt.ylim([-480,0])	
	plt.savefig('./' + 'images/' + PATHH.split('_')[0] + '_' + grnd_name + '.png')
	plt.close()
	print('done: ' + grnd_name)