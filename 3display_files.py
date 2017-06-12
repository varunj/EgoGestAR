import glob, os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from dtw import dtw
from math import sqrt
from pprint import pprint

# make grid of 3x3
# dic_grndFiles = {}
# for fileName in glob.glob("./train1_resampled_200/*"):
# 	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
# 	arr = np.array(file.ix[:, :])
# 	dic_grndFiles[fileName] = arr 

# for grnd_name, arr_grnd in dic_grndFiles.items():
# 	print(grnd_name)
# 	n = 1
# 	for i in range(0, len(arr_grnd), int(len(arr_grnd)/8)):
# 		plt.subplot(3,3,n)
# 		n = n+1
# 		plt.plot([x[0] for x in arr_grnd[:i]], [-y[1] for y in arr_grnd[:i]])
# 		plt.xlim([0,640])
# 		plt.ylim([-480,0])
# 		plt.xticks([0,640])
# 		plt.yticks([-480,0])
# 		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# 	plt.show()


# make single image
dic_grndFiles = {}
for fileName in glob.glob("./train1_resampled_200/*"):
	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
	arr = np.array(file.ix[:, :])
	dic_grndFiles[fileName] = arr 

for grnd_name, arr_grnd in dic_grndFiles.items():
	print(grnd_name)
	n = 1
	plt.plot([x[0] for x in arr_grnd], [-y[1] for y in arr_grnd])
	plt.xlim([0,640])
	plt.ylim([-480,0])
	plt.xticks([0,640])
	plt.yticks([-480,0])
	plt.savefig('./images/single_' + grnd_name.split('_')[-2] + '.png')
	plt.close()