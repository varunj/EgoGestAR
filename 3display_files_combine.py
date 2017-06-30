import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pprint import pprint

PATHH = 'train1_resampled_200'


# make grid of 3x3
# dic_grndFiles = {}
# for fileName in glob.glob("./" + PATHH + "/*.txt"):
# 	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
# 	arr = np.array(file.ix[:, :])
# 	fileName = fileName.split('_')[-2]
# 	if (fileName not in dic_grndFiles):
# 		dic_grndFiles[fileName] = [arr]
# 	else:
# 		dic_grndFiles[fileName].append(arr)
		
# for grnd_name, arr_grnd in dic_grndFiles.items():
# 	n = 1
# 	for i in range(0, len(arr_grnd[0]), int(len(arr_grnd[0])/8)):
# 		plt.subplot(3,3,n)
# 		n = n+1
# 		for each in arr_grnd:
# 			plt.plot([x[0] for x in each[:i]], [-y[1] for y in each[:i]])
# 		plt.xlim([0,640])
# 		plt.ylim([-480,0])
# 		plt.xticks([0,640])
# 		plt.yticks([-480,0])
# 		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# 	plt.savefig('./' + 'images/' + PATHH.split('_')[0] + '_' + grnd_name + '.png')
# 	plt.close()
# 	print('done: ' + grnd_name)


# make single image of 60 instances
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

	for each in arr_grnd[:60]:
		plt.plot([x[0] for x in each], [480-y[1] for y in each])
	plt.xlim([0,640])
	plt.ylim([0,480])
	plt.xticks([0,640], fontsize=18)
	plt.yticks([480], fontsize=18)
	plt.savefig('./images/all60_' + grnd_name + '.svg', format='svg', bbox_inches='tight')
	plt.close()
	print('done: ' + grnd_name)
