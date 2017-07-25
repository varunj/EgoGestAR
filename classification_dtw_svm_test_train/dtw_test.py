import glob, os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from dtw import dtw
from math import sqrt
from pprint import pprint
import operator
import pickle

'''
refer https://github.com/pierre-rouanet/dtw/blob/master/dtw.py
'''

# read ground truths
dic_grndFiles = {}
for fileName in glob.glob("../perfect_resampled_200/*"):
	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
	arr = np.array(file.ix[:, :])
	dic_grndFiles[fileName] = arr 

# read input files
dic_seedFiles = {}
for fileName in glob.glob("../test_resampled_200/*"):
	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
	arr = np.array(file.ix[:, :])	
	dic_seedFiles[fileName] = arr 

print(len(dic_grndFiles), len(dic_seedFiles))

dic_toDump = {}
c = 0
# analyse each file
for seed_name, arr_seed in dic_seedFiles.items():
	dic_ans = {}
	c = c + 1
	for grnd_name, arr_grnd in dic_grndFiles.items():
		distance, cost, acc, path = dtw(arr_seed, arr_grnd, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
		dic_ans[str(grnd_name)] = distance*100.0/(640+480)
		# plt.plot(path[0], path[1])
		# plt.show()
	# print(seed_name)	
	# print(min(dic_ans, key=dic_ans.get))
	# print(sorted(dic_ans.items(), key=operator.itemgetter(1)))
	dic_toDump[seed_name] = dic_ans
	print(c)

pickle.dump(dic_toDump, open( "./dic_dtw_results.pickle", "wb" ), protocol=2)