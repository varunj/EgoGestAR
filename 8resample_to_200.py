	# processes folder trainx to generate trainx_resampled_200
# works only if reduction is by a factor of atmost 2
import glob
import matplotlib.pyplot as plt
import pprint
import numpy as np
import scipy as sp
import scipy.interpolate
import pandas as pd

TARGET_LEN = 200
PATHH = 'perfect'

def addBetween(inpList, x):
	a = np.zeros(shape=(1,2))
	a[0,0] = (inpList[x-1][0]+inpList[x][0])/2.0
	a[0,1] = (inpList[x-1][1]+inpList[x][1])/2.0
	toReturn = np.append(inpList[:x,:], a, axis=0)
	toReturn = np.append(toReturn, inpList[x:,:], axis=0)
	return toReturn

for fileName in glob.glob("./" + PATHH + "/*.txt"):
	print('processing: ' + fileName)
	fileNameSplit = fileName.replace('\\', '/').split('/')
	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
	arr = np.array(file.ix[:, :])
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
		arr = np.delete(arr, listRem, axis=0)


	ff = open('./' + PATHH + '_resampled_' + str(TARGET_LEN) + '/' + fileNameSplit[-1], 'a')
	for x in arr:
		ff.write("%e" % x[0] + " %e" % x[1] + '\n')
	