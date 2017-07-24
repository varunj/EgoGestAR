import glob, os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from math import sqrt
from pprint import pprint

dic_grndFiles = {}
for fileName in glob.glob("./train_resampled_200/*"):
	file = pd.read_csv(fileName, delim_whitespace = True, header = None)
	arr = np.array(file.ix[:, :])
	dic_grndFiles[fileName] = arr 

c = 0
for grnd_name, arr_grnd in dic_grndFiles.items():
	c = c + 1
	print(c, grnd_name)
	plt.plot([x[0] for x in arr_grnd], [-y[1] for y in arr_grnd], linewidth=9, color="1")
	plt.plot(arr_grnd[0][0], -arr_grnd[0][1], 'w.', markersize=60.0)
	plt.xlim([0,640])
	plt.ylim([-480,0])
	plt.xticks([0,640])
	plt.yticks([-480,0])
	plt.axis('off')
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	nameSplit = grnd_name.replace('\\','_').replace('.','_').split('_')
	plt.savefig('.'+ nameSplit[1] + '_images/' + nameSplit[4] + '_' + nameSplit[5] + '_' + nameSplit[6] + '.png', dpi=100, facecolor='black')
	plt.close()