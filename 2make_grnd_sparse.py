import glob, os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from dtw import dtw
from math import sqrt
from pprint import pprint
import operator

for fileName in glob.glob("./train1/*"):
	with open(fileName, "r") as f:
		i = 0
		for eachLine in f:
			ff = open("./train1_sparse/" + fileName.replace('\\', '/').split('/')[2], 'a')
			if (i%2 == 0):
				ff.write(eachLine)
			i = i+1