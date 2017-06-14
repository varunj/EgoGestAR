import glob
import matplotlib.pyplot as plt
import pprint
import numpy as np

# histogram of x:length of gestures, y:nos of gestures. place all files from trainx into zcomb
dic = {}
dicFiles = {}
for fileName in glob.glob("./ztemp200all/*.txt"):
	with open(fileName, "r") as f:
		nosLines = sum(1 for eachLine in open(fileName, "r"))
		if not nosLines in dic:
			dic[nosLines] = 1
		else:
			dic[nosLines] += 1

		if not nosLines in dicFiles:
			dicFiles[nosLines] = [fileName]
		else:
			dicFiles[nosLines].append(fileName)

pprint.pprint(dicFiles)
plt.xticks([0,50,100,150,200,250,300], fontsize=12)
plt.yticks([0,5,10,15,20], fontsize=18)
plt.bar(list(dic.keys()), dic.values(), color='g')
plt.savefig('./images/hist.png', format='png', dpi=1200, bbox_inches='tight')


# mean stddev of lenth of gestures for each class. place all files from trainx into zcomb
# dic = {}
# for fileName in glob.glob("./train3/*.txt"):
# 	with open(fileName, "r") as f:
# 		gestureName = fileName.split('_')[-2]
# 		nosLines = sum(1 for eachLine in open(fileName, "r"))
# 		if (gestureName not in dic):
# 			dic[gestureName] = [nosLines]
# 		else:
# 			dic[gestureName].append(nosLines)

# for key, val in dic.items():
# 	print(key, np.std(val), sum(val)*1.0/len(val))

# mean stddev of time taken to draw gestures for each class. place all files from trainx into zcombtime
# dic = {}
# for fileName in glob.glob("./zcombtime/*.txt"):
# 	with open(fileName, "r") as f:
# 		gestureName = fileName.replace('.','_').split('_')[-2]
# 		dic[gestureName] = [float(eachLine[:-1]) for eachLine in open(fileName, "r")]

# for key, val in dic.items():
# 	print(key, np.std(val), sum(val)*1.0/len(val))