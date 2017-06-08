# move to trainx_raw. make folder ./result/
import glob
import matplotlib.pyplot as plt
import pprint

dic = {}
dicFiles = {}
for fileName in glob.glob("./train2_resampled200/*.txt"):
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

plt.bar(list(dic.keys()), dic.values(), color='g')
plt.show()