import matplotlib.pyplot as plt
from pprint import pprint
import math

def pointsCircle(r, n = 100):
	return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]

def pointsStar(r, n = 100):
	allPts = pointsCircle(r, n)
	fiveParts = len(allPts)//5
	c = 15
	return [allPts[1*fiveParts-c], allPts[3*fiveParts-c], allPts[5*fiveParts-c], allPts[2*fiveParts-c], allPts[4*fiveParts-c], allPts[1*fiveParts-c]]

def shift(arr, x=320, y=240):
	return [(-pt[0]+x, -pt[1]+y) for pt in arr]

dic = {}
dic['up'] = [(320, 60), (320, 420)]
dic['down'] = [(320, 60), (320, 420)]
dic['left'] = [(80, 240), (560, 240)]
dic['right'] = [(80, 240), (560, 240)]
dic['star'] = shift(pointsStar(200))
dic['del'] = [(520, 50), (120, 430), (520, 430), (120, 50)]
dic['square'] = [(100, 100), (540, 100), (540, 380), (100, 380), (100, 100)]
dic['carret'] = [(100, 380), (320, 100), (540, 380)]
dic['tick'] = [(100, 140), (160, 350), (540,160)]
dic['circlecc'] = shift(pointsCircle(200))

for gestureName, gesturePts in dic.items():
	plt.plot([x[0] for x in gesturePts], [480-y[1] for y in gesturePts], linewidth=4)
	plt.xlim([0,640])
	plt.ylim([0,480])
	plt.xticks([0,640], fontsize=18)
	plt.yticks([480], fontsize=18)
	plt.savefig('./images/single_' + gestureName + '.eps', format='eps', bbox_inches='tight')
	plt.close()
	print(gestureName)