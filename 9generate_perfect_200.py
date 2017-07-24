import matplotlib.pyplot as plt
from pprint import pprint
import math
# canvas size M*N
# tab
# M = 1280
# N = 800
# pc
M = 800
N = 600

def rotate(l, n):
    return l[-n:] + l[:-n]

def pointsCircle(r, n = 100):
	l = [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]
	return list(reversed(rotate(l, n*63/100)))

def pointsStar(r, n = 100):
	allPts = pointsCircle(r, n)
	allPts = rotate(allPts, n*50/100)
	fiveParts = len(allPts)//5
	c = 15
	return [allPts[1*fiveParts-c], allPts[4*fiveParts-c], allPts[2*fiveParts-c], allPts[5*fiveParts-c], allPts[3*fiveParts-c], allPts[1*fiveParts-c]]

def shift(arr, x=320, y=240):
	return [(-pt[0]+x, -pt[1]+y) for pt in arr]

dic = {}
dic['down'] = [(320, 60), (320, 420)]
dic['up'] = [(320, 420), (320, 60)]
dic['left'] = [(560, 240), (80, 240)]
dic['right'] = [(80, 240), (560, 240)]
dic['star'] = shift(pointsStar(200))
dic['del'] = [(120, 50), (520, 430), (120, 430), (520, 50)]
dic['square'] = [(100, 100), (540, 100), (540, 380), (100, 380), (100, 100)]
dic['carret'] = [(100, 380), (320, 100), (540, 380)]
dic['tick'] = [(100, 140), (160, 350), (540,160)]
dic['circlecc'] = shift(pointsCircle(200))

for gestureName, gesturePts in dic.items():
	f = open('perfect/perfect_' + gestureName + '_1.txt', 'a')
	pointList = []
	for x in gesturePts:
		pointList.append([x[0], 480-x[1]])
	for x in range(0, len(pointList)):
		f.write("%e" %float(pointList[x][0]*640.0/M) + ' ' + "%e" %float((N-pointList[x][1])*480.0/N))
		f.write('\n')
	f.close()
