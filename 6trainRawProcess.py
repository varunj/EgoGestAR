# processes folder trainx_raw to generate trainx
import glob

PATHH = "train3_raw"
START = 101

for fileName in glob.glob("./" + PATHH + "/*.txt"):
	if (fileName.split('_')[-2] != 'time'):
		print(fileName)
		with open(fileName, "r") as f:
			print('Processing for: ' + fileName)
			c = START
			for eachLine in f:
				if (eachLine == '-----------\n'):
					c = c + 1
					ff.close()
					print('done: ' + fileName + ': ' + str(c-1))
				else:
					tempFileName = fileName.replace('.', '_').split('_')
					ff = open('./' + PATHH.split('_')[0] + '/train_' + tempFileName[-2] + '_' +str(c) + '.txt' , 'a')
					ff.write(eachLine)