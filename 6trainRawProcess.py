# move to trainx_raw. make folder ./result/
import glob

for fileName in glob.glob("./*.txt"):
	fileNameSplit = fileName.replace('.', '_').split('_')
	if (fileNameSplit[1] != 'time'):
		with open(fileName, "r") as f:
			print('Processing for: ' + fileName)
			c = 1
			for eachLine in f:
				if (eachLine == '-----------\n'):
					c = c + 1
					ff.close()
					print('Written ' + str(c))
				else:
					ff = open('./result/train_' + fileNameSplit[2] + '_' +str(c) + '.txt' , 'a')
					ff.write(eachLine)