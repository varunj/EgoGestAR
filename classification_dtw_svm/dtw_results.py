import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

'''
https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
'''

dic = pickle.load(open("./dic_dtw_results.pickle", "rb"))

# get true, pred
y_true = []
y_pred = []
for key, val in dic.items():
	y_true.append(key.split('_')[3])
	y_pred.append(min(val, key=val.get).split('_')[3])

print(confusion_matrix(y_true, y_pred, labels=['up','down','left','right','star','del','square','carret','tick','circlecc']))
print(accuracy_score(y_true, y_pred))
print(precision_score(y_true, y_pred, average='weighted'))
print(recall_score(y_true, y_pred, average='weighted'))
print(f1_score(y_true, y_pred, average='weighted'))

'''
[[34  1  9  0  0  1  0  2  3  0]
 [ 0 31  1  0  0  0 14  0  4  0]
 [ 0  1 48  0  0  0  1  0  0  0]
 [ 1  0  2 38  0  0  3  2  4  0]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  3  1  0  0 38  8  0  0  0]
 [ 0  0  0  0  0  0 50  0  0  0]
 [ 1  0  2 11  2  0  0 34  0  0]
 [ 0  3  0 40  0  0  0  0  5  2]
 [ 0  0  0  0  0  0  0  0  3 47]]
accuracy_score 	= (34+31+48+38+50+38+50+34+5+47)/500 = 0.75
precision_score	= avg(34.0/36,31.0/39,48.0/63,38.0/89,50.0/52,38.0/39,50.0/76,34.0/38,5.0/19,47.0/49) = 0.7639057876406866
recall_score 	= avg(34.0/50,31.0/50,48.0/50,38.0/50,50.0/50,38.0/50,50.0/50,34.0/50,5.0/50,47.0/50) = 0.7499999999999998
f1_score		= 2pr/p+r = 0.7568890289050075
time			= 2384.79674/500
'''