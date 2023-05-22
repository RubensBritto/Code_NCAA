import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC

features = ['GP', 'Ortg', 'TPA', 'adjoe', 'rimmade/(rimmade+rimmiss)', 'dunksmade', 'dunksmiss+dunksmade', 'adrtg', 'dporpag', 'stops', 'gbpm', 'stl', 'blk', 'pts']

data = pd.read_csv("newDataSet.csv")
X = data[features]
y = data["ROUND"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=50)




clf = SVC(C=1, gamma=1, kernel='linear').fit(X_train, y_train)
y_pred=clf.predict(X_test)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
