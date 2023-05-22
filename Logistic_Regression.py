import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


features = ['GP', 'Ortg', 'TPA', 'adjoe', 'rimmade/(rimmade+rimmiss)', 'dunksmade', 'dunksmiss+dunksmade', 'adrtg', 'dporpag', 'stops', 'gbpm', 'stl', 'blk', 'pts']

data = pd.read_csv("newDataSet.csv")
X = data[features]
y = data["ROUND"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=50)

clf= LogisticRegression(random_state=33, max_iter=200).fit(X_train, y_train)
y_pred=clf.predict(X_test)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

