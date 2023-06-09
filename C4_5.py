# -*- coding: utf-8 -*-

!pip install chefboost

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from chefboost import Chefboost as chef

data = pd.read_csv('newDataSet.csv')

features = ['GP', 'Ortg', 'TPA', 'adjoe', 'rimmade/(rimmade+rimmiss)', 'dunksmade', 'dunksmiss+dunksmade', 'adrtg', 'dporpag', 'stops', 'gbpm', 'stl', 'blk', 'pts','ROUND']
new_data = data[features]

new_data['ROUND'] = new_data['ROUND'].replace(1.0, 'um')
new_data['ROUND'] = new_data['ROUND'].replace(0.0, 'zero')
new_data['ROUND'] = new_data['ROUND'].replace(2.0, 'dois')



config = {'algorithm': 'C4.5'}

model = chef.fit(new_data, config = config, target_label = 'ROUND')
