import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

data = pd.read_csv("OriginalData.csv")  #enter name of csv here
train, test = train_test_split(data, test_size = 0.2)

[data_train, response_train] = train.ix[:,2:60], train.ix[:,60]
[data_test, response_test] = test.ix[:,2:60], test.ix[:,60]

regrTree = RandomForestRegressor(max_depth=100, min_samples_leaf=500)
regrTree.fit(data_train, response_train)

predictions = regrTree.predict(data_test)
data_test['predicted_shares'] = predictions

#this value has been chosen as the one at the 75th percentile
data_test['popular'] = predictions >= 4370.279611

pop = data_test['popular']

print pop.value_counts()

data_test.popular = data_test.popular.astype(int)

classifier = RandomForestClassifier(n_estimators = 100, max_depth = 100, min_samples_leaf = 500)

[classifier_train ,classifier_test] = data_test[1:5786], data_test[5786:9645]
[train_classifier, train_response] = classifier_train.ix[:,0:58], classifier_train.ix[:,59]
[test_classifier, test_response] = classifier_test.ix[:,0:58], classifier_test.ix[:,59]

classifier.fit(train_classifier,train_response)

class_predictions = classifier.predict(test_classifier)
print classifier.score(test_classifier, test_response)


