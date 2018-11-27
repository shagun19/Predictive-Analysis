import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

data = pd.read_csv("OriginalData.csv")  #enter name of csv here
train, test = train_test_split(data, test_size = 0.2)

#data must be in this format : [data_features, response variable] = data.ix[<index of features>], data.ix[<index of the response variable>]
[data_train, response_train] = train.ix[:,2:60], train.ix[:,60]       
[data_test, response_test] = test.ix[:,2:60], test.ix[:,60]
regrTree = RandomForestRegressor(max_depth=100, min_samples_leaf=500)
regrTree.fit(data_train, response_train)

predictedTest = regrTree.predict(data_test)
featureImp = regrTree.feature_importances_
names = train.columns.values[2:60]
merged = np.transpose([names,featureImp])
merged = merged[merged[:, 1].argsort()][::-1]
merged = np.asarray(merged)
print merged

final_Merged = np.array(merged[0:29,:])
print('Features Selected: ', final_Merged)

plt.figure(figsize=(10, 60))
plt.title('Random Forest regression Predictions')
plt.xlabel('Article Number')
plt.ylabel('Share Count')
plt.scatter(range(len(response_test.axes[0])), response_test, alpha=0.5, color='blue')
plt.scatter(range(len(response_test.axes[0])), predictedTest, alpha=0.5, color='green')
plt.legend(('True Share Count', 'Random Forest Predicted Share Count'))
plt.show()
