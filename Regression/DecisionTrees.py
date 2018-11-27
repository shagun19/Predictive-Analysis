import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split

data = pd.read_csv("OriginalData.csv")  #enter name of csv here
train, test = train_test_split(data, test_size = 0.2)

[data_train, response_train] = train.ix[:,2:60], train.ix[:,60]
[data_test, response_test] = test.ix[:,2:60], test.ix[:,60]

regrTree = DecisionTreeRegressor(max_depth=500, min_samples_leaf=500)
regrTree.fit(data_train, response_train)
predictedTest = regrTree.predict(data_test)

r2 = regrTree.score(data_test,response_test)
print('Coefficient of determination on test', r2)

plt.figure(figsize=(10, 60))
plt.title('Decision tree Predictions')
plt.xlabel('Article Number')
plt.ylabel('Share Count')
plt.scatter(range(len(response_test.axes[0])), response_test, alpha=0.5, color='blue')
plt.scatter(range(len(response_test.axes[0])), predictedTest, alpha=0.5, color='green')
plt.legend(('True Share Count', 'Decision tree predicted share count'))
plt.show()
