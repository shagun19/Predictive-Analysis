import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

data = pd.read_csv("OriginalData.csv")
train, test = train_test_split(data, test_size = 0.2)

#data must be in this format : [data_features, response variable] = data.ix[<index of features>], data.ix[<index of the response variable>]
[data_train, response_train] = train.ix[:,2:60], train.ix[:,60]
[data_test, response_test] = test.ix[:,2:60], test.ix[:,60]

linearRegObj = linear_model.LinearRegression()
linearRegObj.fit(data_train,response_train)
predictedTest = linearRegObj.predict(data_test)

print('Intercept:', linearRegObj.intercept_)
print('Coefficients:', linearRegObj.coef_)
print('R2 on test', r2_score(response_test, predictedTest))

plt.figure(figsize=(10, 60))
plt.title('Linear regression Predictions')
plt.xlabel('Article Number')
plt.ylabel('Share Count')
plt.scatter(range(len(response_test.axes[0])), response_test, alpha=0.5, color='blue')
plt.scatter(range(len(response_test.axes[0])), predictedTest, alpha=0.5, color='green')
plt.legend(('True Share Count', 'Linear model Predicted Share Count'))
plt.show()
