from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

data = pd.read_csv("OriginalData.csv")  #enter name of csv here
train, test = train_test_split(data, test_size = 0.2)

[data_train, response_train] = train.ix[:,2:60], train.ix[:,60]
[data_test, response_test] = test.ix[:,2:60], test.ix[:,60]

linearRidge = Ridge(alpha = 0.001,solver = 'sag',random_state=88)
linearRidge.fit(data_train,response_train)
predictedTest = linearRidge.predict(data_test)

print('Mean squared error',mean_squared_error(response_test,predictedTest))
print('Intercept:', linearRidge.intercept_)
print('Coefficients:', linearRidge.coef_)
print('R2 on test', r2_score(response_test, predictedTest))

plt.figure(figsize=(10, 60))
plt.title('Ridge regression predictions')
plt.xlabel('Article Number')
plt.ylabel('Share Count')
plt.scatter(range(len(response_test.axes[0])), response_test, alpha=0.5, color='blue')
plt.scatter(range(len(response_test.axes[0])), predictedTest, alpha=0.5, color='green')
plt.legend(('True Share Count', 'Ridge regression predicted share count'))
plt.show()
