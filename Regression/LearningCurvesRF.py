import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve

data = pd.read_csv("OriginalData.csv")
X = data.ix[:,2:60]
y = data.ix[:,60]

regrTree = RandomForestRegressor(max_depth=500, min_samples_leaf=500)

train_sizes, train_scores, test_scores = learning_curve(regrTree, X, y, train_sizes=np.arange(1,15000,2000), cv=5)

plt.figure(figsize=(10, 60))
plt.title('Random Forest Regressor Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.show()
