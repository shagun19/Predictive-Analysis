import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.learning_curve import learning_curve

df = pd.read_csv("C:\Users\Shreyas\Desktop\Projects\OnlineNewsPopularity\Phase - 2 Normalization, feature selection\Training-Test splits\NormalizedDataSet.csv")
X = df.ix[:,:58]
y = df.ix[:,58]

knn =  neighbors.KNeighborsRegressor(1000,weights = 'distance')

train_sizes, train_scores, test_scores = learning_curve(knn, X, y, train_sizes=np.arange(1000,10000,2000), cv=5)

plt.figure(figsize=(10, 60))
plt.title('Nearest Neighbours Learning Curve')
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
