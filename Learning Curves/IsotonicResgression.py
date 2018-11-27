#Essential Modules to import!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection

from sklearn.isotonic import IsotonicRegression
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve

df = pd.read_csv("C:\Users\Shreyas\Desktop\Projects\OnlineNewsPopularity\Phase - 2 Normalization, feature selection\Training-Test splits\NormalizedDataSet.csv")
X = df.ix[:,:58]
y = df.ix[:,58]

ir = IsotonicRegression(increasing = 'auto')

train_sizes, train_scores, test_scores = learning_curve(ir, X.ix[:,25], y, train_sizes=np.arange(1,15000,2000), cv=5)

plt.figure(figsize=(10, 60))
plt.title('Isotonic Regressor Learning Curve')
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
