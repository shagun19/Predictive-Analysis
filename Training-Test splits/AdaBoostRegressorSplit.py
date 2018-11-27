#Essential Modules to import!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

df = pd.read_csv("C:\Users\Shreyas\Desktop\Projects\OnlineNewsPopularity\Phase - 2 Normalization, feature selection\Training-Test splits\NormalizedDataSet.csv")
X = df.ix[:,:58]
y = df.ix[:,58]

increments = np.arange(0.1,1,0.1)
for i in increments:
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=i, random_state=42)
    abr = AdaBoostRegressor(n_estimators=2)
    abr.fit(x_train, y_train)
    print('Coefficient of determination at : ',i, abr.score(x_test,y_test))
