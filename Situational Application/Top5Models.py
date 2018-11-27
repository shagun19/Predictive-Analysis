import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import r2_score

from sklearn.cross_validation import train_test_split

df = pd.read_csv('.\World.csv')

X = df.ix[:,3:61]
y = df.ix[:,61]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)

regrTree = RandomForestRegressor(max_depth=100, min_samples_leaf=500)
regrTree.fit(x_train, y_train)
print('Coefficient of determination Random Forest: ', regrTree.score(x_test,y_test))

lasso = linear_model.Lasso(alpha=0.01)
lasso.fit(x_train,y_train)
print('Coefficient of determination Lasso: ', lasso.score(x_test,y_test))

ridge = linear_model.Ridge(alpha=0.01,solver='sag')
ridge.fit(x_train,y_train)
print('Coefficient of determination Ridge: ', ridge.score(x_test,y_test))

linearRegObj = linear_model.LinearRegression()
linearRegObj.fit(x_train,y_train)
print('Coefficient of determination Linear: ', linearRegObj.score(x_test,y_test))

knn =  neighbors.KNeighborsRegressor(1000,weights = 'distance')
knn.fit(x_train, y_train)
print('Coefficient of determination KNN: ', knn.score(x_test,y_test))

