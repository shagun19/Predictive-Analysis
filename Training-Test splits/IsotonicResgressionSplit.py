#Essential Modules to import!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("C:\Users\Shreyas\Desktop\Projects\OnlineNewsPopularity\Phase - 2 Normalization, feature selection\Training-Test splits\NormalizedDataSet.csv")
X = df.ix[:,:58]
y = df.ix[:,58]

increments = np.arange(0.1,1,0.1)
for i in increments:
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=i, random_state=42)
    x_train = x_train.ix[:,25] #x = kw_avg_avg
    x_test = x_test.ix[:,25]
    ir = IsotonicRegression(increasing = 'auto')
    ir.fit(x_train, y_train)
    y_ir = ir.predict(x_test)
    print('Coefficient of determination at : ',i, r2_score(y_test,y_ir))
