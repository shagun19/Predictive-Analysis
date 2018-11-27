import pandas as pd
import numpy as np

data = pd.read_csv("OriginalData.csv")

#calculating the share count  based on topics 

lifestyleData = data.loc[data.ix[:,13]==1]
entertainmentData = data.loc[data.ix[:,14]==1]
busData = data.loc[data.ix[:,15]==1]
socmedData = data.loc[data.ix[:,16]==1]
techData = data.loc[data.ix[:,17]==1]
worldData = data.loc[data.ix[:,18]==1]

print("Lifestyle share count mean", lifestyleData.ix[:,60].mean())
print("Entertainment share count mean", entertainmentData.ix[:,60].mean())
print("Business share count mean", busData.ix[:,60].mean())
print("Social media share count mean",socmedData.ix[:,60].mean())
print("Tech share count mean", techData.ix[:,60].mean())
print("World share count mean", worldData.ix[:,60].mean())
print(" ")
print("Lifestyle share count sum",lifestyleData.ix[:,60].sum())
print("Entertainment share count sum", entertainmentData.ix[:,60].sum())
print("Business share count sum", busData.ix[:,60].sum())
print("Social media share count sum", socmedData.ix[:,60].sum())
print("Tech share count sum", techData.ix[:,60].sum())
print("World share count sum", worldData.ix[:,60].sum())
