import pandas as pd
import numpy as np

data = pd.read_csv("OriginalData.csv")

labels = data.columns.values
sortedData = data.sort_values(by = labels[60],ascending = False)
topTopics = sortedData.head(10)

lifestyle = topTopics.loc[topTopics.ix[:,13]==1]
entertainment = topTopics.loc[topTopics.ix[:,14]==1]
bus = topTopics.loc[topTopics.ix[:,15]==1]
socmed = topTopics.loc[topTopics.ix[:,16]==1]
tech = topTopics.loc[topTopics.ix[:,17]==1]
world = topTopics.loc[topTopics.ix[:,18]==1]

print "Top 10"
print 
print("Lifestyle",len(lifestyle))
print("Entertainment",len(entertainment))
print("Business",len(bus))
print("Social media",len(socmed))
print("Tech",len(tech))
print("World",len(world))




