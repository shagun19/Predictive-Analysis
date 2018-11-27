import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("C:\Users\Shreyas\Desktop\Projects\OnlineNewsPopularity\Phase - 2 Normalization, feature selection\Situational Application\OnlineNewsPopularity.csv")
lifestyle = df.ix[df.ix[:,13] == 1] #Select Lifestyle
lifestyle.to_csv(path_or_buf="Lifestyle.csv")

ent = df.ix[df.ix[:,14] == 1] #Select Entertainment
ent.to_csv(path_or_buf="Entertainment.csv")

business = df.ix[df.ix[:,15] == 1] #Select Business
business.to_csv(path_or_buf="Business.csv")

socmed = df.ix[df.ix[:,16] == 1] #Select Social Media
socmed.to_csv(path_or_buf="socmed.csv")

tech = df.ix[df.ix[:,17] == 1] #Select Technology
tech.to_csv(path_or_buf="Technology.csv")

world = df.ix[df.ix[:,18] == 1] #Select World
world.to_csv(path_or_buf="World.csv")
