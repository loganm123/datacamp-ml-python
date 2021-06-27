#CART project
#using XGBoost
#train CART to learn whether an adult will have income greater than $50k based on census data

# In[cell1]
#import pandas
import pandas as pd

#import train test split
from sklearn.model_selection import train_test_split

import os
print(os.getcwd())
os.chdir("/Users/loganmarek/Development/datacamp-ml-python/scikit_practice/CART_Census_Data")
print(os.getcwd())

#imports xgboost
import xgboost as xgb

#imports numpy
import numpy as np

# In[cell2]
#reads in the census data
census_data = pd.read_csv("adult.csv")

census_data.info()

census_data.head(10)
