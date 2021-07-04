#CART project
#using XGBoost
#train CART to learn whether an adult will have income greater than $50k based on census data


#fix the value error from the y_train data... I formatted it wrong somehow

# In[cell1]
#import pandas
import pandas as pd

#import train test split
from sklearn.model_selection import train_test_split


#setting the right working directory
import os
print(os.getcwd())
os.chdir("/Users/loganmarek/Development/datacamp-ml-python/scikit_practice/CART_Census_Data")
print(os.getcwd())

#imports xgboost
import xgboost as xgb

#imports numpy-
import numpy as np

# In[cell2]
#reads in the census data
census_data = pd.read_csv("adult.csv")

#create a new column with a binary flag of "greater than 5ok or less than or equal to 50k"

census_data['greater_than_50k']=np.where(census_data['income'] == '>50K',1,0)

census_data.head(15)

#create target variable and features dataframes
X,y = census_data.iloc[:,:-1],census_data.iloc[:,-1:]

#train test split the data with a random state so it's repeatable
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.15, random_state = 70)

#create XGB Classifier
xg_cl = xgb.XGBClassifier(objective = 'binary: logistic', n_estimators = 10, seed = 12345)

xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)

accuracy = float(np.sums(preds==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))
