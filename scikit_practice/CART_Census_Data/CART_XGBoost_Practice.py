#CART project
#using XGBoost
#train CART to learn whether an adult will have income greater than $50k based on census data


#to do
#fix the xgbboost error
#apply one hot encoder to the integer stand ins

# %%
#import pandas
import pandas as pd

#import train test split
from sklearn.model_selection import train_test_split

#import label encoder so we can setup the data for xgboost
from sklearn.preprocessing import LabelEncoder

#imports OneHotEncoder so we can get rid of the natural ordering that results in label encoding
#so the XGBoost classifier is not biased, e.g. education level 9 is better than 5
from sklearn.preprocessing import OneHotEncoder

#create a label encoder
le = LabelEncoder()

# %%
#setting the right working directory
import os
print(os.getcwd())
os.chdir("/Users/loganmarek/Development/datacamp-ml-python/scikit_practice/CART_Census_Data")
print(os.getcwd())

#imports xgboost
import xgboost as xgb

#imports numpy-
import numpy as np

# %%
#reads in the census data
census_data = pd.read_csv("adult.csv")

#create a new column with a binary flag of "greater than 5ok or less than or equal to 50k"
#%%
#no longer needed with label encoding
#census_data['greater_than_50k']=np.where(census_data['income'] == '>50K',1,0)

#drops the education column because it is labeled with numbers
#census_data.drop(['education'],axis = 1)

#create a boolean mask for categorical colums
cat_mask = (census_data.dtypes == 'object')

#creates a OneHotEncoder from the cat mask booleans
ohe = OneHotEncoder(categorical_features = cat_mask, sparse = False)

#get a list of categorical column names
cat_col = census_data.columns[cat_mask].tolist()

#apply LabelEncoder to categorical columns
census_data[cat_col] = census_data[cat_col].apply(lambda x: le.fit_transform(x))

#drops old number for identifying education
census_data.drop('education.num',axis=1)

# %%
#apply one OneHotEncoder to categorical columns, note the output is NO LONGER
#a dataframe
census_data_encoded = ohe.fit_transform(census_data)

#%%
#create target variable and features dataframes
X,y = census_data.iloc[:,:-2],census_data.iloc[:,-1]

#train test split the data with a random state so it's repeatable
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.15, random_state = 70)

#create XGB Classifier
xg_cl = xgb.XGBClassifier(objective = 'binary: logistic', n_estimators = 10, seed = 12345)

#%%
xg_cl.fit(X_train, y_train)

#%%
preds = xg_cl.predict(X_test)

accuracy = float(np.sums(preds==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))
