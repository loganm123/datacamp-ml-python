#requires pandas, numpy, and scikit

#imports pandas
import pandas as pd

#imports numpy
import numpy as np

#imports the datasets from scikit
from sklearn import datasets

#imports matplot
import matplotlib.pyplot as plt

#imports the KNeighborsClassifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

#info on the breast_cancer dataset https://scikit-learn.org/stable/datasets/toy_dataset.html
#loads in the breast cancer data set and assigns it to b_cancer
b_cancer = datasets.load_breast_cancer()

#assigns the target array of 1 for malignant and 0 for benign
y = b_cancer.target
#assigns the X to all the other data
X = b_cancer.data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.333, random_state = 69, stratify = y)

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))
