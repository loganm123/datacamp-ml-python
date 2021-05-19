#requires pandas, numpy, and scikit

#imports pandas
import pandas as pd

#imports numpy
import numpy as np

#imports the datasets from scikit
from sklearn import datasets

#imports matplot
import matplotlib.pyplot as plt

#imports the KNeighborsClassifier from sklearn and other important things
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

#info on the breast_cancer dataset https://scikit-learn.org/stable/datasets/toy_dataset.html
#loads in the breast cancer data set and assigns it to b_cancer
b_cancer = datasets.load_breast_cancer()

#assigns the target array of 1 for malignant and 0 for benign
y = b_cancer.target
#assigns the X to all the other data
X = b_cancer.data
#create a training set composed of a third of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.333, random_state = 69, stratify = y)


#create a range of k values from 1 - 50
neighbors = list(range(1,51))

#empty list to hold cv scores
cv_scores = []

#perform five fold cross-validation on each k number of n_neighbors
for k in neighbors:
    #initialize the classifier at the number of neighbors
    knn = KNeighborsClassifier(n_neighbors= k)
    #compute the scores for that k neighbors with a five fold scoring off of accuracy
    scores = cross_val_score(knn, X_train, y_train, cv = 5, scoring = "accuracy")
    #add the mean of the scores to the cv_scores list
    cv_scores.append(scores.mean())

#plot neighbors against avg cv score to find the best number of neighbors for a k means clustering
plt.plot(neighbors,cv_scores)
plt.xlabel("Neighbors")
plt.ylabel("CV Score Average (5 fold)")
plt.show()
