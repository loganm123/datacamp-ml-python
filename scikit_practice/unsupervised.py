#clustering

#imports pandas
import pandas as pd

#imports numpy
import numpy as np

#imports numpy
import numpy as np

#imports the datasets from scikit
from sklearn import datasets

#imports matplot
import matplotlib.pyplot as plt

# Import TSNE
from sklearn.manifold import TSNE

#loads wine datasets as a numpy array
samples, y = datasets.load_wine(return_X_y=True)

#builds TSNE model
model = TSNE(learning_rate=50)

#creates features
tsne_features = model.fit_transform(samples)

#print(tsne_features)

#select 0th feature
xs = tsne_features[:,0]
#select the first feature
ys = tsne_features[:,1]
#creates scatter plot and colors by each unique wine set
plt.scatter(xs, ys, c = y)
#shows the clusters
plt.show()
