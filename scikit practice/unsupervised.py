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
wine = datasets.load_wine()

#converts to numpy array
wine_a = np.asarray(wine)

#builds TSNE model
model = TSNE(learning_rate=50)

tsne_features = model.fit_transform(wine_a)

print(tsne_features)
