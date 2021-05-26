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

tsne_features = model.fit_transform(samples)

print(tsne_features)
