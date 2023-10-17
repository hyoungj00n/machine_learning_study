# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:35:33 2022

@author: jeeha
"""
# load packages
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load and assign the data
iris = load_iris()
iris_petal = iris.data[:, [2, 3]]
labels = iris.target

plt.scatter(iris_petal[:, 0], iris_petal[:, 1], c = 'b', alpha = 0.7)
plt.show()

# training
Kmean = KMeans(n_clusters = 3)
Kmean.fit(iris_petal)

# result
print(Kmean.cluster_centers_)
#print(Kmean.cluster_centers_[0])
print(Kmean.labels_)

colours = ['tab:blue', 'tab:orange', 'tab:green']
c_colours = ['b', 'r', 'g']
for i in range(len(Kmean.labels_)):
    label = Kmean.labels_[i]
    plt.scatter(iris_petal[i][0], iris_petal[i][1], c = colours[label], alpha = 0.5)

for i in range(len(Kmean.cluster_centers_)):
    plt.scatter(Kmean.cluster_centers_[i][0], Kmean.cluster_centers_[i][1], s = 200, c = c_colours[i])

plt.show()

for i in range(len(labels)):
    label = labels[i]
    plt.scatter(iris_petal[i][0], iris_petal[i][1], c = colours[label], alpha = 0.5)

plt.show()

import numpy as np

u, indices = np.unique(Kmean.labels_, return_index = True)
lowest = indices.tolist().index(min(indices))
highest = indices.tolist().index(max(indices))
print (lowest, highest)



