# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:35:33 2022

@author: jeeha
"""



from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
iris_petal = iris.data[:, [2, 3]]
labels = iris.target

plt.scatter(iris_petal[:, 0], iris_petal[:, 1], c = 'b', alpha = 0.7)
plt.show()


Kmean = KMeans(n_clusters = 7)
Kmean.fit(iris_petal)

print(Kmean.cluster_centers_)
print(Kmean.cluster_centers_[0])
print(Kmean.labels_)

plt.scatter(iris_petal[ : , 0], iris_petal[ : , 1], c = Kmean.labels_, cmap = 'rainbow', alpha = 0.5)
#plt.scatter(iris_petal[ : , 0], iris_petal[ : , 1], c = labels, marker = '+', cmap = 'rainbow')

#plt.scatter(Kmean.cluster_centers_[0][0], Kmean.cluster_centers_[0][1], s = 200, c='g', marker='s')
#plt.scatter(Kmean.cluster_centers_[1][0], Kmean.cluster_centers_[1][1], s = 200, c='r', marker='s')
#plt.scatter(Kmean.cluster_centers_[2][0], Kmean.cluster_centers_[2][1], s = 200, c='k', marker='s')

plt.show()

