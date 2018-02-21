# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:29:39 2017

@author: adhingra
"""
# K-Means clustering 

# imporing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,3:].values

# no splitting of data into training and test sets
# no feature scaling

# using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('The Elbow Method')
plt.show()

# fitting k-means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 30, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# visualising dataset to the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c = 'green' , label = 'Cluster - 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c = 'violet' , label = 'Cluster - 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c = 'orange' , label = 'Cluster - 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c = 'black' , label = 'Cluster - 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c = 'blue' , label = 'Cluster - 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, color = 'yellow', label = 'Centroids')
plt.legend()
plt.title('K-Means Cluster')
plt.xlabel('Annual Income - k$')
plt.ylabel('Spending Score: 0-99')
plt.show()