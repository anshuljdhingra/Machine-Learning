# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:26:36 2017

@author: adhingra
"""
# Hierarchical Clustering

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,3:].values

# using dendrograms to determine the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X , method = 'ward'))
plt.title('Dendrogram')
plt.xtitle('Customers')
plt.ytitle('Euclidean Distance')
plt.show()

# fitting Hierarchical Clustering Algorithm to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# visualising HC
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s =100, c = 'red' , label = 'Cluster - 1' )
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s =100, c = 'yellow' , label = 'Cluster - 2' )
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s =100, c = 'black' , label = 'Cluster - 3' )
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s =100, c = 'orange' , label = 'Cluster - 4' )
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s =100, c = 'blue' , label = 'Cluster - 5' )
plt.xlabel('Annual Income - k$')
plt.ylabel('Spending')
plt.title('Hierarchical Clustering of Customers')
plt.legend()
plt.show()

'''# visualising HC using for loop
colors = ['red', 'blue', 'green', 'black', 'pink']
labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']
i = 0
while i < 5:
    plt.scatter(X[y_hc == i,0], X[y_hc == i, 1], s = 100, c = colors[i], label = labels[i])
    i = i+1
plt.xlabel('Annual Income - k$')
plt.ylabel('Spending')
plt.title('Hierarchical Clustering of Customers')
plt.legend()
plt.show() '''
    
