# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:04:11 2017

@author: adhingra
"""
# K-Nearest Neighbors (K-NN)

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

# splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# feature scaling of the indepdendent variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting K-Nearest Neighbors Classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

# predicting test set results
y_pred = classifier.predict(X_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# visualizing training set results
from matplotlib.colors import ListedColormap
X_set,y_set = X_train, y_train
X1,X2 = np.meshgrid(np.arange(start = X_set.min()-1,stop=X_set.max()+1,step=0.01),
                    np.arange(start=X_set.min()-1,stop=X_set.max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             cmap=ListedColormap(('red','green')), alpha=0.75)
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],c = ListedColormap(('white','black'))(i), label=j)
plt.title("KNN Classifier - Training set")
plt.legend()
plt.xlabel('age')
plt.ylabel('salary')
plt.show()

# visualizing test set results
from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start=X1.min()-1,stop=X1.max()+1,step=0.01),
                    np.arange(start=X2.min()-1,stop=X2.max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                c=ListedColormap(('yellow','blue'))(i), label =j)
plt.title('KNN Classifier - Test Set')
plt.legend()
plt.xlabel('age')
plt.ylabel('salary')
plt.show()
