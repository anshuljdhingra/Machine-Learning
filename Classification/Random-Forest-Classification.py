# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:55:47 2017

@author: adhingra
"""
# Random Forest Classification (using feature scaling for high resolution graphs)

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

# splitting dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state =0)

# feature scaling for the dependent varaibles
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting Random Forest Classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

# predicting test set results
y_pred = classifier.predict(X_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# visualing training set results
from matplotlib.colors import ListedColormap
X1,X2 = np.meshgrid(np.arange(start = X_train[:,0].min() -1, stop = X_train[:,0].max() +1, step = 0.01),
                    np.arange(start = X_train[:,1].min() -1, stop = X_train[:,1].max() +1, step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             cmap = ListedColormap(('yellow', 'pink')), alpha = 0.88)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train == j, 0], X_train[y_train == j, 1],
                c = ListedColormap(('brown', 'blue'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Random Forest - Training Set')
plt.legend()
plt.show()

# visualising test set results
from matplotlib.colors import ListedColormap
X1,X2 = np.meshgrid(np.arange(start = X_test[:,0].min() -1, stop = X_test[:,0].max() +1, step = 0.01),
                    np.arange(start = X_test[:,1].min() -1, stop = X_test[:,1].max() +1, step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             cmap = ListedColormap(('yellow', 'pink')), alpha = 0.88)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_train[y_test == j, 0], X_train[y_test == j, 1],
                c = ListedColormap(('brown', 'blue'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Random Forest - Training Set')
plt.legend()
plt.show()
