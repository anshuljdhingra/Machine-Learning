# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:46:21 2017

@author: adhingra
"""
# Support Vector Machine (SVM)

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

# splitting dataset into training and test results
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# feature scaling on the independent variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting SVM model to the training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)

# predicting test set results
y_pred = classifier.predict(X_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# visualizing training set results
from matplotlib.colors import ListedColormap
X1,X2 = np.meshgrid(np.arange(start = X_train[:,0].min()-1,stop = X_train[:,0].max()+1, step=0.01),
                    np.arange(start = X_train[:,1].min()-1,stop=X_train[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             cmap = ListedColormap(('red','green')),alpha =0.75)
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train == j,0],X_train[y_train == j,1],
                c = ListedColormap(('white','blue'))(i), label = j)
plt.title('SVM - Training Set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# vislualising test set results
from matplotlib.colors import ListedColormap
X1,X2 = np.meshgrid(np.arange(start = X_test[:,0].min() -1,stop = X_test[:,0].max() +1, step = 0.01),
                    np.arange(start = X_test[:,1].min() -1, stop = X_test[:,1].max() +1, step =0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             cmap = ListedColormap(('white', 'black')), alpha = .70)
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_test[y_test == j,0], X_test[y_test == j, 1],
                c = ListedColormap(('red', 'yellow'))(i), label = j)
plt.legend()
plt.title('SVM - Test set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
