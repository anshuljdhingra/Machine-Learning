# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:35:50 2017

@author: adhingra
"""
# Linear Discriminant Analysis - Feature Extraction
# Dimensionality Reduction

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

# splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# analysing lda 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)

# fitting logistic regression model to the dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

# predicting test set results
y_pred = classifier.predict(X_test)

# making confusion matrix, accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

# visualing training set results
from matplotlib.colors import ListedColormap
X1,X2 = np.meshgrid(np.arange(start = X_train[:,0].min()-1, stop = X_train[:,0].max() +1, step = 0.01),
                    np.arange(start = X_train[:,1].min() -1, stop = X_train[:,1].max() +1, step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'blue', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train == j, 0], X_train[y_train == j, 1],
                c = ListedColormap(('white', 'yellow', 'brown'))(i), label =j)
plt.legend()
plt.title('PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
    
# visualising test set results
from matplotlib.colors import ListedColormap
X1,X2 = np.meshgrid(np.arange(start = X_test[:,0].min()-1, stop = X_test[:,0].max()+1, step =0.01),
                    np.arange(start = X_test[:,1].min()-1,stop = X_test[:,1].max()+1, step =0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             aplha = 0.6, cmap = ListedColormap(('brown', 'violet', 'black')))
plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_test[y_test == j, 0], X_test[y_test == j, 1],
                c= ListedColormap(('white','red', 'yellow'))(i), label = j)
plt.legend()
plt.title('PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
