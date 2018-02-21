

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:42:44 2017
@author: adhingra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

#splitting the dataset into training and test examples
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

# fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(len(X_train),1),y_train)

# predicting test results
y_pred = regressor.predict(X_test.reshape(len(X_test),1))

# visualizing train results with regression model
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train.reshape(len(X_train),1)))
plt.title('Exp vs Salary [Training Set]')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()

#visualising test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train.reshape(len(X_train),1)),color = 'blue')
#plt.plot(X_test,y_pred,color = 'black')
plt.title('Exp vs Salary [Test Set]')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About
