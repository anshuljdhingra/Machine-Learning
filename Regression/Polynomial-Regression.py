# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:14:28 2017

@author: adhingra
"""

# Polynmial Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values   # to make matrix, provide upper bound
y = dataset.iloc[:,2].values

# we won't be splitting dataset into test and training sets as the obersvations are a few

# fitting linear regression model to dataset
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)

# fitting polynomial regression model to dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree=4)
X_poly = polyreg.fit_transform(X)
linreg2 = LinearRegression()
linreg2.fit(X_poly,y)

# visualising linear regression results
plt.scatter(X,y,color='red')
plt.plot(X, linreg.predict(X), color = 'blue')
plt.title('Level vs Salary - Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# visualising polynomial regression results
plt.scatter(X,y,color='red')
plt.plot(X,linreg2.predict(polyreg.fit_transform(X)),color='blue')
plt.title('Level vs Salary - Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# graph smoothening and higher resolution for visualising polynomial regression
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,linreg2.predict(polyreg.fit_transform(X_grid)),color = 'blue')
plt.title('Level vs Salary - Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# predicitng a new result with linear regrssion
y_pred_lin = linreg.predict(6.5)

# predicting a new result with polynomial regression
y_pred_poly = linreg2.predict(polyreg.fit_transform(6.5))


