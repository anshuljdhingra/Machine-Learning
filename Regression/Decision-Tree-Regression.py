# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:03:25 2017

@author: adhingra
"""
# Decision Tree Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# splitting into test and training sets not reqd because of small dataset
# feature scaling not reqd because its not based on Euclidian Distance

# fitting dataset into Decision Tree Regression Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# predicting values for unknown x values
y_pred = regressor.predict(6.5)

# visualising Decision Tree Regression Results **provide continuous result**
#plt.scatter (X,y,color='red')
#plt.plot(X,regressor.predict(X), color='blue')
#plt.xlabel('Level')
#plt.ylabel('Salary')
#plt.title('Level vs Salary')
#plt.show()

# visualizing Decision Tree Regression Results (for Smoother curves and higher resolution) 
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Level vs Salary')
plt.show()
