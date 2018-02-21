# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:45:47 2017

@author: adhingra
"""

# Random Forest Regression Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# splitting dataset into training and test set not reqd as dataset is smaller
# feature scaling is not required as Random Forest is not based on Euclidean Distance

# Fitting dataset into Random Forest Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X,y)

# predicting a new result
y_pred = regressor.predict(6.5)

# visualising Decision Tree Regression Results **provide continuous result**
#plt.scatter (X,y,color='red')
#plt.plot(X,regressor.predict(X), color='blue')
#plt.xlabel('Level')
#plt.ylabel('Salary')
#plt.title('Level vs Salary')
#plt.show()

# visualizing Random Forest Regression results (for high resolution and smooth curves)
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Level vs Salary')
plt.show()
