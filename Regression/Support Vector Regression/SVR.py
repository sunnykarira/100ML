#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:34:11 2018

@author: sunnykarira
"""
# SVR Intuition
"""
Support Vector Machines support linear and non linear regression
Instead of trying to fit the largest possible street  b/w two classes
while limiting margin violations, SVR tries to fit as many instances as possible 
on the street while limiting margin violations

The width of the street is controlled by a hyper parameter Epsilon

SVR performs linear regression in higher dimensional space

"""


""" 
In a classification problem , the vectors X(bar) are used to define a hyperplane
that seperates the two diff classes in your soln.

These vectors are used to perform linear regression. The vectors closest to the test point
are referred to as support vectors. We can evaluate our function so any could be closest to 
our test evaluation location.
"""

"""
##########################
BUILDING AN SVR

Collect a training set T = {X(bar), Y(bar)}
Choose a kernel and it's parameters as well as regularization needed.
Form the coorelation matrix k(bar).
Train your machine to get contraciton coeff a(bar) = {a(i) }
Use those coeff and create an estimator  f(X(bar), a(bar), x(start)) = y (star)


-----
Prominent Kernel for SVM
* Gaussian - Regularization
* Noise

In SVR the goal is that the errors do not exceed the threshold.
#############################
"""

# SVR

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

"""
# SVR does not do feature scaling
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fitting the Regression Model to the dataset
# Create your regressor here
#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf") # Gaussian Kernel
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
inverse_y_pred = sc_y.inverse_transform(y_pred)


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



