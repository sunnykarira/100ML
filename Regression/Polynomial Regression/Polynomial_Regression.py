e#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 22:24:19 2018

@author: sunnykarira
"""

# SLR : y = b0 + b1 * (x)
# MLR : y = b0 + b1 *(x1) + b2 *(x2) + b3 * (x3) .... bn * (xn)
# PLR : y = b0 + b1 * (x1) + b2 * (x1)^2 + b3 * (x1)^3

# When to use PLR - When data is varying in parabolic form ( case for y = b0 + b1 * (x1) + b2 * (x1) ^ 2)

# Why is PLR called linear
# Linear , Non Linear refers to the coefficients and not the x powered values

# Important => PLR is special case of MLR

# Case  Predict if a new hire is true or making bluff according to his level and salary
# Level = 6.5 , Salary = 160k

# PR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


"""from sklearn.model_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed = 0)"""

# Fitting LR to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting PR to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing LR results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth / Bluff (LR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# Visualizing PR results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth / Bluff (PR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting with LR
lin_reg.predict(6.5)  # array([330378.78787879])

# Predicting with PR
lin_reg_2.predict(poly_reg.fit_transform(6.5))  # array([158862.45265153]) # Truth



