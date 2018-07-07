#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 21:36:14 2018

@author: sunnykarira
"""

# SLR = Fitting a line that best fits the data (linearly)
# y = b0 + b1 * (x)
# y = So this is the dependent variable the dependent variable something you're trying to explain
# x = You're assuming that it is causing the depend variable to change.
# b1 = A unit change in x how that affects a unit change in Y.
# b0 = Constant (The point where regression line crosses vertical axis) i.e 
# when experience = 0 the what will be the salary.

# For this example Salary = b0 + b1 * (Experience)

# How does SLR finds the best fit line?
# This is done by OLS (Ordinary least squares method)
# SUM(yi - yi(bar))^2 -> MINIMUM


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting data into test set and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state= 0)

# No feature scaling ( library will take care for us in SLR)

# Fitting SLR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results
y_pred = regressor.predict(X_test)


# Visualize the results

# Training Set Visualization
plt.scatter(X_train, y_train, color = "red") # Dataset points
plt.plot(X_train, regressor.predict(X_train), color = "blue") # Regression line for training set
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Test Set Visualization
plt.scatter(X_test, y_test, color="red") # Just changed the scatter plot of points to test set points
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()









