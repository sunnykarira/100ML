#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:13:12 2018

@author: sunnykarira
"""
# Problem Statment 
# Find out how a startup do based on (Reasearch Spends, Marketing etc. => Independent Variables, Profir => Dependent Variables)
 # i.e Any coorelation b/w profit and various spends

# MLR
# y = b0 + b1*(x1) + b2 *(x2) + b3 *(x3) .... + bn * (xn)
 
# Variables should be numeric and not categorical

# CAVEAT WITH LINEAR REGRESSION
# Assumptions of Linear Regression:
# --- Linearity

# --- Homoscedasicity
######  In statistics, a sequence or a vector of random variables is homoscedastic /ˌhoʊmoʊskəˈdæstɪk/ if all random variables in the sequence or vector have the same finite variance. 
######  This is also known as homogeneity of variance.
# --- Multivariate Normality
###### Something like a Bell curve in all dimensions.
# --- Independence of errors
# --- Lack of multicollinearity

# http://www.statisticssolutions.com/assumptions-of-linear-regression/

# See Dummy Variable Trap => Important
# Always omit one dummy variable
 
# P-Value
# https://www.mathbootcamps.com/what-is-a-p-value/
# https://www.wikihow.com/Calculate-P-Value
 
# Building a model 
# PDF
 
# All in
# Backward Elimination   => Fastest of all methods
# Forward Selection
# Bidirection Elimination
# All Possible Models (Score Comparison (R^2))
 

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

onehotEncoder  = OneHotEncoder(categorical_features=[3])
X = onehotEncoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting MLR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results
y_pred = regressor.predict(X_test)



 
 
 
 