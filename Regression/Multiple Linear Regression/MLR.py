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
 