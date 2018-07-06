#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:10:31 2018

@author: sunnykarira
"""

# Data Preprocessing Templates

# Importing the libraries
import numpy as np # For mathematical calculation
import matplotlib.pyplot as plt # For plotting charts
import pandas as pd  # To import and manage datasets

# Importing the dataset
# To import Data.csv file locate to the path of Data.csv and run F5
dataset = pd.read_csv('Data.csv')
# The dataset can be seen in variable explorer => Index start as 0

# Find out independent and dependent variable vectors

### Independent Variables
# iloc[Take all rows, Take all columns except last column]
X = dataset.iloc[:,:-1].values

''' ([['France', 44.0, 72000.0],
       ['Spain', 27.0, 48000.0],
       ['Germany', 30.0, 54000.0],
       ['Spain', 38.0, 61000.0],
       ['Germany', 40.0, nan],
       ['France', 35.0, 58000.0],
       ['Spain', nan, 52000.0],
       ['France', 48.0, 79000.0],
       ['Germany', 50.0, 83000.0],
       ['France', 37.0, 67000.0]])'''

### Dependent Variable => Purchased / Not Purchased
# iloc[Take all rows, Take last column]
y = dataset.iloc[:, 3].values

'''array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'])'''


