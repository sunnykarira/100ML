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

# Preparing the data

# Take care of missing data

from sklearn.preprocessing import Imputer
# cmd + I to inspect (MAC)
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0)
# Fitting the imputer to matrix 
imputer = imputer.fit(X[:, 1:3])
X[:, 1: 3] = imputer.transform(X[:, 1:3])

# Took mean of columns and replaced nan values
'''([['France', 44.0, 72000.0],
       ['Spain', 27.0, 48000.0],
       ['Germany', 30.0, 54000.0],
       ['Spain', 38.0, 61000.0],
       ['Germany', 40.0, 63777.77777777778],
       ['France', 35.0, 58000.0],
       ['Spain', 38.77777777777778, 52000.0],
       ['France', 48.0, 79000.0],
       ['Germany', 50.0, 83000.0],
       ['France', 37.0, 67000.0]])'''


# Categorical Variables
# Country and Purchased
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

'''So the problem is still the same machine learning models are based on equations and that's good that

we replaced the text by numbers so that we can include the numbers in the equations.

However since one is greater than zero and two is greater than one the equations in the model will think

that Spain has a higher value than Germany and France and Germany has a higher value than France.

And that's not the case.

These are actually three categories and there is no relational order between the three.'''

# We use dummy encoding to solve this problem
onehotEncoder  = OneHotEncoder(categorical_features=[0])
X = onehotEncoder.fit_transform(X).toarray()

'''
1	0	0	44	72000
0	0	1	27	48000
0	1	0	30	54000
0	0	1	38	61000
0	1	0	40	63777.8
1	0	0	35	58000
0	0	1	38.7778	52000
1	0	0	48	79000
0	1	0	50	83000
1	0	0	37	67000
'''

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
'''[0, 1, 0, 0, 1, 1, 0, 1, 0, 1]''' # No/Yes

# Splitting data into training set and test set

from sklearn.model_selection import train_test_split
#Putting 20% data in test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Normalizing the data
'''We have two variables age and salary.

So you can picture age as the x coordinate and the salary as the y coordinates.

And in the machine running those equations some Euclidean distances between observation points.

For example this one and this one are computed based on these two coordinates.

And actually since the salary has a much wider range of values because it's going from zero to 100 k

the Euclidean distance will be dominated by the salary'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # Since it fitted to X_train we do not fit again for test

'''
As you can see the dummy variables takes values either 0 or 1.

So do we need to scale that because it seems to be already scale right.

So if you ask this question to Google you will find several answers you will find several teams for

that.

Some will say that no you don't need to scale these dummy variables.

Some say that yes you need to do it because you want some accuracy in your predictions.

And if you're interested in my opinion I would say that it depends on the context.

It depends on how much you want to keep interpretation in your models.

Because if we scale this it will be good because everything will be on the same scale we will be happy

with that and it will be good for our predictions but who will lose the interpretation of knowing which

observations belongs to which country etc..
'''









