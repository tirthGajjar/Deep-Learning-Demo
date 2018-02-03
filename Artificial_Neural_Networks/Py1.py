# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 19:05:25 2018

@author: Tirth Gajjar
"""
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imporrting the dataset
dataset = pd.read_csv("D:\Deep Learning\Artificial_Neural_Networks\Churn_Modelling.csv");
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEnocder_X_1 = LabelEncoder();
X[:,1] = labelEnocder_X_1.fit_transform(X[:,1])
labelEnocder_X_2 = LabelEncoder();
X[:,2] = labelEnocder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state =0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

