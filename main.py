import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *

'''
Training set = 70%
Validation set = 20%
Test set = 10%
'''

#TO BE FILLED IN IN SOME CAPACITY
dataFeatures = [] 
dataLables = []

trainFeatures, _, trainLabels, _ = train_test_split(X=dataFeatures, y=dataLables, test_size=0.7, random_state=1)
validationFeatures, testFeatures, validationLables, testLables = train_test_split(X=trainFeatures, y=trainLabels, test_size=0.66, random_state=1)

#Normalize data
scaler = StandardScaler()
trainFeatures = scaler.fit_transform(trainFeatures)
validationFeatures = scaler.transform(validationFeatures)
testFeatures = scaler.transform(testFeatures)


'''
Input width: how many samples are we looking back in time for prediction
    Each of our samples account for 15 minutes. So input width of 4 means training on the previous hour
Offset (shift): How far into the future we want our prediction to be
Label Width: How many time steps we want to predict
'''


window = WindowGenerator(input_width=20, label_width=4, shift=1, label_columns=['Nitrate/Nitrite Concentration'])
window.split_window = split_window