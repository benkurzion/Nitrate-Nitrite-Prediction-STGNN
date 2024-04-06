import os
import torch
import tsl
import numpy as np
import pandas as pd
from tsl.datasets import MetrLA
from utils import *

# Presets (per TSL Ninjas)
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(edgeitems=3, precision=3)
torch.set_printoptions(edgeitems=2, precision=3)



dataset = MetrLA(root='./data')
print(dataset)
print(f"Sampling period: {dataset.freq}")
print(f"Has missing values: {dataset.has_mask}")
print(f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%")
print(f"Has exogenous variables: {dataset.has_covariates}")
print(f"Covariates: {', '.join(dataset.covariates.keys())}")
dataset.dataframe()

connectivity = dataset.get_connectivity(threshold=0.1, include_self=False, normalize_axis=1, layout="edge_index")
print("connectivity = \n", connectivity)

edge_index, edge_weight = connectivity

print(f'edge_index {edge_index.shape}:\n', edge_index)
print(f'edge_weight {edge_weight.shape}:\n', edge_weight)
'''
Data contains time series for __ sensors from March 6th, 2021 to March 1st, 2022. 
Each day contains 48 (plus or minus 1 or 2) readings from any of these sensors!

We are using March 6th, 2021 --> April 31st, 2021 as training data
We are using March 1st, 2023 as testing data
'''

# Load data
fullData = pd.read_csv("waterflowdataset.txt", sep=",")
adjMatrix = pd.read_csv("adjmatrix.txt", sep=",", header=None)

# Split data to features and label
labels = fullData["nitrite+nitrate"]
features = fullData[["site_no","dateTime","water discharge","latitude","longitude","altitude"]]


'''TODO: rework the adj matrix such that it is two array of length E number of edges
        array[0][i] = source, array[1][i] = sink for every edge i in the graph
        -> and a second array weights of length E number of edges
        weights[i] = weight of edge i for all i
        
        Figure out how the data should be formatted in relation to:
            1) where do the features go?
            2) where do the labels go?
'''