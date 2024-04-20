from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, PemsBayDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric_temporal import STConv, StaticGraphTemporalSignal
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM, GCLSTM, TGCN, TGCN2, A3TGCN, DCRNN
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
import datetime
from datetime import datetime
from utils import *



'''
40 training 40 validation 20 testing
batch of training data has sliding window. 
use all of the features - past 5 days and altidude and discharge. date doenst matter
split the data using percentages of seasons. (uniform sampling?)
only scale if we need to

search up for stgnn layers
stgcn? instead

'''

train_set, train_lab, val_set, val_lab, test_set, test_lab = getSeasonalData()

edge_index, edge_weight = getEdgeInformation()


# Translate data to pytorch geometric temporal

train_dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=train_set, targets=train_lab)
validation_dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=val_set, targets=val_lab)
test_dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=test_set, targets=test_lab)



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1) #these are hyperparameters we will tune with validation data. loop through values for out channels and K
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
model = RecurrentGCN(node_features= 8) 


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()


'''for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()'''

window_size = 7
for epoch in tqdm(range(200)):
    for index, start in enumerate(train_dataset):
        if index >= train_dataset.snapshot_count:
            break
        #print("index = ", index, " start = ", start)
        cost = 0
        performBackPropagation = True
        window_length = 1
        for _, snapshot in enumerate(train_dataset, start=index): # Loop over window
            if window_length == window_size: # check if reached window_size 
                performBackPropagation = True
                break 
            if snapshot.x[0][7] - start.x[0][7] > window_size: # check if window goes over 2 different seasons
                performBackPropagation = False
                break 
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2) # aggregate MSE for every element in the window
            window_length += 1
        if performBackPropagation:
            cost = cost / (window_size)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()


model.eval()
mse = 0
mape = 0
mae = 0
num_samples = test_dataset.snapshot_count

for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    mse += torch.mean((y_hat - snapshot.y) ** 2).item()
    mape += torch.mean(torch.abs((snapshot.y - y_hat) / snapshot.y)).item()
    mae += torch.mean(torch.abs(y_hat - snapshot.y)).item()

mse /= num_samples
mape /= num_samples
mae /= num_samples

print("Mean Squared Error = ", mse)
print("Mean Absolute Error = ", mae)
print("Mean Absolute Percentage Error = ", (mape * 100))

