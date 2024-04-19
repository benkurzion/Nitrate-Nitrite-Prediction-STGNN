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

train_set, val_set, test_set = getSeasonalData()

targets = getTargets()

edge_index, edge_weight = getEdgeInformation()



# Translate data to pytorch geometric temporal

dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=(357/360)) # The last 3 day as testing


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

#CREATE SLIDING WINDOWS OF SIZE ~7 AND MAKE THEM ALL INTO STATICGRAPHTEMPORALSIGNALS
#DO THE SAME FOR VALIDATION AND TEST DATA??

for epoch in tqdm(range(50)):
    for time, snapshot in enumerate(train_dataset): #change this to sliding window use LMM
        y_hat = model(snapshot.x)
        cost = torch.mean((y_hat-snapshot.y)**2)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()


model.eval()
mse = 0
mape = 0
mae = 0
num_samples = test_dataset.snapshot_count

for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x)
    mse += torch.mean((y_hat - snapshot.y) ** 2).item()
    mape += torch.mean(torch.abs((snapshot.y - y_hat) / snapshot.y)).item()
    mae += torch.mean(torch.abs(y_hat - snapshot.y)).item()

mse /= num_samples
mape /= num_samples
mae /= num_samples

print("Mean Squared Error = ", mse)
print("Mean Absolute Error = ", mae)
print("Mean Absolute Percentage Error = ", (mape * 100))

