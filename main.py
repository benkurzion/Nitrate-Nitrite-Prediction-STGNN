from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric_temporal import STConv, StaticGraphTemporalSignal
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM, GCLSTM, TGCN

PATH_TO_NODE_FEATURES = "node_features.txt"
PATH_TO_ADJ_MAT = "adjMatrix.txt"


# Load the data
fullData = pd.read_csv(filepath_or_buffer=PATH_TO_NODE_FEATURES, sep=",")
adjMatrix = pd.read_csv(filepath_or_buffer=PATH_TO_ADJ_MAT, sep=",", header=None, index_col=None)


# Remove the 'observation' column from full data
fullData = fullData.drop(fullData.columns[0], axis=1)

# Change dateTime column from strings to datetime object
fullData['dateTime'] = pd.to_datetime(fullData['dateTime'], format='%Y-%m-%d')

# Save the node ID column in the adjacency matrix
#nodeIDs = adjMatrix.iloc[: , 0].to_numpy()

# Drop the column with node IDs
adjMatrix = adjMatrix.drop(adjMatrix.columns[0], axis=1)

#construct edge_index and edge_weight arrays
sourceNodes = []
sinkNodes = []
edge_weight = []
for i in range(0, adjMatrix.shape[0]):
    for j in range (0, adjMatrix.shape[1]):
        if adjMatrix.iat[i,j] != 0:
            sourceNodes.append(j)
            sinkNodes.append(i)
            edge_weight.append(adjMatrix.iat[i,j])

edge_index = np.array([sourceNodes, sinkNodes])
edge_weight = np.array(edge_weight)

features = np.zeros((360, adjMatrix.shape[0], 4)) #Time step, nodes, node features
nodeNumber = 0
for i in range (fullData.shape[0]):
    if i > 0 and fullData.iat[i, 0] != fullData.iat[i - 1, 0]: # Begin data entry for next node 
        nodeNumber += 1
    dayIndex = (fullData.iat[i, 1] - fullData.iat[0, 1]).days
    features[dayIndex][nodeNumber][0] = fullData.iat[i, 2]
    features[dayIndex][nodeNumber][1] = fullData.iat[i, 4] #skip col 3 because its the label
    features[dayIndex][nodeNumber][2] = fullData.iat[i, 5]
    features[dayIndex][nodeNumber][3] = fullData.iat[i, 6]


targets = np.zeros((360, adjMatrix.shape[0]))
nodeNumber = 0
for i in range (fullData.shape[0]):
    if i > 0 and fullData.iat[i, 0] != fullData.iat[i - 1, 0]: # Begin data entry for next node 
        nodeNumber += 1
    dayIndex = (fullData.iat[i, 1] - fullData.iat[0, 1]).days
    targets[dayIndex][nodeNumber] = fullData.iat[i, 3] # the nitrate+nitrite column = label



# Translate data to pytorch geometric temporal

dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.00833333333) # The last 3 days


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, filters)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        #print(type(h))
        h = F.relu(h)
        h = self.linear(h)
        return h
    
model = RecurrentGCN(node_features=4, filters=10)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(50)):
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = torch.mean((y_hat-snapshot.y)**2)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
total_mape = 0  # Initialize to accumulate total MAPE across snapshots
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    # Ensure non-zero ground truth for MAPE calculation
    y = snapshot.y.clone()  # Clone to avoid modifying original data
    y[y == 0] = 1e-7  # Replace zeros with a small value to avoid division by zero
    mape = torch.mean(torch.abs((y_hat - y) / y) * 100)  # Calculate MAPE
    total_mape += mape
average_mape = total_mape / (time + 1)  # Calculate average MAPE across snapshots
print("MAPE: {:.4f}%".format(average_mape))