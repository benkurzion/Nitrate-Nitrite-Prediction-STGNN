from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric_temporal import STConv, StaticGraphTemporalSignal
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM, GCLSTM, TGCN, TGCN2, A3TGCN
from sklearn.preprocessing import StandardScaler


PATH_TO_NODE_FEATURES = "node_features.txt"
PATH_TO_ADJ_MAT = "prunedEdgesAdjMatrix"



'''
Change the training such that we train with sliding window d days back to predict
remove longitude latitude. 
use nitrate+nitrite. features are historical values and label is current value. 
add in elevation as another variable to see if prediction accuracy improves
could design function for edge weights which uses elevation and location as inputs
discharge may not be necessary. 
scale features
'''

# Load the data
fullData = pd.read_csv(filepath_or_buffer=PATH_TO_NODE_FEATURES, sep=",")
adjMatrix = pd.read_csv(filepath_or_buffer=PATH_TO_ADJ_MAT, sep=",", header=None, index_col=None)


# Change dateTime column from strings to datetime object
fullData['dateTime'] = pd.to_datetime(fullData['dateTime'], format='%Y-%m-%d')

#Scale the "nitrite+nitrate" column 
scaler = StandardScaler()
#scaler.fit(fullData[["nitrite+nitrate"]])
#fullData["nitrite+nitrate"] = scaler.transform(fullData[["nitrite+nitrate"]])


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


'''features = np.zeros((360, adjMatrix.shape[0], 4)) #Time step, nodes, node features
nodeNumber = 0
for i in range (fullData.shape[0]):
    if i > 0 and fullData.iat[i, 1] != fullData.iat[i - 1, 1]: # Begin data entry for next node 
        nodeNumber += 1
    dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days
    features[dayIndex][nodeNumber][0] = fullData.iat[i, 3]
    features[dayIndex][nodeNumber][1] = fullData.iat[i, 5] #skip col 4 because its the label
    features[dayIndex][nodeNumber][2] = fullData.iat[i, 6]
    features[dayIndex][nodeNumber][3] = fullData.iat[i, 7]
    
targets = np.zeros((360, adjMatrix.shape[0]))
nodeNumber = 0
for i in range (fullData.shape[0]):
    if i > 0 and fullData.iat[i, 1] != fullData.iat[i - 1, 1]: # Begin data entry for next node 
        nodeNumber += 1
    dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days
    targets[dayIndex][nodeNumber] = fullData.iat[i, 4] # the nitrate+nitrite column = label'''


features = np.zeros((353, adjMatrix.shape[0], 7)) # Time step, nodes, 7 previous "nitrite+nitrate" concentrations
nodeNumber = 0
for i in range (7, fullData.shape[0]):
    if i > 0 and fullData.iat[i, 1] != fullData.iat[i - 1, 1]: # Begin data entry for next node 
        nodeNumber += 1
    dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days - 7
    # Fill in previous 7 "nitrite+nitrate" concentrations (scaled)
    for j in range (7):
        features[dayIndex][nodeNumber][j] = fullData.iat[i - j - 1, 4]
    
targets = np.zeros((353, adjMatrix.shape[0]))
nodeNumber = 0
for i in range (7, fullData.shape[0]):
    if i > 0 and fullData.iat[i, 1] != fullData.iat[i - 1, 1]: # Begin data entry for next node 
        nodeNumber += 1
    dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days - 7
    targets[dayIndex][nodeNumber] = fullData.iat[i, 4] # the "nitrite+nitrate" column = label


# Translate data to pytorch geometric temporal

dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=(357/360)) # The last 3 day as testing


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
model = RecurrentGCN(node_features=7, filters=32)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(50)):
    for time, snapshot in enumerate(train_dataset): #change this to sliding window use LMM
        #print(snapshot.x.shape)

        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = torch.mean((y_hat-snapshot.y)**2)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()

cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
