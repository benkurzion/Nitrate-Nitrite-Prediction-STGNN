from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
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
scaler.fit(fullData[["nitrite+nitrate"]])
fullData["nitrite+nitrate"] = scaler.transform(fullData[["nitrite+nitrate"]])

#Scale the "water discharge" column
scaler = StandardScaler()
scaler.fit(fullData[["water discharge"]])
fullData["water discharge"] = scaler.transform(fullData[["water discharge"]])

#Scale the "altitude" column
scaler = StandardScaler()
scaler.fit(fullData[["altitude"]])
fullData["altitude"] = scaler.transform(fullData[["altitude"]])



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




# Following code constructs feature matrix for the latter 360 - daysPrior days using the previous daysPrior labels as features.
daysPrior = 5
features = np.zeros((360 - daysPrior, adjMatrix.shape[0], daysPrior + 1)) # Time step, nodes, daysPrior previous "nitrite+nitrate" concentrations and water discharge and altitude
nodeNumber = -1
i = 0
while i < (fullData.shape[0]):
    if i == 0 or fullData.iat[i, 1] != fullData.iat[i - 1, 1]: # Begin data entry for next node 
        nodeNumber += 1
        i = i + daysPrior 
    dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days - daysPrior
    # Fill in previous daysPrior "nitrite+nitrate" concentrations (scaled)
    for j in range (daysPrior):
        features[dayIndex][nodeNumber][j] = fullData.iat[i - j - 1, 4]
    features[dayIndex][nodeNumber][daysPrior] = fullData.iat[i, 3] #water discharge feature (scaled)
    #features[dayIndex][nodeNumber][daysPrior + 1] = fullData.iat[i, 7] #altitude feature (scaled)
    i += 1

    
targets = np.zeros((360 - daysPrior, adjMatrix.shape[0])) # Time step, nodes
nodeNumber = -1
i = 0
while i < (fullData.shape[0]):
    if i == 0 or fullData.iat[i, 1] != fullData.iat[i - 1, 1]: # Begin data entry for next node 
        nodeNumber += 1
        i = i + daysPrior 
    dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days - daysPrior
    targets[dayIndex][nodeNumber] = fullData.iat[i, 4] # the "nitrite+nitrate" column (scaled) = label
    i += 1

# Translate data to pytorch geometric temporal

dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=(357/360)) # The last 3 day as testing


class NN_Model(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(NN_Model, self).__init__()
        self.recurrent = TGCN(node_features, filters)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        #print(type(h))
        h = F.relu(h)
        h = self.linear(h)
        return h

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

model = RecurrentGCN(node_features=6, filters=32)


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
mse = 0
mape = 0
mae = 0
num_samples = test_dataset.snapshot_count

for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    #print("In Eval step = ", time, " predictions =\n", y_hat)
    mse += torch.mean((y_hat - snapshot.y) ** 2).item()
    mape += torch.mean(torch.abs((snapshot.y - y_hat) / snapshot.y)).item()
    mae += torch.mean(torch.abs(y_hat - snapshot.y)).item()

mse /= num_samples
mape /= num_samples
mae /= num_samples

print("Mean Squared Error = ", mse)
print("Mean Absolute Error = ", mae)
print("Mean Absolute Percentage Error = ", (mape * 100))





'''Following code constructs the feature matrix for all 360 days using all features (performs poorly)
features = np.zeros((360, adjMatrix.shape[0], 4)) #Time step, nodes, node features
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