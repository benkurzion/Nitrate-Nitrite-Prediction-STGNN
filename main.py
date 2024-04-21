from process_data import preprocess_data_for_stgnn
from utils import PairDataset
from utils import compute_metrics
from utils import get_normalized_adj
from torch.utils.data import DataLoader
from stgnn import ProposedSTGNN
from stgnn_trainer import ProposedSTGNNTrainer
import torch
import numpy as np
import pandas as pd
import datetime
import csv

# Global variables
PATH_TO_NODE_FEATURES = "node_features.txt"
PATH_TO_ADJ_MAT = "prunedEdgesAdjMatrix"
SPLIT_DATE = datetime.datetime(2022, 3, 12)
TIME_STEPS = 7
BATCH_SIZE = 7
EPOCHS = 50
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')


# get data and order it as (n_samples, n_timeseries)
fullData = pd.read_csv(filepath_or_buffer=PATH_TO_NODE_FEATURES, sep=",")
#adjMatrix = pd.read_csv(filepath_or_buffer=PATH_TO_ADJ_MAT, sep=",", header=None, index_col=None)

adj = np.loadtxt(PATH_TO_ADJ_MAT, delimiter=",")

# Change dateTime column from strings to datetime object
fullData['dateTime'] = pd.to_datetime(fullData['dateTime'], format='%Y-%m-%d')

# removing rows which have measurements between '2021-04-01' and '2021-04-06' inclusive
fullData = fullData[(fullData['dateTime'] > '2021-04-06')]


# Convert the 'nitrite+nitrate' column to float
fullData['nitrite+nitrate'] = fullData['nitrite+nitrate'].astype(np.float32)


# Perform data extraction
df = fullData.pivot(columns='site_no', index='dateTime', values='nitrite+nitrate')



X_train, y_train, X_test, y_test, _, _, scaler = preprocess_data_for_stgnn(df, SPLIT_DATE, TIME_STEPS)

X_train = torch.tensor(X_train).unsqueeze(-1)
y_train = torch.tensor(y_train).unsqueeze(-1)
X_test = torch.tensor(X_test).unsqueeze(-1)
y_test = torch.tensor(y_test).unsqueeze(-1)
n_test_samples = len(y_test)

train_dl = DataLoader(PairDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# Fix formatting
adj_mx = adj.astype(np.float32)

adj_mx = get_normalized_adj(adj_mx)
adj = torch.tensor(adj_mx)

model = ProposedSTGNN(n_nodes=adj.shape[0],
                      time_steps=TIME_STEPS,
                      predicted_time_steps=1,
                      in_channels=X_train.shape[3],
                      spatial_channels=32,
                      spatial_hidden_channels=16,
                      spatial_out_channels=16,
                      out_channels=16,
                      temporal_kernel=3,
                      drop_rate=0.2).to(device=device)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = ProposedSTGNNTrainer(model,
                               train_dl,
                               X_test,
                               adj,
                               scaler,
                               loss_func,
                               optimizer,
                               device,
                               callbacks=None,
                               raw_test=df.iloc[-(n_test_samples + 1):].values)

#history = trainer.train(EPOCHS)

predictions = trainer.predict()

pd.DataFrame(predictions,
             index=df.iloc[-n_test_samples:].index).head()

# Compute RMSE of test dataset
m, m_avg = compute_metrics(df.iloc[-n_test_samples:], predictions, metric='mape')
print("MAPE = ", m_avg)

m, m_avg = compute_metrics(df.iloc[-n_test_samples:], predictions, metric='mae')
print("MAE = ", m_avg)