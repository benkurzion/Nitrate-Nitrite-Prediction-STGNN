import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import pandas as pd
import torch


PATH_TO_NODE_FEATURES = "node_features.txt"

# Load the data
fullData = pd.read_csv(filepath_or_buffer=PATH_TO_NODE_FEATURES, sep=",")


nodeByNodeFeatures = []
nodeByNodeLabels = []
currentNodeFeatures = []
currentNodeLabels = []
for i in range (fullData.shape[0]):
    if i > 0 and fullData.iat[i, 1] != fullData.iat[i - 1, 1]: # Begin data entry for next node 
        nodeByNodeFeatures.append(currentNodeFeatures)
        currentNodeFeatures = []
        nodeByNodeLabels.append(currentNodeLabels)
        currentNodeLabels = []
    currentNodeFeatures.append(fullData.iloc[i, [3,5,6,7]].to_numpy(dtype=float))
    currentNodeLabels.append(fullData.iloc[i, 4])

predictions = []
# Train autoregressive model on each node
for i in range(len(nodeByNodeFeatures)):
    #grab the last day as the test day
    testFeatures = nodeByNodeFeatures[i][-1]
    testFeatures = np.array([testFeatures])
    #remove the test data
    nodeByNodeFeatures[i] = nodeByNodeFeatures[i][:-1]

    #grab the last label as the test label
    testLabel = nodeByNodeLabels[i][-1]
    #remove the test label
    nodeByNodeLabels[i] = nodeByNodeLabels[i][:-1]

    #fit the model with the training days
    model = AutoReg(endog=nodeByNodeLabels[i], exog=nodeByNodeFeatures[i], lags=1)
    model = model.fit()

    #predict the test day
    predictions.append(model.predict(start=len(nodeByNodeFeatures[i]), end=len(nodeByNodeFeatures[i]), exog_oos=testFeatures)[0]) #predicting one step into the future

print(predictions)
