import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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
    currentNodeFeatures.append(fullData.iloc[i, [3]].to_numpy(dtype=float))
    currentNodeLabels.append(fullData.iloc[i, 4])

predictions = []
testLabels = []
# Train autoregressive model on each node
for i in range(len(nodeByNodeFeatures)):
    #grab the last day as the test day
    testFeatures = nodeByNodeFeatures[i][-3:]
    #testFeatures = np.array([testFeatures])
    #remove the test data
    nodeByNodeFeatures[i] = nodeByNodeFeatures[i][:-3]

    #grab the last label as the test label
    testLabel = nodeByNodeLabels[i][-3:]
    testLabels.append(testLabel)
    #remove the test label
    nodeByNodeLabels[i] = nodeByNodeLabels[i][:-3]

    #fit the model with the training days
    model = AutoReg(endog=nodeByNodeLabels[i], exog=nodeByNodeFeatures[i], lags=1)
    model = model.fit()

    #predict the test day
    predictions.append(model.predict(start=len(nodeByNodeFeatures[i]), end=len(nodeByNodeFeatures[i]) + 2, exog_oos=testFeatures)) #predicting 3 steps into the future


# MAE
print("MAE = ")
print(np.mean(np.abs(np.array(predictions)-np.array(testLabels))))

# mean squared error
print("\nMSE = ")
print(np.mean(np.square(np.array(predictions)-np.array(testLabels))))

# mean absolute percentage error
sum_term= 0;
for i in range (len(testLabels)):
    for day in range (len (testLabels[0])):
        sum_term+= np.abs((testLabels[i][day]- predictions[i][day])/testLabels[i][day])
print("\nMAPE = ")
print(sum_term/len(testLabels)*100)

# create the scatter plot visual for the labels vs predictions for the third day
'''y = []
for i in range (len(testLabels)):
    y.append(testLabels[i][2])
yhat = []
for i in range (len(predictions)):
    yhat.append(predictions[i][2])
x = range(len(testLabels))

plt.scatter(x, y, label='Labels')
plt.scatter(x,yhat, label='Predictions')
plt.xlabel('Node')
plt.ylabel('nitrate/nitrite concentration')
plt.title('Autoregressive predictions for each node')
plt.legend()
plt.show()'''