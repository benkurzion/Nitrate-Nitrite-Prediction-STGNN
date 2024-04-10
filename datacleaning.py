import os
import zipfile
import numpy as np
import torch
import pandas as pd


def getNodeElevation(nodeID, fullData):
    for i in range(fullData.shape[0]):
        if fullData.iat[i, 0] == nodeID:
            return fullData.iat[i, 6]

def cleanAdjacencyMatrix(pathToAdjMatrix, pathToNodeFeatures):
    # For any pair of nodes with A[node1][node2] = A[node2][node1] != 0
    # If the nodes' elevation are not equal, change the edge from lower elevation -> higher elevation to 0
    # We use Newman's notation: 
                        #  edge(i,j) = 0 if no edge from i --> j
                        #  edge(i,j) = {1,2} if edge from j --> i

    # Load data
    fullData = pd.read_csv(filepath_or_buffer=pathToNodeFeatures, sep=",")
    adjMatrix = pd.read_csv(filepath_or_buffer=pathToAdjMatrix, sep=",", header=None, index_col=None)
    #Remove the 'observation' column from full data
    fullData = fullData.drop(fullData.columns[0], axis=1)

    # Save the node ID column in the adjacency matrix
    nodeIDs = adjMatrix.iloc[: , 0].to_numpy()

    # Drop the column with node IDs
    adjMatrix = adjMatrix.drop(adjMatrix.columns[0], axis=1)

    for i in range(0, adjMatrix.shape[0]):
        for j in range (0, adjMatrix.shape[1]): #ignore the node_id column
            if adjMatrix.iat[i,j] != 0:
                elevationI = getNodeElevation(nodeIDs[i], fullData)
                elevationJ = getNodeElevation(nodeIDs[j], fullData)
                if elevationI > elevationJ: #delete edge from j --> i
                    adjMatrix.iat[i,j] = 0
                elif elevationI < elevationJ: #delete edge from i --> j
                    adjMatrix.iat[j,i] = 0


    # Save the adjacency matrix
    adjMatrix.to_csv('prunedEdgesAdjMatrix', index=False, header=False)

cleanAdjacencyMatrix(pathToAdjMatrix="C:\\Users\\benku\\College\\2023-24\\Machine Learning 446\\Research Project\\data\\adjMatrix.txt", pathToNodeFeatures="C:\\Users\\benku\\College\\2023-24\\Machine Learning 446\\Research Project\\data\\node_features.txt")