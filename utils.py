import os
import torch
import tsl
import numpy as np
import pandas as pd

def print_matrix(matrix):
    return pd.DataFrame(matrix)

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

def getNumSamples(data, date):
    for i in range (len(data)):
        if data.loc[i, "dateTime"] == date:
            numTestData += 1
    print("Number of samples from ", date, " = ", numTestData)