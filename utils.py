import pandas as pd
import numpy as np
import datetime
from datetime import datetime

# Global variables
PATH_TO_NODE_FEATURES = "node_features.txt"
PATH_TO_ADJ_MAT = "adjmatrix.txt"
daysPrior = 5


def getSeasonalData():
    '''
    Returns 3 numpy arrays:
    1) Train set
    2) Validation set
    3) Test set

    All 3 arrays are organized as (Day x Node X Features)
    Train set = 40% of data,
    Validation set = 40% of the data,
    Test set = 20% of the data
    '''

    # Load the data
    fullData = pd.read_csv(filepath_or_buffer=PATH_TO_NODE_FEATURES, sep=",")
    adjMatrix = pd.read_csv(filepath_or_buffer=PATH_TO_ADJ_MAT, sep=",", header=None, index_col=None)

    # Change dateTime column from strings to datetime object
    fullData['dateTime'] = pd.to_datetime(fullData['dateTime'], format='%Y-%m-%d')

    # removing rows which have measurements between '2021-04-01' and '2021-04-06' inclusive
    fullData = fullData[(fullData['dateTime'] > '2021-04-06')]

    range_arr = get_season_range(fullData)

    # Following code constructs feature matrix for the latter 360 - daysPrior days using the previous daysPrior labels as features.
    # Update: Added day number to the feature set new shape is (days, nodes, 8)

    features = np.zeros((360 - daysPrior, adjMatrix.shape[0],
                         daysPrior + 3))  # Time step, nodes, daysPrior previous "nitrite+nitrate" concentrations and water discharge and altitude
    nodeNumber = -1
    i = 0
    print(fullData.iat[5, 2])
    while i < (fullData.shape[0]):
        if i == 0 or fullData.iat[i, 1] != fullData.iat[i - 1, 1]:  # Begin data entry for next node
            nodeNumber += 1
            i = i + daysPrior

        dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days - daysPrior
        print(dayIndex)
        for j in range(daysPrior):
            features[dayIndex][nodeNumber][j] = fullData.iat[i - j - 1, 4]  # historical nitrite+nitrate
        features[dayIndex][nodeNumber][daysPrior] = fullData.iat[i, 3]  # water discharge
        features[dayIndex][nodeNumber][daysPrior + 1] = fullData.iat[i, 7]  # altitude
        features[dayIndex][nodeNumber][daysPrior + 2] = dayIndex  # the day
        i += 1

    # partitioning features array based on season

    labels = getTargets()

    spring_arr_1 = features[: range_arr[0], :, :]

    sum_arr = features[range_arr[0]: range_arr[0] + range_arr[1], :, :]
    fall_arr = features[range_arr[1] + range_arr[0]: range_arr[0] + range_arr[1] + range_arr[2], :, :]
    win_arr = features[
              range_arr[0] + range_arr[1] + range_arr[2]: range_arr[0] + range_arr[1] + range_arr[2] + range_arr[3], :,
              :]

    # extracting the spring data for the year 2022
    spring_arr_2 = features[
                   range_arr[0] + range_arr[1] + range_arr[2] + range_arr[3]: range_arr[0] + range_arr[1] + range_arr[
                       2] + range_arr[3] + 25, :, :]

    # combining spring 2021 and spring 2022 to get final spring set
    final_spring_arr = np.concatenate((spring_arr_1, spring_arr_2), axis=0)

    spring_lab_1 = labels[: range_arr[0], :]
    sum_lab = labels[range_arr[0]: range_arr[0] + range_arr[1], :]
    fall_lab = labels[range_arr[1] + range_arr[0]: range_arr[0] + range_arr[1] + range_arr[2], :]
    win_lab = labels[
              range_arr[0] + range_arr[1] + range_arr[2]: range_arr[0] + range_arr[1] + range_arr[2] + range_arr[3],
              :]

    spring_lab_2 = labels[
                   range_arr[0] + range_arr[1] + range_arr[2] + range_arr[3]: range_arr[0] + range_arr[1] + range_arr[
                       2] + range_arr[3] + 25, :]
    final_spring_lab = np.concatenate((spring_lab_1, spring_lab_2), axis=0)





    # Extracting training, test and validation set

    # Split each season array into train, test, and validation sets along with their corresponding labels
    spring_train, spring_train_lab, spring_val, spring_val_lab , spring_test, spring_test_lab = split_season(final_spring_arr, final_spring_lab, )
    summer_train, summer_train_lab, summer_val, summer_val_lab, summer_test, summer_test_lab = split_season(sum_arr, sum_lab)
    fall_train, fall_train_lab, fall_val, fall_val_lab, fall_test, fall_test_lab = split_season(fall_arr, fall_lab)
    winter_train, winter_train_lab, winter_val, winter_val_lab, winter_test, winter_test_lab = split_season(win_arr, win_lab)

    # Combine the train, test, and validation sets from all seasons

    train_set = np.concatenate((spring_train, summer_train, fall_train, winter_train), axis=0)
    train_lab = np.concatenate((spring_train_lab, summer_train_lab, fall_train_lab, winter_train_lab), axis=0)
    val_set = np.concatenate((spring_val, summer_val, fall_val, winter_val), axis=0)
    val_lab = np.concatenate((spring_val_lab, summer_val_lab, fall_val_lab, winter_val_lab), axis = 0)
    test_set = np.concatenate((spring_test, summer_test, fall_test, winter_test), axis=0)
    test_lab = np.concatenate((spring_test_lab, summer_test_lab, fall_test_lab, winter_test_lab), axis = 0)


    return train_set, train_lab, val_set, val_lab, test_set, test_lab


def get_season_range(df):
    '''
    Helper method that generates the day ranges for each season
    '''

    date1_spr = '2021-04-01'
    date2_spr = '2021-05-31'

    # Convert the date strings to datetime objects
    sp_date1 = datetime.strptime(date1_spr, '%Y-%m-%d')
    sp_date2 = datetime.strptime(date2_spr, '%Y-%m-%d')

    season_range = []
    # Calculate the difference in days
    spring_days = (sp_date2 - sp_date1).days
    season_range.append(spring_days)

    date1_sum = '2021-06-01'
    date2_sum = '2021-08-31'

    sum_date1 = datetime.strptime(date1_sum, '%Y-%m-%d')
    sum_date2 = datetime.strptime(date2_sum, '%Y-%m-%d')

    season_range.append((sum_date2 - sum_date1).days)

    date1_fall = '2021-09-01'
    date2_fall = '2021-11-30'

    fall_date1 = datetime.strptime(date1_fall, '%Y-%m-%d')
    fall_date2 = datetime.strptime(date2_fall, '%Y-%m-%d')

    season_range.append((fall_date2 - fall_date1).days)

    date1_win = '2021-12-01'
    date2_win = '2022-02-28'

    win_date1 = datetime.strptime(date1_win, '%Y-%m-%d')
    win_date2 = datetime.strptime(date2_win, '%Y-%m-%d')

    season_range.append((win_date2 - win_date1).days)
    return season_range


def split_season(season_arr, season_lab):
    '''
    Args:
        numpy array to be split into train, validation and test sets

    Returns:
        train_set, val_set, test_set
    '''

    # Define the percentage for each set
    train_percent = 0.4
    val_percent = 0.4
    test_percent = 0.2  # implied by other two percentages

    num_days = season_arr.shape[0]
    train_size = int(num_days * train_percent)
    val_size = int(num_days * val_percent)

    train_set = season_arr[:train_size]
    train_lab = season_lab[: train_set]
    val_set = season_arr[train_size: train_size + val_size]
    val_lab = season_lab[train_size: train_size + val_size]
    test_set = season_arr[train_size + val_size:]
    test_lab = season_lab[train_size+val_size:]

    return train_set, train_lab, val_set, val_lab, test_set, test_lab


def getEdgeInformation():
    '''
    Returns the edge_index and edge_weight arrays.
    If edge_index[0][x] = 10 and edge_index[1][x] = 3 indicates an edge from node 3 to node 10
    edge_weight[x] = 2 means that the edge from 3 --> 10 has weight 2
    '''

    # Load the adj matrix from file
    adjMatrix = pd.read_csv(filepath_or_buffer=PATH_TO_ADJ_MAT, sep=",", header=None, index_col=None)

    # construct edge_index and edge_weight arrays
    sourceNodes = []
    sinkNodes = []
    edge_weight = []
    for i in range(0, adjMatrix.shape[0]):
        for j in range(0, adjMatrix.shape[1]):
            if adjMatrix.iat[i, j] != 0:
                sourceNodes.append(j)
                sinkNodes.append(i)
                edge_weight.append(adjMatrix.iat[i, j])

    edge_index = np.array([sourceNodes, sinkNodes])
    edge_weight = np.array(edge_weight)

    return edge_index, edge_weight


# need three target sets : train, test and val target sets


def getTargets():
    '''
    Returns the label array formatted as (Day x node) where targets[day][node] = label for node at day
    '''

    # Load the data
    fullData = pd.read_csv(filepath_or_buffer=PATH_TO_NODE_FEATURES, sep=",")
    fullData = fullData[fullData['dateTime'] > '2021-04-06']
    adjMatrix = pd.read_csv(filepath_or_buffer=PATH_TO_ADJ_MAT, sep=",", header=None, index_col=None)

    targets = np.zeros((360 - daysPrior, adjMatrix.shape[0]))  # Time step, nodes
    nodeNumber = -1
    i = 0
    while i < (fullData.shape[0]):
        if i == 0 or fullData.iat[i, 1] != fullData.iat[i - 1, 1]:  # Begin data entry for next node
            nodeNumber += 1
            i = i + daysPrior
        dayIndex = (fullData.iat[i, 2] - fullData.iat[0, 2]).days - daysPrior
        targets[dayIndex][nodeNumber] = fullData.iat[i, 4]  # the "nitrite+nitrate" column (scaled) = label
        i += 1

    return targets






