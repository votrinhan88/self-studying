import pandas as pd
import numpy as np
import torch
    
def get_iris():
    # Load and process data
    data = pd.read_csv('./data/iris.csv')
    data_size = len(data)
    # Map Iris variety to numerical label
    data.loc[data['variety'] == 'Versicolor', 'variety'] = 0
    data.loc[data['variety'] == 'Virginica', 'variety'] = 1
    data.loc[data['variety'] == 'Setosa', 'variety'] = 2
    # Shuffle data, split to train and test set
    data = data.iloc[torch.randperm(data_size), :]
    X, y = (torch.tensor(data.iloc[:, 0:-1].to_numpy()),
            torch.tensor(data.iloc[:, [-1]].to_numpy(dtype = np.int64)))
    num_trains = round(data_size*0.8)
    X_train, y_train = X[0:num_trains], y[0:num_trains]
    X_test, y_test = X[num_trains:], y[num_trains:]

    return X_train, y_train, X_test, y_test