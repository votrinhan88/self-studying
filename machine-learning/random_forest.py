import torch
import pandas as pd
import numpy as np
from decision_tree import Node, DecisionTreeClassifier
from typing import List

NUM_TREES = 10
BAGGING_SIZE = 0.4
MAX_DEPTH = 2

class DecisionTreeInRFClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth = 2):
        super().__init__(max_depth = max_depth)
        self.feature_picks = None

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        pass

    def grow(self, node:Node, X_node:torch.Tensor, y_node:torch.Tensor):
        # Feature selection (sqrt of node's num features)
        # Then grow normally
        pass

class RandomForestClassifier():
    def __init__(self, num_trees = NUM_TREES, bagg_ratio = BAGGING_SIZE, max_depth = MAX_DEPTH):
        self.num_trees  = num_trees
        self.bagg_ratio = bagg_ratio
        self.bagg_size  = None
        self.max_depth  = max_depth

    def __repr__(self):
        return f"RF with {self.num_trees} trees, {self.bagg_ratio*100:.1f}% bagging"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        num_examples = X_train.size()[0]
        self.bagg_size = torch.tensor(self.bagg_ratio*num_examples).ceil().type(torch.int).item()

        # Create forest
        self.trees = [DecisionTreeClassifier(max_depth = self.max_depth)]*self.num_trees
        for tree in self.trees:
            # Bagging
            bag_picks = torch.randint(low = 0, high = num_examples, size = [self.bagg_size])
            X_bag, y_bag = X_train[bag_picks], y_train[bag_picks]
            tree.fit(X_bag, y_bag)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        pred = torch.zeros([input.size()[0], self.num_trees], dtype = torch.int)
        for tree_id, tree in enumerate(self.trees):
            pred[:, tree_id] = tree.forward(input).squeeze()

        yhat = pred.max(dim = 1)[0].unsqueeze(dim = 1)
        return yhat
        
if __name__ == '__main__':
    # Load and process data
    data = pd.read_csv('./data/iris.csv')
    data_size = len(data)
    # Map Iris variety to numerical label
    data.loc[data['variety'] == 'Versicolor', 'variety'] = 0
    data.loc[data['variety'] == 'Virginica', 'variety'] = 1
    data.loc[data['variety'] == 'Setosa', 'variety'] = 2
    # Shuffle data, split to train and test set
    data = data.iloc[torch.randperm(data_size), :]
    X_train, y_train = (torch.tensor(data.iloc[0:round(data_size*0.8), :-1].values.astype(np.float32)),
                        torch.tensor(data.iloc[0:round(data_size*0.8), -1].values.astype(np.int64)).unsqueeze(dim = -1))
    X_test, y_test = (torch.tensor(data.iloc[round(data_size*0.8):, :-1].values.astype(np.float32)),
                      torch.tensor(data.iloc[round(data_size*0.8):, -1].values.astype(np.int64)).unsqueeze(dim = -1))

    h = RandomForestClassifier(max_depth = 2)
    h.fit(X_train, y_train)
    yhat = h.forward(X_test)

    print(h)
    print(f'Accuracy = {((yhat == y_test).sum()/y_test.size()[0]).item():.4f}')