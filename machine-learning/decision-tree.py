# Topic: Supervised, Classification

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
# from typing import List


# Load and process data
data = pd.read_csv('./data/iris.csv')
data_size = len(data)
# Map Iris variety to numerical label
data.loc[data['variety'] == 'Versicolor', 'variety'] = 0
data.loc[data['variety'] == 'Virginica', 'variety'] = 1
data.loc[data['variety'] == 'Setosa', 'variety'] = 2
# Shuffle data, split to train and test set
data = data.iloc[torch.randperm(data_size), :]
# data = data.iloc[0:20, :]
X_train, y_train = (torch.tensor(data.iloc[0:round(data_size*0.8), :-1].values.astype(np.float32)),
                    torch.tensor(data.iloc[0:round(data_size*0.8), -1].values.astype(np.int64)).unsqueeze(dim = -1))
X_test, y_test = (torch.tensor(data.iloc[round(data_size*0.8):, :-1].values.astype(np.float32)),
                  torch.tensor(data.iloc[round(data_size*0.8):, -1].values.astype(np.int64)).unsqueeze(dim = -1))

class Node():
    def __init__(self, depth = None, info = None, feature = None, threshold = None, children = [], path = ''):
        self.is_leaf = False
        # Pre-allocate
        self.depth     = depth
        self.info      = info
        self.feature   = feature
        self.threshold = threshold
        self.children  = children
        self.path      = path

        self.distr     = None

    # Magic method to represent variable
    def __repr__(self) -> str:
        # Magic attribute to get class name
        return f"{self.__class__.__name__} {self.path}, I = {self.info}, distr = {self.distr.numpy()}"

class DecisionTreeClassier():
    def __init__(self, max_depth = 4, method_info = 'entropy'):
        self.max_depth = max_depth
        self.method_info = method_info
        self.depth = 0

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        self.X_train = X_train
        self.y_train = y_train
        self.num_features = X_train.size()[1]
        self.num_classes = y_train.unique().size()[0]

        self.stump = Node(depth = 0)
        self.stump.distr = F.one_hot(y_train.squeeze(dim = 1), num_classes = self.num_classes).sum(dim = 0)
        self.stump.info = self.compute_info(y_train)
        self.grow(node = self.stump, X_node = X_train, y_node = y_train)

    def grow(self, node:Node, X_node:torch.Tensor, y_node:torch.Tensor):
        max_gain = self.find_best_split(node, X_node, y_node)
        if max_gain > 0:
            # Continue growing
            ## Re-split data by optimal feature and threshold for children nodes
            left_node  = Node(depth = node.depth + 1, path = node.path + 'L')
            right_node = Node(depth = node.depth + 1, path = node.path + 'R')
            self.depth = max(self.depth, node.depth + 1)

            node.children = [left_node, right_node]

            left_ind = X_node[:, node.feature] <= node.threshold
            X_left, y_left = X_node[left_ind], y_node[left_ind]
            X_right, y_right = X_node[~left_ind], y_node[~left_ind]
            
            # Leaf node:
            #  - Has a unique class distribution (contains only one class)
            #  - OR has reached max depth
            for (branch, X_branch, y_branch) in zip([left_node, right_node], [X_left, X_right], [y_left, y_right]):
                branch.distr = F.one_hot(y_branch.squeeze(dim = 1), num_classes = self.num_classes).sum(dim = 0)
                branch.info = self.compute_info(y_branch)
                if (len(y_branch.unique()) == 1) | (branch.depth == self.max_depth):
                    branch.is_leaf = True
                else:
                    self.grow(branch, X_branch, y_branch)
    
    def find_best_split(self, node:Node, X_node:torch.Tensor, y_node:torch.Tensor):
        max_gain = -torch.tensor(float('inf'))
        for feature in torch.arange(self.num_features):
            # thresholds = torch.linspace(start = X_node[:, feature].min(),
            #                             end = X_node[:, feature].max(),
            #                             steps = self.num_splits + 2)[1:self.num_splits+1]
            uniques = X_node[:, feature].sort()[0].unique()
            thresholds = (uniques[1:] + uniques[:-1])/2
            for threshold in thresholds:
                left, right = self.split(X_node, y_node, feature, threshold)
                gain = self.compute_gain(node, y_node, left, right)
                if gain > max_gain:
                    max_gain = gain
                    opt_feature = feature
                    opt_threshold = threshold
        
        # Update node with optimal split
        node.feature = opt_feature
        node.threshold = opt_threshold
        
        return max_gain

    def split(self, X_node:torch.Tensor, y_node:torch.Tensor, feature, threshold):
        left_ys  = y_node[X_node[:, feature] < threshold]
        right_ys = y_node[X_node[:, feature] >= threshold]
        return left_ys, right_ys

    def compute_gain(self, node:Node, y_node:torch.Tensor, left:torch.Tensor, right:torch.Tensor):
        left_info = self.compute_info(left)
        right_info = self.compute_info(right)
        gain = node.info - (left.size()[0]*left_info + right.size()[0]*right_info)/y_node.size()[0]
        return gain

    def compute_info(self, label:torch.Tensor):
        distr = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes).sum(dim = 0)
        if self.method_info == 'Gini':
            info = 1 - ((distr/distr.sum())**2).sum()
        elif self.method_info == 'entropy':
            info = 0
        return info

    def print_tree(self):
        def traverse_print(node: Node):
            if node.is_leaf == True:
                print(f"{'    '*node.depth} Branch {node.path}: {node.distr.numpy()}")
            elif node.is_leaf == False:
                if node.depth == 0:
                    print(f"{'    '*node.depth} Stump (x{node.feature.item()}, {node.threshold.item():.2f}): {node.distr.numpy()}")
                elif node.depth > 0:
                    print(f"{'    '*node.depth} Branch {node.path} (x{node.feature.item()}, {node.threshold.item():.2f}): {node.distr.numpy()}")
                for branch in node.children:
                    traverse_print(branch)
            
        traverse_print(self.stump)


h = DecisionTreeClassier(max_depth = 4, method_info = 'Gini')
h.fit(X_train, y_train)
h.print_tree()

print()