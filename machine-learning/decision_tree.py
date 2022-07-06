# Topic: Supervised, Classification, Regression

import torch
import torch.nn.functional as F

from typing import Tuple, Literal

class Node():
    def __init__(self, task:Literal['r', 'c'] = 'c',
                       depth:int = None,
                       path:str = '',
                       parent = None):
        # Arguments
        self.task        = task
        self.depth       = depth
        self.path        = path
        self.parent:Node = parent
        # Pre-allocate
        self.is_leaf = False
        self.info:torch.Tensor = None
        self.feature:torch.Tensor = None
        self.threshold:torch.Tensor = None
        self.children:list[Node] = []
        # Based on task (Classification/Regression)
        self.value:torch.Tensor = None
        if self.task == 'c':
            self.distr:torch.Tensor = None

    def __repr__(self) -> str:
        # Magic attribute to get class name
        if self.depth == 0:
            return f"{self.__class__.__name__}-{self.task} Root, I = {self.info:.3f}, value = {round(self.value, 2)}"
        elif self.depth > 0:
            return f"{self.__class__.__name__}-{self.task} {self.path}, I = {self.info:.3f}, value = {round(self.value, 2)}"

    def branch(self): # -> Tuple[Self, Self]
        left_node  = Node(task = self.task, depth = self.depth + 1, path = self.path + 'L', parent = self)
        right_node = Node(task = self.task, depth = self.depth + 1, path = self.path + 'R', parent = self)
        return left_node, right_node

class DecisionTree():
    def __init__(self, task:Literal['c', 'r'] = 'c',
                       method_info:Literal['gini', 'entropy', 'var', 'std'] = 'gini',
                       max_depth:int = 4,
                       drop_features:bool = True):
        assert ((task == 'c') & (method_info in ['gini', 'entropy']) |
                (task == 'r') & (method_info in ['var', 'std'])), \
               "Use task 'c' with method_info 'gini' or 'entropy', or task 'r' with method_info 'var' or 'std'"
        self.task          = task
        self.method_info   = method_info
        self.max_depth     = max_depth
        self.drop_features = drop_features

        self.depth:int = 0
        self.feature_picks:torch.Tensor = None

    def __repr__(self) -> str:
        return f"DT-{self.task} with depth {self.depth} (max {self.max_depth}), drop features {'ON' if self.drop_features else 'OFF'}"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        self.X_train = X_train
        self.y_train = y_train

        self.num_features = X_train.size()[1]
        self.feature_picks = torch.arange(self.num_features)

        self.root = Node(depth = 0, task = self.task)

        if self.task == 'c':
            self.num_classes = len(y_train.unique())
            self.feed = self.feed_c
            self.compute_info = self.compute_info_c
        elif self.task == 'r':
            self.feed = self.feed_r
            self.compute_info = self.compute_info_r

        idx_root = torch.arange(self.X_train.size()[0])
        self.feed(node = self.root,
                  idx_label = idx_root)
        self.grow(node = self.root,
                  idx_node = idx_root)

    def grow(self, node:Node, idx_node:torch.Tensor):
        max_gain = self.find_best_split(node, idx_node)
        if max_gain > 0:
            # Continue growing
            self.depth = max(self.depth, node.depth + 1)
            left_node, right_node = node.branch()
            node.children = [left_node, right_node]
            # Re-split data by optimal feature and threshold for children nodes
            idx_left, idx_right = self.split(idx_node, feature = node.feature, threshold = node.threshold)
            # Leaf node:
            #  - Has a unique class distribution (contains only one class)
            #  - OR has reached max depth
            for (branch, idx_branch) in zip([left_node, right_node], [idx_left, idx_right]):
                self.feed(branch, idx_branch)
                y_branch = self.y_train[idx_branch]
                if (len(y_branch.unique()) == 1) | (branch.depth == self.max_depth):
                    branch.is_leaf = True
                else:
                    self.grow(branch, idx_branch)
    
    def find_best_split(self, node:Node, idx_node:torch.Tensor) -> torch.Tensor:
        X_node = self.X_train[idx_node]
        # Select random features (sqrt() of previous node's num_features)
        if self.drop_features == True:
            shuffle_idx = torch.randperm(self.feature_picks.numel())
            self.feature_picks = self.feature_picks.view(-1)[shuffle_idx].view(self.feature_picks.size())
            self.feature_picks = self.feature_picks[0:int(self.feature_picks.numel()**0.5)]
        # Split based on the reduced (or not) set of features 
        max_gain = -torch.tensor(float('inf'))
        for feature in torch.arange(self.num_features):
            # thresholds = torch.linspace(start = X_node[:, feature].min(),
            #                             end = X_node[:, feature].max(),
            #                             steps = self.num_splits + 2)[1:self.num_splits+1]
            uniques = X_node[:, feature].sort()[0].unique()
            thresholds = (uniques[1:] + uniques[:-1])/2
            for threshold in thresholds:
                idx_left, idx_right = self.split(idx_node = idx_node,
                                                 feature = feature,
                                                 threshold = threshold)
                gain = self.compute_gain(node,
                                         idx_node = idx_node,
                                         idx_left = idx_left,
                                         idx_right = idx_right)
                if gain > max_gain:
                    max_gain = gain
                    opt_feature = feature
                    opt_threshold = threshold        
        # Update node with optimal split
        node.feature = opt_feature
        node.threshold = opt_threshold
        return max_gain

    def split(self, idx_node:torch.Tensor, feature:torch.Tensor, threshold:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_node = self.X_train[idx_node]

        left_ind = X_node[:, feature] <= threshold
        idx_left = idx_node[left_ind]
        idx_right = idx_node[~left_ind]
        return idx_left, idx_right

    def compute_gain(self, node:Node, idx_node:torch.Tensor, idx_left:torch.Tensor, idx_right:torch.Tensor) -> torch.Tensor:
        left_info = self.compute_info(idx_left)
        right_info = self.compute_info(idx_right)
        
        gain = node.info - (idx_left.numel()*left_info + idx_right.numel()*right_info)/idx_node.numel()
        
        # w_node = self.weights[idx_node].sum()
        # w_left = self.weights[idx_left].sum()
        # w_right = self.weights[idx_right].sum()
        # gain = w_node*node.info - (w_left/w_node*left_info + w_right/w_node*right_info)
        return gain

    def feed_c(self, node:Node, idx_label:torch.Tensor):
        label = self.y_train[idx_label]
        node.info = self.compute_info_c(idx_label)
        onehot:torch.Tensor = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes)
        
        node.distr = onehot.sum(dim = 0)

        # node.distr = (self.weights[idx_label]*onehot).sum(dim = 0)
        # node.distr = node.distr/node.distr.sum()

        node.value = node.distr.max(dim = 0)[1]

    def feed_r(self, node:Node, idx_label:torch.Tensor):
        label = self.y_train[idx_label]
        node.info = self.compute_info_r(idx_label)
        node.value = label.mean()
    
    def compute_info_c(self, idx_label:torch.Tensor) -> torch.Tensor:
        label = self.y_train[idx_label]
        
        # Classification: reducing Gini impurity or entropy
        onehot:torch.Tensor = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes)
        distr = onehot.sum(dim = 0)
        distr = distr/distr.sum()
        if self.method_info == 'gini':
            info = 1 - (distr**2).sum()
        elif self.method_info == 'entropy':
            # Side note:
            # 1. Entropy of a system (all classes) = sum of entropy of its parts (each class)
            # 2. Moreover, if a class has no examples, it is absolutely predictable absent, hence its
            #   entropy is 0.
            # 3. To ease dealing with absent classes (which yields log(0) = -inf), and (2.), the 
            #   computation only considers existing classes.
            info = -(distr[distr != 0]*distr[distr != 0].log()).sum()
        return info
    
    def compute_info_r(self, idx_label:torch.Tensor) -> torch.Tensor:
        label = self.y_train[idx_label]

        # Regression: reducing Variance or Standard deviation
        if self.method_info == 'var':
            info = ((label - label.mean())**2).sum()/label.size()[0]
        elif self.method_info == 'std':
            info = ((label - label.mean())**2).sum().sqrt()/label.size()[0]
        return info

    def print_tree(self):
        def traverse_print(node: Node):
            # Print node
            if node.depth == 0:
                print(f"Root:")
            elif node.depth > 0:
                print(f"{'    '*node.depth}Branch {node.path} " +
                      f"(x{node.parent.feature.item()} {'≤' if node.path[-1] == 'L' else '>'} {node.parent.threshold.item():.2f}):" +
                      f"{f' {node.distr.numpy()}' if self.task == 'c' else ''}" +
                      f"{f' = {round(node.value.item(), 2)}' if node.is_leaf else ''}")
            # Go to children branches
            if node.is_leaf == False:
                for branch in node.children:
                    traverse_print(branch)
            
        traverse_print(self.root)

    def forward(self, input:torch.Tensor, method = 'all') -> torch.Tensor:
        if method == 'each':
            # Method 1: Loop and traverse each example through the tree
            def traverse_forward(node:Node, input:torch.Tensor, yhat) -> torch.Tensor:
                if yhat is not None:
                    return yhat
                elif yhat is None:
                    if node.is_leaf == True:
                        return node.value
                    elif node.is_leaf == False:
                        if input[node.feature] < node.threshold:
                            return traverse_forward(node.children[0], input, yhat)
                        elif input[node.feature] >= node.threshold:
                            return traverse_forward(node.children[1], input, yhat)
            
            yhat = -torch.ones([input.size()[0], 1], dtype = torch.int8)
            for example in torch.arange(input.size()[0]):
                yhat[example] = traverse_forward(self.root, input[example, :], yhat = None)
            return yhat

        elif method == 'all':
            # Method 2: Traverse all examples through the tree at once
            def traverse_forward(node:Node, input:torch.Tensor, yhat:torch.Tensor, yhat_id:torch.Tensor) -> torch.Tensor:
                if node.is_leaf == True:
                    yhat[yhat_id.squeeze()] = node.value
                    return yhat
                elif node.is_leaf == False:
                    left_ind = input[:, node.feature] < node.threshold
                    left_input, right_input = input[left_ind, :], input[~left_ind, :]
                    idx_left, idx_right = yhat_id[left_ind, :], yhat_id[~left_ind, :]
                    for branch, branch_input, idx_branch in zip(node.children, [left_input, right_input], [idx_left, idx_right]):
                        if len(idx_branch) > 0:
                            yhat = traverse_forward(branch, branch_input, yhat, idx_branch)
                    return yhat

            yhat = -torch.ones([input.size()[0], 1])
            if self.task == 'c':
                yhat = yhat.type(torch.long)
            elif self.task == 'r':
                yhat = yhat.type(torch.float)
            yhat_id = torch.arange(input.size()[0]).unsqueeze(dim = 1)
            yhat = traverse_forward(self.root, input, yhat, yhat_id)
            return yhat

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils_data import get_iris

    NUM_OBSERVED = 20

    # Task 1: Classification on Iris
    # Load Iris
    X, y = get_iris()
    num_trains = round(X.size()[0]*0.8)
    X_train, y_train = X[0:num_trains], y[0:num_trains]
    X_test, y_test = X[num_trains:], y[num_trains:]

    h = DecisionTree(task = 'c', method_info = 'gini', max_depth = 4, drop_features = True)
    h.fit(X_train, y_train)
    yhat = h.forward(X_test)
    print(h)
    h.print_tree()
    print(f'Accuracy = {((yhat == y_test).sum()/y_test.size()[0]).item():.4f}')

    # Task 2: Regression on 1D Signal
    # Generate dummy data
    def signal(input):
        return 0.8*input + torch.sin(input)
    X_train = 10*torch.rand([NUM_OBSERVED, 1])
    y_train = signal(X_train)

    h = DecisionTree(task = 'r', method_info = 'var', max_depth = 3, drop_features = True)
    h.fit(X_train, y_train)

    X_test = torch.arange(start = -2, end = 12, step = 0.01).unsqueeze(dim = 1)
    y_test = h.forward(X_test)
    
    fig, ax = plt.subplots(constrained_layout = True)
    ax.scatter(X_train, y_train,
            color = 'black', s = 40, label = 'Observed')
    ax.step(X_test, y_test,
            linewidth = 2, color = 'blue', label = 'DT prediction')
    ax.plot(X_test, 0.8*X_test + torch.sin(X_test),
            linewidth = 1, linestyle = 'dashed', color = 'red', alpha = 0.5, label = 'Ground truth')
    ax.set(title = h)
    
    ax.legend()

    print()
    print(h)
    h.print_tree()

    plt.show()

'''
Snippet to benchmarking tree.forward() using method = 'each' vs 'all'

import time

start = time.time()
yhat_each = h.forward(X_test, method = 'each')
end = time.time()
time1 = end - start
print(f'Forwarding examples one-by-one: {time1:.2e} s')

start = time.time()
yhat_all = h.forward(X_test, method = 'all')
end = time.time()
time2 = end - start
print(f'Forwarding examples all at once: {time2:.2e} s')

print(f'Faster how many times? {time1/time2:.2f}')
print(f'Confirm results of both methods are the same: {(yhat_each == yhat_all).prod().bool().item()}')
'''