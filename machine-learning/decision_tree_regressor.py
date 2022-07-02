import torch
from typing import List

class NodeRegressor():
    def __init__(self, depth = None, info = None, feature = None, threshold = None, parent = None,
                 children = [], path:str = ''):
        self.is_leaf = False
        # Pre-allocate
        self.depth = depth
        self.info:torch.Tensor            = info
        self.feature:torch.Tensor         = feature
        self.threshold:torch.Tensor       = threshold
        self.parent:NodeRegressor         = parent
        self.children:List[NodeRegressor] = children
        self.path                         = path

        self.mean:torch.Tensor = None

    # Magic method to represent variable
    def __repr__(self) -> str:
        # Magic attribute to get class name
        if self.depth == 0:
            return f"{self.__class__.__name__} Root, I = {self.info:.3f}, mean = {self.mean.numpy()}"
        elif self.depth > 0:
            return f"{self.__class__.__name__} {self.path}, I = {self.info:.3f}, mean = {self.mean.numpy()}"

    def branch(self):
        left_node  = NodeRegressor(depth = self.depth + 1, path = self.path + 'L', parent = self)
        right_node = NodeRegressor(depth = self.depth + 1, path = self.path + 'R', parent = self)
        return left_node, right_node

class DecisionTreeRegressor():
    def __init__(self, max_depth = 4, method_info = 'var'):
        self.max_depth = max_depth
        self.method_info = method_info
        self.depth = 0

    def __repr__(self):
        return f"DT with depth {self.depth} (max {self.max_depth})"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        self.X_train = X_train
        self.y_train = y_train
        self.num_features = X_train.size()[1]

        self.root = NodeRegressor(depth = 0)
        self.root.mean = y_train.mean()
        self.root.info = self.compute_info(y_train)
        self.grow(node = self.root, X_node = X_train, y_node = y_train)

    def grow(self, node:NodeRegressor, X_node:torch.Tensor, y_node:torch.Tensor):
        max_gain = self.find_best_split(node, X_node, y_node)
        if max_gain > 0:
            # Continue growing
            ## Re-split data by optimal feature and threshold for children nodes
            self.depth = max(self.depth, node.depth + 1)
            left_node, right_node = node.branch()
            node.children = [left_node, right_node]

            left_ind = X_node[:, node.feature] <= node.threshold
            X_left, y_left = X_node[left_ind], y_node[left_ind]
            X_right, y_right = X_node[~left_ind], y_node[~left_ind]
            # Leaf node:
            #  - Has a unique class distribution (contains only one class)
            #  - OR has reached max depth
            for (branch, X_branch, y_branch) in zip([left_node, right_node], [X_left, X_right], [y_left, y_right]):
                branch.mean = y_branch.mean()
                branch.info = self.compute_info(y_branch)
                if (len(y_branch.unique()) == 1) | (branch.depth == self.max_depth):
                    branch.is_leaf = True
                else:
                    self.grow(branch, X_branch, y_branch)
    
    def find_best_split(self, node:NodeRegressor, X_node:torch.Tensor, y_node:torch.Tensor):
        max_gain = -torch.tensor(float('inf'))
        for feature in torch.arange(self.num_features):
            # thresholds = torch.linspace(start = X_node[:, feature].min(),
            #                             end = X_node[:, feature].max(),
            #                             steps = self.num_splits + 2)[1:self.num_splits+1]
            uniques = X_node[:, feature].sort()[0].unique()
            thresholds = (uniques[1:] + uniques[:-1])/2
            for threshold in thresholds:
                y_left, y_right = self.split(X_node, y_node, feature, threshold)
                gain = self.compute_gain(node, y_node, y_left, y_right)
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

    def compute_gain(self, node:NodeRegressor, y_node:torch.Tensor, left:torch.Tensor, right:torch.Tensor):
        left_info = self.compute_info(left)
        right_info = self.compute_info(right)
        gain = node.info - (left.size()[0]*left_info + right.size()[0]*right_info)/y_node.size()[0]
        return gain

    def compute_info(self, label:torch.Tensor) -> torch.Tensor:
        if self.method_info == 'var':
            info = ((label - label.mean())**2).sum()/label.size()[0]
        elif self.method_info == 'std':
            info = ((label - label.mean())**2).sum().sqrt()/label.size()[0]
        return info

    def print_tree(self):
        def traverse_print(node:NodeRegressor):
            # Print node
            if node.depth == 0:
                print(f"Root: {node.mean.item()}")
            elif node.depth > 0:
                if node.path[-1] == 'L':
                    print(f"{'    '*node.depth}Branch {node.path} (x{node.parent.feature.item()} ≤ {node.parent.threshold.item():.2f}): {node.mean.item()}")
                if node.path[-1] == 'R':
                    print(f"{'    '*node.depth}Branch {node.path} (x{node.parent.feature.item()} > {node.parent.threshold.item():.2f}): {node.mean.item()}")
            # Go to children branches
            if node.is_leaf == False:
                for branch in node.children:
                    traverse_print(branch)
            
        traverse_print(self.root)

    def forward(self, input:torch.Tensor, method = 'all') -> torch.Tensor:
        if method == 'each':
            # Method 1: Loop and traverse each example through the tree
            def traverse_forward(node:NodeRegressor, input:torch.Tensor, yhat) -> torch.Tensor:
                if yhat is not None:
                    return yhat
                elif yhat is None:
                    if node.is_leaf == True:
                        return node.mean
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
            def traverse_forward(node:NodeRegressor, input:torch.Tensor, yhat:torch.Tensor, yhat_id:torch.Tensor) -> torch.Tensor:
                if node.is_leaf == True:
                    yhat[yhat_id.squeeze()] = node.mean
                    return yhat
                elif node.is_leaf == False:
                    left_ind = input[:, node.feature] < node.threshold
                    left_input, right_input = input[left_ind, :], input[~left_ind, :]
                    left_yhat_id, right_yhat_id = yhat_id[left_ind, :], yhat_id[~left_ind, :]
                    for branch, branch_input, branch_yhat_id in zip(node.children, [left_input, right_input], [left_yhat_id, right_yhat_id]):
                        if len(branch_yhat_id) > 0:
                            yhat = traverse_forward(branch, branch_input, yhat, branch_yhat_id)
                    return yhat

            yhat = torch.zeros([input.size()[0], 1])
            yhat_id = torch.arange(input.size()[0]).unsqueeze(dim = 1)
            yhat = traverse_forward(self.root, input, yhat, yhat_id)
            return yhat

# Dummy data
X_train = 10*torch.rand([20, 1])
y_train = 0.8*X_train + torch.sin(X_train)

h = DecisionTreeRegressor(max_depth = 4)
h.fit(X_train, y_train)

X_test = torch.arange(start = -2, end = 12, step = 0.01).unsqueeze(dim = 1)
y_test = h.forward(X_test)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(constrained_layout = True)
ax.scatter(X_train, y_train,
           color = 'black', s = 40, label = 'Observed')
ax.step(X_test, y_test,
        linewidth = 2, color = 'blue', label = 'DT prediction')
ax.plot(X_test, 0.8*X_test + torch.sin(X_test),
        linewidth = 1, linestyle = 'dashed', color = 'red', alpha = 0.5, label = 'Ground truth')
ax.set(title = f"Decision Tree Regressor with depth {h.depth} (max {h.max_depth})")
ax.legend()
plt.show()