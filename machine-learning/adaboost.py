# Topic: Supervised, Classification, Ensemble, Boosting
import torch
import torch.nn.functional as F
from typing import Literal, Union, Tuple

from decision_tree import DecisionTree, Node


class DecisionTreeAdaBoost(DecisionTree):
    def __init__(self, method_info:Literal['gini', 'entropy'] = 'gini',
                       max_depth:int = 1,
                       drop_features:bool = True):
        self.task = 'c'
        super().__init__(self.task, method_info, max_depth, drop_features)
        self.weights:torch.Tensor = None

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor, weights:torch.Tensor):
        self.weights = weights
        self.X_train = X_train
        self.y_train = y_train

        self.num_features = X_train.size()[1]
        self.feature_picks = torch.arange(self.num_features)

        self.root = Node(depth = 0, task = self.task)

        if self.task == 'c':
            self.num_classes = len(y_train.unique())
            self.feed = self.feed_c
            self.compute_info = self.compute_info_c

        idx_root = torch.arange(self.X_train.size()[0])
        self.feed(self.root, idx_root)
        self.grow(self.root, idx_root)

    def grow(self, node:Node, idx_node:torch.Tensor):
        X_node = self.X_train[idx_node]
        max_gain = self.find_best_split(node, idx_node)
        if max_gain > 0:
            # Continue growing
            ## Re-split data by optimal feature and threshold for children nodes
            self.depth = max(self.depth, node.depth + 1)
            left_node, right_node = node.branch()
            node.children = [left_node, right_node]

            left_ind = X_node[:, node.feature] <= node.threshold
            idx_left = idx_node[left_ind]
            idx_right = idx_node[~left_ind]
            # X_left, y_left = X_node[left_ind], y_node[left_ind]
            # X_right, y_right = X_node[~left_ind], y_node[~left_ind]
            
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

    def find_best_split(self, node:Node, idx_node:torch.Tensor):
        X_node = self.X_train[idx_node]
        # Select random features (sqrt() of previous node's num_features)
        if self.drop_features == True:
            shuffle_idx = torch.randperm(self.feature_picks.numel())
            self.feature_picks = self.feature_picks.view(-1)[shuffle_idx].view(self.feature_picks.size())
            self.feature_picks = self.feature_picks[0:int(self.feature_picks.numel()**0.5)]
        # Split based on the reduced set of features (or not)
        max_gain = -torch.tensor(float('inf'))
        for feature in torch.arange(self.num_features):
            # thresholds = torch.linspace(start = X_node[:, feature].min(),
            #                             end = X_node[:, feature].max(),
            #                             steps = self.num_splits + 2)[1:self.num_splits+1]
            uniques = X_node[:, feature].sort()[0].unique()
            thresholds = (uniques[1:] + uniques[:-1])/2
            for threshold in thresholds:
                idx_left, idx_right = self.split(idx_node, feature, threshold)
                gain = self.compute_gain(node, idx_node, idx_left, idx_right)
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
        gain = node.info - (idx_left.size()[0]*left_info + idx_right.size()[0]*right_info)/idx_node.size()[0]
        return gain

    def feed_c(self, node:Node, idx_label:torch.Tensor):
        label = self.y_train[idx_label]
        node.distr:torch.Tensor = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes).sum(dim = 0)
        node.value = node.distr.max(dim = 0)[1]
        node.info = self.compute_info_c(idx_label)

    def compute_info_c(self, idx_label:torch.Tensor) -> torch.Tensor:
        label = self.y_train[idx_label]
        weights = self.weights[idx_label]
        
        # Classification: reducing Gini impurity or entropy
        distr:torch.Tensor = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes)
        # Weighted distribution
        distr = (weights*distr).sum(dim = 0)
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

class AdaBoost():
    EPSILON = torch.finfo(torch.float).eps
    def __init__(self, method_info:Literal['gini', 'entropy'] = 'gini', num_learners:int = 100, max_depth:int = 1, learning_rate:float = 0.5):
        self.method_info = method_info
        self.num_learners = num_learners
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with {self.num_learners} weak-learners, max depth {self.max_depth}"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        self.num_classes = len(y_train.unique())
        num_examples = X_train.size()[0]
        # Init
        self.learners = [DecisionTreeAdaBoost(method_info = self.method_info,       # 
                                              max_depth   = self.max_depth,
                                              drop_features = False)
                        for learner in range(self.num_learners)]
        self.weights = torch.ones([num_examples, 1])/num_examples
        self.alpha = torch.zeros([self.num_learners])  # Weight of learners
        self.num_cur_learners = 0
        for learner_idx, learner in enumerate(self.learners):
            # Fit weak-learner and predict
            learner.fit(X_train, y_train, self.weights)
            yhat = learner.forward(X_train)
            incorrects = (yhat != y_train)

            error = (self.weights*incorrects).sum()/num_examples
            alpha = 0.5*torch.log((1 - error)/(error + self.EPSILON))
            
            # Adjust examples' weight (favoring wrong predictions)
            self.weights[incorrects]  = self.weights[incorrects] *torch.exp(alpha*self.learning_rate)
            self.weights[~incorrects] = self.weights[~incorrects]*torch.exp(-alpha*self.learning_rate)
            self.weights = self.weights/self.weights.sum()
            # Save weak-learner's weight
            self.alpha[learner_idx] = alpha

            self.num_cur_learners = self.num_cur_learners + 1

    def strong_learner(self, input:torch.Tensor):
        pred_learners = torch.zeros([input.size()[0], self.num_cur_learners], dtype = torch.long)
        for l_idx in torch.arange(self.num_cur_learners):
            pred_learner = self.learners[l_idx].forward(input)
            pred_learner[pred_learner == 0] = -1
            pred_learners[:, [l_idx]] = pred_learner
        
        # Apply weights to each learners
        pred_learners = self.alpha * pred_learners

        # Strong learner's prediction is based on the sign of sum of weighted predictions from weak-learners
        yhat = (pred_learners.sum(dim = 1, keepdim = True) >= 0).type(torch.long)
        return yhat

    def forward(self, input:torch.Tensor):
        return self.strong_learner(input)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from utils_data import get_clusters_2D

    NUM_CLUSTERS = 2
    PLOT_STEP = 0.05
    COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black', 'tab:blue', 'tab:orange', 'tab:green',
            'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    X_train, y_train = get_clusters_2D(num_clusters = NUM_CLUSTERS, sigma_diag = 0.7, num_examples = 200)
    y_train[(X_train[:, 0] < 1) & (X_train[:, 1] > 0.5), :] = 0
    y_train[(X_train[:, 0] >-1) & (X_train[:, 1] < -0.5), :] = 1
    # y_train[y_train == 0] = -1

    h = AdaBoost(num_learners = 10, max_depth = 1, learning_rate = 0.2)
    h.fit(X_train, y_train)

    yhat = h.forward(X_train)
    print(f'Train accuracy: {((yhat == y_train).sum()/y_train.size()[0]).item():.4f}')

    # Visualize regions of each class
    ptp_X = X_train.max(dim = 0)[0] - X_train.min(dim = 0)[0]
    plot_x1 = torch.arange(X_train[:, 0].min() - 0.2*ptp_X[0], X_train[:, 0].max() + 0.2*ptp_X[1], PLOT_STEP)
    plot_x2 = torch.arange(X_train[:, 1].min() - 0.2*ptp_X[0], X_train[:, 1].max() + 0.2*ptp_X[1], PLOT_STEP)
    x1, x2 = torch.meshgrid([plot_x1 - PLOT_STEP/2, plot_x2 - PLOT_STEP/2])
    X_plot = torch.cat([x1.flatten().unsqueeze(dim = 1), x2.flatten().unsqueeze(dim = 1)], dim = 1)
    yhat_plot = h.forward(X_plot).reshape([plot_x1.size()[0], plot_x2.size()[0]])
    
    # Plot all examples
    fig, ax = plt.subplots()
    ax.set_title(f'{h} for {NUM_CLUSTERS} clusters')
    for label in torch.arange(h.num_classes):
        idx_label = (y_train.squeeze() == label)
        # Training data
        ax.scatter(X_train[idx_label, 0], X_train[idx_label, 1],
                   s = torch.min(h.weights[idx_label]/h.weights.min(), torch.tensor([200])),
                   color = COLORS[label], zorder = 100)

    ax.pcolormesh(x1, x2, yhat_plot, cmap = ListedColormap(COLORS[0:h.num_classes]), alpha = 0.5, shading = 'auto')

    plt.show()

    print()

