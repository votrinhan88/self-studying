# Topic: Supervised > Classification, Ensemble > Boosting

import torch
import torch.nn.functional as F
from typing import Literal, Tuple

from decision_tree import DecisionTree, Node

class DecisionTreeWeighted(DecisionTree):
    def __init__(self, method_info:Literal['gini', 'entropy'] = 'gini',
                       max_depth:int = 1,
                       drop_features:bool = True):
        self.task = 'c'
        super().__init__(self.task, method_info, max_depth, drop_features)
        self.weights:torch.Tensor = None

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor, weights:torch.Tensor):
        self.weights = weights
        super().fit(X_train = X_train,
                    y_train = y_train)

    def compute_gain(self, node:Node, idx_node:torch.Tensor, idx_left:torch.Tensor, idx_right:torch.Tensor) -> torch.Tensor:
        left_info = self.compute_info(idx_left)
        right_info = self.compute_info(idx_right)
        # gain = node.info - (idx_left.size()[0]*left_info + idx_right.size()[0]*right_info)/idx_node.size()[0]
        w_node = self.weights[idx_node].sum()
        w_left = self.weights[idx_left].sum()
        w_right = self.weights[idx_right].sum()
        gain = w_node*node.info - (w_left/w_node*left_info + w_right/w_node*right_info)
        return gain

    def feed_c(self, node:Node, idx_label:torch.Tensor):
        label = self.y_train[idx_label]
        node.info = self.compute_info_c(idx_label)
        
        # node.distr:torch.Tensor = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes).sum(dim = 0)
        onehot:torch.Tensor = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes)
        node.distr = (self.weights[idx_label]*onehot).sum(dim = 0)
        node.distr = node.distr/node.distr.sum()
        
        node.value = node.distr.max(dim = 0)[1]

    def compute_info_c(self, idx_label:torch.Tensor) -> torch.Tensor:
        label = self.y_train[idx_label]
        weights = self.weights[idx_label]
        
        # Classification: reducing Gini impurity or entropy
        onehot:torch.Tensor = F.one_hot(label.squeeze(dim = 1), num_classes = self.num_classes)
        # Weighted distribution
        weighted_distr = (weights*onehot).sum(dim = 0)
        weighted_distr = weighted_distr/weighted_distr.sum()
        if self.method_info == 'gini':
            info = 1 - (weighted_distr**2).sum()
        elif self.method_info == 'entropy':
            # Side note:
            # 1. Entropy of a system (all classes) = sum of entropy of its parts (each class)
            # 2. Moreover, if a class has no examples, it is absolutely predictable absent, hence its
            #   entropy is 0.
            # 3. To ease dealing with absent classes (which yields log(0) = -inf), and (2.), the 
            #   computation only considers existing classes.
            info = -(weighted_distr[weighted_distr != 0]*torch.log(weighted_distr[weighted_distr != 0])).sum()
        return info   

class AdaBoost():
    def __init__(self, method_info:Literal['gini', 'entropy'] = 'gini',
                 num_learners:int = 20,
                 max_depth:int = 1,
                 drop_features:bool = True,
                 learning_rate:float = 0.5):
        self.method_info = method_info
        self.num_learners = num_learners
        self.max_depth = max_depth
        self.drop_features = drop_features
        self.learning_rate = learning_rate

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with {self.num_learners} weak-learners, max depth {self.max_depth}"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        self.num_classes = len(y_train.unique())
        num_examples = X_train.size()[0]
        self.EPSILON = 1e-7/num_examples
        # Init
        self.learners = [DecisionTreeWeighted(method_info = self.method_info,
                                              max_depth   = self.max_depth,
                                              drop_features = True)
                        for learner in range(self.num_learners)]
        self.weights = torch.ones([num_examples, 1])/num_examples
        self.alpha = torch.zeros([self.num_learners])               # Weight of learners
        self.num_cur_learners = 0
        for learner_idx, learner in enumerate(self.learners):
            # Fit weak-learner and predict
            learner.fit(X_train, y_train, self.weights)
            yhat = learner.forward(X_train)
            incorrects = (yhat != y_train)

            error = self.weights[incorrects].sum()/num_examples
            alpha = 0.5*torch.log((1 - error)/(error + self.EPSILON))
            
            # Adjust examples' weight (favoring wrong predictions)
            self.weights[incorrects]  = self.weights[incorrects] *torch.exp(alpha*self.learning_rate)
            self.weights[~incorrects] = self.weights[~incorrects]*torch.exp(-alpha*self.learning_rate)
            # Set lower boundary for very small weights for stability
            self.weights[self.weights < self.EPSILON] = self.EPSILON
            self.weights = self.weights/self.weights.sum()
            # Save weak-learner's weight
            self.alpha[learner_idx] = alpha

            self.num_cur_learners = self.num_cur_learners + 1

    def strong_learner(self, input:torch.Tensor):
        yhat = torch.zeros([input.size()[0], 1], dtype = torch.float)
        for l_idx in torch.arange(self.num_cur_learners):
            pred_learner = self.learners[l_idx].forward(input)
            pred_learner[pred_learner == 0] = -1
            # Apply weights to each learners
            yhat = yhat + self.alpha[l_idx] * pred_learner
        
        # Strong learner's prediction is based on the sign of sum of weighted predictions from weak-learners
        yhat = (yhat >= 0).type(torch.long)
        return yhat

    def forward(self, input:torch.Tensor):
        return self.strong_learner(input)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.ensemble import AdaBoostClassifier

    from utils_data import get_clusters_2D

    # Dummy data
    NUM_CLUSTERS = 2
    SIGMA_DIAG = 0.5
    NUM_EXAMPLES = 200
    # Model
    METHOD_INFO = 'gini'
    NUM_LEARNERS = 15
    MAX_DEPTH = 1
    LEARNING_RATE = 0.5
    DROP_FEATURES = False
    # Visual
    PLOT_STEP = 0.01
    COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black', 'tab:blue', 'tab:orange', 'tab:green',
            'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Dummy data (slightly swirl pattern with noise in the center)
    X_train, y_train = get_clusters_2D(num_clusters = NUM_CLUSTERS,
                                       sigma_diag = SIGMA_DIAG,
                                       num_examples = NUM_EXAMPLES)
    y_train[(X_train[:, 0] < 1) & (X_train[:, 1] > 0.5), :] = 0
    y_train[(X_train[:, 0] >-1) & (X_train[:, 1] < -0.5), :] = 1

    # Model & Fit (also from sklearn for sanity check)
    h = AdaBoost(method_info = METHOD_INFO,
                 num_learners = NUM_LEARNERS, 
                 max_depth = MAX_DEPTH,
                 learning_rate = LEARNING_RATE,
                 drop_features = DROP_FEATURES)
    # h = AdaBoostClassifier(n_estimators = NUM_LEARNERS, learning_rate = LEARNING_RATE, algorithm = 'SAMME')
    if h.__class__ == AdaBoost:
        h.fit(X_train, y_train)
        yhat = h.forward(X_train)
    elif h.__class__ == AdaBoostClassifier:
        h.num_classes = X_train.size()[1]
        h.forward = h.predict
        h.fit(X_train, y_train)
        yhat = torch.tensor(h.forward(X_train)).unsqueeze(dim = 1)
    print(f'Train accuracy: {((yhat == y_train).sum()/y_train.size()[0]).item():.4f}')

    # Visualize regions of each class
    ptp_X = X_train.max(dim = 0)[0] - X_train.min(dim = 0)[0]
    plot_x1 = torch.arange(X_train[:, 0].min() - 0.2*ptp_X[0], X_train[:, 0].max() + 0.2*ptp_X[1], PLOT_STEP)
    plot_x2 = torch.arange(X_train[:, 1].min() - 0.2*ptp_X[0], X_train[:, 1].max() + 0.2*ptp_X[1], PLOT_STEP)
    x1, x2 = torch.meshgrid([plot_x1, plot_x2])
    X_plot = torch.cat([x1.flatten().unsqueeze(dim = 1), x2.flatten().unsqueeze(dim = 1)], dim = 1)
    # yhat_plot = h.forward(X_plot).reshape([plot_x1.size()[0], plot_x2.size()[0]])
    yhat_plot = torch.tensor(h.forward(X_plot)).reshape([plot_x1.size()[0], plot_x2.size()[0]])
    
    # Plot all examples
    fig, ax = plt.subplots()
    ax.set_title(f'{h} for {h.num_classes} clusters')
    for label in torch.arange(NUM_CLUSTERS):
        idx_label = (y_train.squeeze() == label)
        # Training data
        if h.__class__ == AdaBoostClassifier:
            ax.scatter(X_train[idx_label, 0], X_train[idx_label, 1],
                       color = COLORS[label], zorder = 100)
        elif h.__class__ == AdaBoost:
            s = (h.weights[idx_label].log() - h.weights.min().log() + 3)*(10/(h.weights.max().log() - h.weights.min().log() + 3))
            s = s**2
            ax.scatter(X_train[idx_label, 0], X_train[idx_label, 1],
                       s = s, color = COLORS[label], zorder = 100)

    ax.pcolormesh(x1, x2, yhat_plot, cmap = ListedColormap(COLORS[0:h.num_classes]), alpha = 0.5, shading = 'auto')

    plt.show()

    print()