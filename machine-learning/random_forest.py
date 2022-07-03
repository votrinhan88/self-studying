import torch
from decision_tree import DecisionTree

from typing import Literal

class RandomForest():
    def __init__(self, task:Literal['r', 'c'] = 'c',
                       method_info:Literal['gini', 'entropy', 'var', 'std'] = 'gini',
                       bagg_ratio:float = 0.4,
                       num_trees:int = 100,
                       max_depth:int = 2):
        self.task        = task
        self.method_info = method_info
        self.bagg_ratio  = bagg_ratio
        self.num_trees   = num_trees
        self.max_depth   = max_depth

        self.bagg_size:torch.Tensor = None

    def __repr__(self) -> str:
        return f"RF-{self.task} with {self.bagg_ratio*100:.1f}% bagging, {self.num_trees} trees, max depth {self.max_depth}"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        num_examples = X_train.size()[0]
        self.num_classes = len(y_train.unique())
        self.bagg_size = torch.tensor(self.bagg_ratio*num_examples).ceil().type(torch.int).item()
        # Create forest
        self.trees = [DecisionTree(task = self.task,
                                   method_info = self.method_info,
                                   max_depth = self.max_depth)
                      for tree in range(self.num_trees)]
        for tree in self.trees:
            # Bagging
            bag_picks = torch.randint(low = 0, high = num_examples, size = [self.bagg_size])
            X_bag, y_bag = X_train[bag_picks], y_train[bag_picks]
            tree.fit(X_bag, y_bag)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        # Pre-allocate for predictions of treess
        pred = torch.zeros([input.size()[0], self.num_trees])
        if self.task == 'c':
            pred = pred.type(torch.long)
        elif self.task == 'r':
            pred = pred.type(torch.float)
        # Forward through every trees
        for tree_id, tree in enumerate(self.trees):
            pred[:, tree_id] = tree.forward(input).squeeze()
        # Give final prediction
        if self.task == 'c':
            # Major voting
            yhat = pred.mode(dim = 1, keepdim = True)[0]
        elif self.task == 'r':
            # Average
            yhat = pred.mean(dim = 1, keepdim = True)        
        return yhat
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    from utils_data import get_clusters_2D

    pi = torch.acos(torch.zeros(1)).item()*2

    NUM_CLUSTERS = 4
    PLOT_STEP = 0.01
    COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black', 'tab:blue', 'tab:orange', 'tab:green',
            'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Task 1: Classification for 4 2D clusters
    X_train, y_train = get_clusters_2D(num_clusters = NUM_CLUSTERS)

    h = RandomForest(task = 'c', method_info = 'gini', num_trees = 100, max_depth = 2)
    h.fit(X_train, y_train)
    
    # Visualize regions of each class
    ptp_X = X_train.max(dim = 0)[0] - X_train.min(dim = 0)[0]
    plot_x1 = torch.arange(X_train[:, 0].min() - 0.2*ptp_X[0], X_train[:, 0].max() + 0.2*ptp_X[1], PLOT_STEP)
    plot_x2 = torch.arange(X_train[:, 1].min() - 0.2*ptp_X[0], X_train[:, 1].max() + 0.2*ptp_X[1], PLOT_STEP)
    x1, x2 = torch.meshgrid([plot_x1 - PLOT_STEP/2, plot_x2 - PLOT_STEP/2])
    X_plot = torch.cat([x1.flatten().unsqueeze(dim = 1), x2.flatten().unsqueeze(dim = 1)], dim = 1)
    yhat_plot = h.forward(X_plot).reshape([plot_x1.size()[0], plot_x2.size()[0]])
    
    # Plot all centroids and examples
    fig, ax = plt.subplots()
    ax.set_title(f'{h} for {NUM_CLUSTERS} clusters')
    for label in torch.arange(h.num_classes):
        # Training data
        ax.scatter(X_train[y_train.squeeze() == label, 0], X_train[y_train.squeeze() == label, 1],
                   color = COLORS[label], alpha = 0.7, s = 10, zorder = 100)

    ax.pcolormesh(x1, x2, yhat_plot, cmap = ListedColormap(COLORS[0:h.num_classes]), alpha = 0.5, shading = 'auto')

    # Task 2: Regression for 4 same 2D clusters
    # Generate dummy data: 2d circular wave centered at 0
    num_examples = 600
    X_train = torch.cat([-1 + 2*torch.rand([num_examples, 1]),
                         -1 + 2*torch.rand([num_examples, 1])],
                        dim = 1)
    def circular_wave(input:torch.Tensor) -> torch.Tensor:
        t = 0.5
        frequency = 0.5
        wavelength = 1
        amplitude = 2

        radius = (input[:, [0]]**2 + input[:, [1]]**2).sqrt()
        u = amplitude*torch.cos((2*pi/wavelength)*radius+(2*pi*frequency)*t)
        return u
    
    y_train = circular_wave(X_train)

    h = RandomForest(task = 'r', method_info = 'var', num_trees = 20, max_depth = 5)
    h.fit(X_train, y_train)

    # Visualize regions of each class
    ptp_X = X_train.max(dim = 0)[0] - X_train.min(dim = 0)[0]
    plot_x1 = torch.arange(X_train[:, 0].min() - 0.2*ptp_X[0], X_train[:, 0].max() + 0.2*ptp_X[1], PLOT_STEP)
    plot_x2 = torch.arange(X_train[:, 1].min() - 0.2*ptp_X[0], X_train[:, 1].max() + 0.2*ptp_X[1], PLOT_STEP)
    x1, x2 = torch.meshgrid([plot_x1 - PLOT_STEP/2, plot_x2 - PLOT_STEP/2])
    X_plot = torch.cat([x1.flatten().unsqueeze(dim = 1), x2.flatten().unsqueeze(dim = 1)], dim = 1)
    yhat_plot = h.forward(X_plot).reshape([plot_x1.size()[0], plot_x2.size()[0]])

    # Plot all centroids and examples
    fig, ax = plt.subplots()
    ax.set_title(f'{h} for 2D circular wave')
    # Training data
    ax.scatter(X_train[:, 0], X_train[:, 1],
               c = y_train.squeeze(), cmap = 'plasma', alpha = 0.7, s = 10, zorder = 100)
    ax.pcolormesh(x1, x2, yhat_plot, cmap = 'plasma', alpha = 0.5, shading = 'auto')
    
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, constrained_layout = True)
    # ax.scatter(X_train[:, 0], X_train[:, 1], y_train.squeeze(),
    #            s = 5, color = 'black', alpha = 1)
    ax.plot_surface(x1.numpy(), x2.numpy(), yhat_plot.numpy(),
               cmap = 'plasma', alpha = 0.7)
    ax.set_title(f'{h} for 2D circular wave')
    plt.show()