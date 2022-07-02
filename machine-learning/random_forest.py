import torch
from decision_tree import Node, DecisionTreeClassifier

class DecisionTreeInRFClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth = 2):
        super().__init__(max_depth = max_depth)
        self.feature_picks = None

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        # This is a little awkward, self.num_features will be overwritten
        # with the same value in super() 
        self.num_features = X_train.size()[1]
        self.feature_picks = torch.arange(self.num_features)
        super().fit(X_train, y_train)
    
    def find_best_split(self, node:Node, X_node:torch.Tensor, y_node:torch.Tensor):
        # Select random features (sqrt() of previous node's num_features)
        shuffle_idx = torch.randperm(self.feature_picks.numel())
        self.feature_picks = self.feature_picks.view(-1)[shuffle_idx].view(self.feature_picks.size())
        self.feature_picks = self.feature_picks[0:int(self.feature_picks.numel()**0.5)]
        # Split based on the reduced set of features
        max_gain = -torch.tensor(float('inf'))
        for feature in self.feature_picks:
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

class RandomForestClassifier():
    def __init__(self, num_trees = 100, bagg_ratio = 0.4, max_depth = 2):
        self.num_trees  = num_trees
        self.bagg_ratio = bagg_ratio
        self.bagg_size  = None
        self.max_depth  = max_depth

    def __repr__(self):
        return f"RF with {self.num_trees} trees, max depth {self.max_depth}, {self.bagg_ratio*100:.1f}% bagging"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        num_examples = X_train.size()[0]
        self.num_classes = len(y_train.unique())
        self.bagg_size = torch.tensor(self.bagg_ratio*num_examples).ceil().type(torch.int).item()
        # Create forest
        self.trees = [DecisionTreeInRFClassifier(max_depth = self.max_depth) for i in range(self.num_trees)]
        for tree in self.trees:
            # Bagging
            bag_picks = torch.randint(low = 0, high = num_examples, size = [self.bagg_size])
            X_bag, y_bag = X_train[bag_picks], y_train[bag_picks]
            tree.fit(X_bag, y_bag)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        pred = torch.zeros([input.size()[0], self.num_trees], dtype = torch.int)
        for tree_id, tree in enumerate(self.trees):
            pred[:, tree_id] = tree.forward(input).squeeze()
        # Major voting
        yhat = pred.mode(dim = 1)[0].unsqueeze(dim = 1)
        return yhat
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from utils_data import get_2D_clusters

    NUM_CLUSTERS = 4
    PLOT_STEP = 0.01
    COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black', 'tab:blue', 'tab:orange', 'tab:green',
            'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    X_train, y_train = get_2D_clusters(num_clusters = NUM_CLUSTERS)

    h = RandomForestClassifier(num_trees = 200, max_depth = 2)
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
    ax.set_title(f'Random Forest Classifier ({h.num_trees} trees, max depth {h.max_depth}, {h.bagg_ratio*100:.1f}% bagging) ' +
                 f'for {NUM_CLUSTERS} clusters')
    for label in torch.arange(h.num_classes):
        # Training data
        ax.scatter(X_train[y_train.squeeze() == label, 0], X_train[y_train.squeeze() == label, 1],
                   color = COLORS[label], alpha = 0.7, s = 10, zorder = 100)

    ax.pcolormesh(x1, x2, yhat_plot, cmap = ListedColormap(COLORS[0:h.num_classes]), alpha = 0.5, shading = 'auto')
    plt.show()