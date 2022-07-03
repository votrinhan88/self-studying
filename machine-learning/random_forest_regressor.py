import torch
from decision_tree_regressor import NodeRegressor, DecisionTreeRegressor

class DecisionTreeInRFRegressor(DecisionTreeRegressor):
    def __init__(self, max_depth:int = 2, drop_features:bool = True):
        super().__init__(max_depth = max_depth)
        self.feature_picks = None
        self.drop_features = drop_features

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        # This is a little awkward, self.num_features will be overwritten
        # with the same value in super() 
        self.num_features = X_train.size()[1]
        self.feature_picks = torch.arange(self.num_features)
        super().fit(X_train, y_train)
    
    def find_best_split(self, node:NodeRegressor, X_node:torch.Tensor, y_node:torch.Tensor):
        # Select random features (sqrt() of previous node's num_features)
        if self.drop_features == True:
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


class RandomForestRegressor():
    def __init__(self, num_trees:int = 100, max_depth:int = 2,bagg_ratio:float = 0.4,  drop_features:bool = True):
        self.num_trees     = num_trees
        self.max_depth     = max_depth
        self.bagg_ratio    = bagg_ratio
        self.bagg_size     = None
        self.drop_features = drop_features

    def __repr__(self):
        return (f"RF with {self.num_trees} trees, max depth {self.max_depth}, {self.bagg_ratio*100:.1f}% bagging, " + 
                f"drop features {'on' if self.drop_features else 'off'}")

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        num_examples = X_train.size()[0]
        self.num_classes = len(y_train.unique())
        self.bagg_size = torch.tensor(self.bagg_ratio*num_examples).ceil().type(torch.int).item()
        # Create forest
        self.trees = [DecisionTreeInRFRegressor(max_depth = self.max_depth, drop_features = self.drop_features)
                      for tree in range(self.num_trees)]
        for tree in self.trees:
            # Bagging
            bag_picks = torch.randint(low = 0, high = num_examples, size = [self.bagg_size])
            X_bag, y_bag = X_train[bag_picks], y_train[bag_picks]
            tree.fit(X_bag, y_bag)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        pred = torch.zeros([input.size()[0], self.num_trees])
        for tree_id, tree in enumerate(self.trees):
            pred[:, tree_id] = tree.forward(input).squeeze()
        # Mean of predictions
        yhat = pred.mean(dim = 1, keepdim = True)
        return yhat

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Dummy data
    X_train = 10*torch.rand([30, 1])
    y_train = 0.8*X_train + torch.sin(X_train)

    h = RandomForestRegressor(num_trees = 100, max_depth = 2, bagg_ratio = 0.4, drop_features = True)
    h2 = RandomForestRegressor(num_trees = 100, max_depth = 4, bagg_ratio = 0.4, drop_features = False)
    h.fit(X_train, y_train)
    h2.fit(X_train, y_train)

    X_test = torch.arange(start = -2, end = 12, step = 0.01).unsqueeze(dim = 1)
    y_test = h.forward(X_test)
    y2_test = h2.forward(X_test)


    fig, ax = plt.subplots(constrained_layout = True)
    ax.scatter(X_train, y_train,
            color = 'black', s = 40, label = 'Observed')
    ax.step(X_test, y_test,
            linewidth = 2, color = 'blue', label = h)
    ax.step(X_test, y2_test,
            linewidth = 2, label = h2)
    ax.plot(X_test, 0.8*X_test + torch.sin(X_test),
            linewidth = 1, linestyle = 'dashed', color = 'red', alpha = 0.5, label = 'Ground truth')
    ax.set(title = f"Random Forest Regressor with max {h.max_depth}")
    ax.legend()
    plt.show()