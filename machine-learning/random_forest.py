import torch
from decision_tree import Node, DecisionTreeClassifier
from utils_data import get_iris

NUM_TREES = 100
BAGGING_SIZE = 0.4
MAX_DEPTH = 2

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
    def __init__(self, num_trees = NUM_TREES, bagg_ratio = BAGGING_SIZE, max_depth = MAX_DEPTH):
        self.num_trees  = num_trees
        self.bagg_ratio = bagg_ratio
        self.bagg_size  = None
        self.max_depth  = max_depth

    def __repr__(self):
        return f"RF with {self.num_trees} trees, max depth {self.max_depth}, {self.bagg_ratio*100:.1f}% bagging"

    def fit(self, X_train:torch.Tensor, y_train:torch.Tensor):
        num_examples = X_train.size()[0]
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
    X_train, y_train, X_test, y_test = get_iris()

    for i in range(5):
        h = RandomForestClassifier(max_depth = 2)
        h.fit(X_train, y_train)
        yhat = h.forward(X_test)
        if i == 0:
            print(h)
        print(f'Run {i}: Accuracy = {((yhat == y_test).sum()/y_test.size()[0]).item():.4f}')