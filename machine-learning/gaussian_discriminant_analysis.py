import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Constants
DATASET_SIZE = 600
NUM_CLUSTERS = 4
NUM_CLASSES = NUM_CLUSTERS
RADIUS = 1
SIGMA_DIAG = 0.2
K = 8
COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black', 'tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
PLOT_STEP = 0.01
pi = torch.acos(torch.zeros(1)).item()*2

# Generate dummy data
torch.manual_seed(17)
# Mu and Sigma for Gaussian distributions
mu = torch.cat(
    [RADIUS*torch.cos(torch.arange(NUM_CLUSTERS)*2*pi/NUM_CLUSTERS).unsqueeze(dim = 1),
     RADIUS*torch.sin(torch.arange(NUM_CLUSTERS)*2*pi/NUM_CLUSTERS).unsqueeze(dim = 1)],
    dim = 1)
sigma = SIGMA_DIAG*torch.tensor([[1., 0.], [0., 1.]])
# Sample from Gaussian distributions
examples_per_cluster = (torch.tensor(DATASET_SIZE)/NUM_CLUSTERS).type(torch.int)
x_raw = torch.cat([torch.distributions.multivariate_normal.MultivariateNormal(mu[cluster, :], sigma).sample([examples_per_cluster]) for cluster in range(NUM_CLUSTERS)], dim = 0)
y_raw = torch.cat([torch.tensor([[cluster]]*examples_per_cluster) for cluster in range(NUM_CLUSTERS)], dim = 0)
# Shuffle data
shuffle_index = torch.randperm(x_raw.size()[0])
x_raw = x_raw[shuffle_index]
y_raw = y_raw[shuffle_index]

class GaussianDiscriminantAnalyst:
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        self.x_train = x_train
        self.y_train = y_train

        self.num_classes = y_train.max() + 1
        self._fit()

    def _fit(self):
        class_distribution = F.one_hot(y_raw, NUM_CLASSES).sum(dim = 0).T

        self.theta = class_distribution/y_raw.size()[0]
        self.mu = F.one_hot(y_raw.squeeze(), NUM_CLASSES).T.float() @ self.x_train / class_distribution.repeat([1, 2])
        # Sigma is the same for every class = average variance of examples by their classes
        diff_example_to_mean = self.x_train - mu[y_raw.squeeze(), :]
        self.Sigma = (1/y_raw.size()[0]) * (diff_example_to_mean.T @ diff_example_to_mean)

    def forward(self, input:torch.Tensor):
        '''
        Bayes: p(y = c|x) = p(x|y = c).p(y = c) / p(x), where:
            p(x|y = c): Determined by the Gaussian distribution nabla(mu_c, sigma)
            p(y = c):   Proportions of example in class c
            p(x):       Distribution of x in its domain, is a constant
        
        Maximizing likelihood: find a set of parameters (theta, mu's, sigma); can skip p(x)
            argmax[p(y = c|x)] = argmax[p(x|y = c).p(y = c)]

        Broadcasting dimensions = {
          0: batches;
          1: classes;
          2, 3: matmul dimension}
        Hence:
          size()[2] = 1 for row vector
          size()[3] = 1 for column vector
        '''
        self.norm_coef = 1/((2*pi)**(self.num_classes/2) * self.Sigma.det().sqrt())
        input_unsqueeze = (input.unsqueeze(dim = 1) - self.mu).unsqueeze(dim = -1)
        p = self.norm_coef* \
            torch.exp(-0.5 * (input_unsqueeze.transpose(-2, -1) @ self.Sigma.inverse() @ input_unsqueeze)).squeeze()
        yhat = torch.max(p, dim = 1)[1].reshape([plot_x1.size()[0], plot_x2.size()[0]])
        return p, yhat

h = GaussianDiscriminantAnalyst(x_train = x_raw, y_train = y_raw)

# Visualize model
plot_x1 = torch.arange(x_raw[:, 0].min() - 0.2*RADIUS, x_raw[:, 0].max() + 0.2*RADIUS, PLOT_STEP)
plot_x2 = torch.arange(x_raw[:, 1].min() - 0.2*RADIUS, x_raw[:, 1].max() + 0.2*RADIUS, PLOT_STEP)
x1, x2 = torch.meshgrid([plot_x1, plot_x2])
x = torch.cat([x1.flatten().unsqueeze(dim = 1), x2.flatten().unsqueeze(dim = 1)], dim = 1)

p, plot_yhat = h.forward(x)
plot_yhat = plot_yhat.reshape([plot_x1.size()[0], plot_x2.size()[0]])

fig, ax = plt.subplots()
ax.set_title(f'Gaussian Discriminant Analysis with {NUM_CLASSES} classes')
for label in torch.arange(h.num_classes):
    ax.scatter(x_raw[(y_raw == label).squeeze(), 0], x_raw[(y_raw == label).squeeze(), 1], 
               edgecolors = None, color = COLORS[label], alpha = 0.5, s = 10, zorder = 100)
    ax.contour(x1, x2, p[:, label].reshape([plot_x1.size()[0], plot_x2.size()[0]]),
               linestyles = 'dashed', cmap = ListedColormap(COLORS[label]), alpha = 0.5)

ax.pcolormesh(x1, x2, plot_yhat, cmap = ListedColormap(COLORS[0:NUM_CLASSES]), alpha = 0.3, shading = 'auto')


plt.show()
print()