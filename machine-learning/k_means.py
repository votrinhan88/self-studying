import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Constants
DATASET_SIZE = 600
NUM_CLUSTERS = 4
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


# K-means Algorithm
class KMeansClassifier():
    def __init__(self, K, train_data):
        self.K = K
        self.centroids:torch.Tensor = train_data[torch.randperm(train_data.size()[0])][range(self.K)]

    # Fix centroids, update labels
    def forward(self, input:torch.Tensor):
        distance = torch.zeros([input.size()[0], self.K])
        for k in torch.arange(self.K):
            distance[:, k] = torch.sqrt(torch.sum((self.centroids[k, :] - input)**2, dim = 1))
        
        (_, yhat) = torch.min(distance, dim = 1)
        yhat = yhat.unsqueeze(dim = 1)
        
        return yhat

    # Fix labels, update centroids
    def backward(self, yhat:torch.Tensor):
        for k in torch.arange(self.K):
            self.centroids[k, :] = torch.mean(x_raw[yhat.squeeze() == k, :], dim = 0)

# Init
h = KMeansClassifier(K, x_raw)
centroids_log = h.centroids.clone().unsqueeze(dim = 0)
# Training loop
for epoch in torch.arange(20):
    # Fix centroids, update labels
    yhat = h.forward(x_raw)
    # Fix labels, update centroids
    h.backward(yhat)

    # Check for stopping condition: when centroids stop moving
    if (centroids_log[-1, :, :] == h.centroids).prod().bool() == True:
        print(f'Stopped at epoch {epoch.item()}! Centroids stop moving.')
        break
    
    # Log centroids position
    centroids_log = torch.cat([centroids_log, h.centroids.unsqueeze(dim = 0)], dim = 0)

# Plot all centroids and examples 
fig, ax = plt.subplots()
ax.set_title(f'{K}-means with {NUM_CLUSTERS} clusters')
for k in torch.arange(h.K):
    # Centroids' path
    ax.plot(centroids_log[:, k, 0], centroids_log[:, k, 1],
            color = COLORS[k], linestyle = 'dashed')
    ax.scatter(centroids_log[:, k, 0], centroids_log[:, k, 1],
               color = COLORS[k], marker = 'x')
    # Training data
    ax.scatter(x_raw[yhat.squeeze() == k, 0], x_raw[yhat.squeeze() == k, 1],
               color = COLORS[k], alpha = 0.7, s = 10, zorder = 100)

# Voronoi diagram for centroids
plot_x1 = torch.arange(x_raw[:, 0].min() - 0.2*RADIUS, x_raw[:, 0].max() + 0.2*RADIUS, PLOT_STEP)
plot_x2 = torch.arange(x_raw[:, 1].min() - 0.2*RADIUS, x_raw[:, 1].max() + 0.2*RADIUS, PLOT_STEP)
x1, x2 = torch.meshgrid([plot_x1 - PLOT_STEP/2, plot_x2 - PLOT_STEP/2])
x = torch.cat([x1.flatten().unsqueeze(dim = 1), x2.flatten().unsqueeze(dim = 1)], dim = 1)
plot_yhat = h.forward(x).reshape([plot_x1.size()[0], plot_x2.size()[0]])

ax.pcolormesh(x1, x2, plot_yhat, cmap = ListedColormap(COLORS[0:h.K]), alpha = 0.5, shading = 'auto')

plt.show()
print()