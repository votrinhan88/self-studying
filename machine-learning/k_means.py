# Topic: Unsupervised, Clustering

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils_data import get_2D_clusters

# Constants
NUM_CLUSTERS = 4
K = 8
COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black', 'tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
PLOT_STEP = 0.01

# K-means Algorithm
class KMeansClassifier():
    def __init__(self, K, X_train):
        self.K = K
        self.centroids:torch.Tensor = X_train[torch.randperm(X_train.size()[0])][range(self.K)]
    
    # Fix centroids, update labels
    def forward(self, X_train:torch.Tensor):
        distance = torch.zeros([X_train.size()[0], self.K])
        for k in torch.arange(self.K):
            distance[:, k] = torch.sqrt(torch.sum((self.centroids[k, :] - X_train)**2, dim = 1))
        
        (_, yhat) = torch.min(distance, dim = 1, keepdim = True)        
        return yhat

    # Fix labels, update centroids
    def backward(self, X_train:torch.Tensor, yhat:torch.Tensor):
        for k in torch.arange(self.K):
            self.centroids[k, :] = torch.mean(X_train[yhat.squeeze() == k, :], dim = 0)

X_train = get_2D_clusters(num_clusters=NUM_CLUSTERS)[0]

# Init
h = KMeansClassifier(K, X_train)
centroids_log = h.centroids.clone().unsqueeze(dim = 0)
# Training loop
for epoch in torch.arange(20):
    # Fix centroids, update labels
    yhat = h.forward(X_train)
    # Fix labels, update centroids
    h.backward(X_train, yhat)
    # Check for stopping condition: when centroids stop moving
    if (centroids_log[-1, :, :] == h.centroids).prod().bool() == True:
        print(f'Stopped at epoch {epoch.item()}! Centroids stop moving.')
        break
    # Log centroids position
    centroids_log = torch.cat([centroids_log, h.centroids.unsqueeze(dim = 0)], dim = 0)

# Plot all centroids and examples 
fig, ax = plt.subplots()
ax.set_title(f'K-means clustering (K = {K}) with {NUM_CLUSTERS} given clusters')
for k in torch.arange(h.K):
    # Centroids' path
    ax.plot(centroids_log[:, k, 0], centroids_log[:, k, 1],
            color = COLORS[k], linestyle = 'dashed')
    ax.scatter(centroids_log[:, k, 0], centroids_log[:, k, 1],
               color = COLORS[k], marker = 'x')
    # Training data
    ax.scatter(X_train[yhat.squeeze() == k, 0], X_train[yhat.squeeze() == k, 1],
               color = COLORS[k], alpha = 0.7, s = 10, zorder = 100)

# Voronoi diagram for centroids
ptp_X = X_train.max(dim = 0)[0] - X_train.min(dim = 0)[0]
plot_x1 = torch.arange(X_train[:, 0].min() - 0.2*ptp_X[0], X_train[:, 0].max() + 0.2*ptp_X[1], PLOT_STEP)
plot_x2 = torch.arange(X_train[:, 1].min() - 0.2*ptp_X[0], X_train[:, 1].max() + 0.2*ptp_X[1], PLOT_STEP)
x1, x2 = torch.meshgrid([plot_x1 - PLOT_STEP/2, plot_x2 - PLOT_STEP/2])
x = torch.cat([x1.flatten().unsqueeze(dim = 1), x2.flatten().unsqueeze(dim = 1)], dim = 1)
plot_yhat = h.forward(x).reshape([plot_x1.size()[0], plot_x2.size()[0]])

ax.pcolormesh(x1, x2, plot_yhat, cmap = ListedColormap(COLORS[0:h.K]), alpha = 0.5, shading = 'auto')

plt.show()
print()