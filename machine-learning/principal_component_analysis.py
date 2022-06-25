# Topic: Unsupervised, Dimensionality Reduction
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
DATASET_SIZE = 600
DEGREE_INPUT = 3         # Maximum 4 to visualize
THRESHOLD_VARIANCE = 0.8 # The minimum amount of variance to retain
COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black', 'tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
EPSILON = 1e-6
PLOT_STEP = 0.01
pi = torch.acos(torch.zeros(1)).item()*2

# Generate dummy data from Gaussian distributions
mu = -1 + 2*torch.rand([DEGREE_INPUT])
rand_square_mat = -1 + 2*torch.rand([DEGREE_INPUT, DEGREE_INPUT])
Sigma = rand_square_mat.T @ rand_square_mat
X_raw = torch.distributions.multivariate_normal.MultivariateNormal(mu, Sigma).sample([DATASET_SIZE])

# Principle Component Analysis
## Shift data to its mean
X_mean = X_raw.mean(dim = 0)
X_hat = X_raw - X_mean
## Covariance matrix
cov_mat = (1/DATASET_SIZE)*(X_hat.T @ X_hat)
# Principal components = eigenvectors with largest eigenvalues
eigenvalues, eigenvectors = torch.linalg.eigh(cov_mat)
_, sorted = torch.sort(eigenvalues, dim = 0, descending = True)
eigenvalues, eigenvectors = eigenvalues[sorted], eigenvectors[:, sorted]
# Choosing how many eigenvectors to retain as principal components
DEGREE_PCA = torch.where(eigenvalues.cumsum(dim = 0)/eigenvalues.sum() > THRESHOLD_VARIANCE)[0][0].item() + 1
print(f'Choosing {DEGREE_PCA} principal components.')
U = eigenvectors[:, 0:DEGREE_PCA]
# Project data onto the principal components
z = X_hat @ U
X_approx = z @ U.T + X_mean

# Visualize results
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.. This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Self-note: Thanks stranger on the Internet.
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_box_aspect((1, 1, 1))

if DEGREE_INPUT == 2:
    # 2D graph of input and PCA
    fig, ax = plt.subplots(constrained_layout = True)
    ax.scatter(X_raw[:, 0], X_raw[:, 1],
               s = 5, color = 'blue', alpha = 0.3)
    ax.scatter(X_approx[:, 0], X_approx[:, 1],
               s = 5, color = 'red', alpha = 0.2)
    # Expected values and principal components
    ax.scatter(X_mean[0], X_mean[1],
               s = 40, color = 'black', zorder = 100)
    ax.axline(xy1 = (X_mean[0], X_mean[1]), slope = U[1]/U[0],
              linewidth = 0.5, color = 'black')
    
    linestyle = ['solid']*DEGREE_PCA + ['dashdot']*(DEGREE_INPUT - DEGREE_PCA)
    ax.quiver(X_mean[0].repeat([DEGREE_INPUT]).numpy(), X_mean[1].repeat([DEGREE_INPUT]).numpy(),                                         # Starting point of vectors
            #   eigenvalues.sqrt()*eigenvectors[0, :]*cov_mat.diag().norm(),  # Directions of vectors
            #   eigenvalues.sqrt()*eigenvectors[1, :]*cov_mat.diag().norm(),
              eigenvalues.sqrt()*eigenvectors[0, :].numpy(),
              eigenvalues.sqrt()*eigenvectors[1, :].numpy(),
              linestyle = linestyle, color = 'black', zorder = 100, scale = 10)
    
    # Set x:y aspect ratio 1:1 so that projection can be seen clearer
    ax.set_aspect('equal') # , adjustable = 'box'
    
elif DEGREE_INPUT == 3:
    # 3D graph of input and PCA
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, constrained_layout = True)
    # Inputs and their approximate (projection onto principal components)
    ax.scatter(X_raw[:, 0], X_raw[:, 1], X_raw[:, 2],
               s = 5, color = 'blue', alpha = 0.3)
    ax.scatter(X_approx[:, 0], X_approx[:, 1], X_approx[:, 2],
               s = 5, color = 'red', alpha = 0.3)

    # Data mean & Principal components
    ax.scatter(X_mean[0], X_mean[1], X_mean[2], color = 'black', s = 40, zorder = 100)

    linestyle = ['solid']*DEGREE_PCA + ['dashdot']*(DEGREE_INPUT - DEGREE_PCA)
    ax.quiver(X_mean[0], X_mean[1], X_mean[2],                              # Starting point of vectors
            #   eigenvalues.sqrt()*eigenvectors[0, :]*cov_mat.diag().norm(),  # Directions of vectors
            #   eigenvalues.sqrt()*eigenvectors[1, :]*cov_mat.diag().norm(),
            #   eigenvalues.sqrt()*eigenvectors[2, :]*cov_mat.diag().norm(),
              eigenvalues.sqrt()*eigenvectors[0, :]*cov_mat.diag().norm(),  # Directions of vectors
              eigenvalues.sqrt()*eigenvectors[1, :]*cov_mat.diag().norm(),
              eigenvalues.sqrt()*eigenvectors[2, :]*cov_mat.diag().norm(),
              linewidth = 2, linestyle = linestyle, color = 'black', zorder = 100)
          
    # Make aspect ratio x:y:z = 1:1:1
    set_axes_equal(ax)
    if DEGREE_PCA == 2:
        # Plane of 2 principal components
        x_plot, y_plot = torch.meshgrid([torch.tensor(ax.get_xlim(), dtype = torch.float),
                                         torch.tensor(ax.get_ylim(), dtype = torch.float)])
        # Normal vector of the plane
        nvec = torch.cross(U[:, 0], U[:, 1])
        z_plot = -(nvec[0]*x_plot + nvec[1]*y_plot)/nvec[2]
        ## The equivalent offset of X_mean in z-axis (so that the surface stays at xlim, ylim)
        ## Reminder ||nvec|| == ||z_unit|| == 1
        z_unit = torch.tensor([0, 0, 1], dtype = torch.float)
        cos_X_mean_nvec = (X_mean.T @ nvec)/((X_mean**2).sum().sqrt())
        cos_nvec_z = nvec.T @ z_unit      
        z_offset = (X_mean**2).sum().sqrt() * cos_X_mean_nvec / cos_nvec_z
        z_plot = z_plot + z_offset

        ax.plot_surface(x_plot.numpy(), y_plot.numpy(), z_plot.numpy(),
                        color = 'green', alpha = 0.2)

    if DEGREE_PCA == 1:
        # Line of 1 principal components
        # Find and extend two furthest ends of the line
        _, min_id = ((X_approx @ X_mean)).min(dim = 0)
        _, max_id = ((X_approx @ X_mean)).max(dim = 0)
        X_plot = torch.cat(
            [(X_approx[min_id, :] - 0.1*(X_approx[max_id, :] - X_approx[min_id, :])).unsqueeze(dim = 0),
            (X_approx[max_id, :] + 0.1*(X_approx[max_id, :] - X_approx[min_id, :])).unsqueeze(dim = 0)],
            dim = 0)

        ax.plot(X_plot[:, 0].numpy(), X_plot[:, 1].numpy(), X_plot[:, 2].numpy(),
                color = 'green', linewidth = 1)
    
elif (DEGREE_INPUT > 3) & (DEGREE_PCA <= 3):
    cmap = cm.get_cmap('plasma')
    fig, ax = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, constrained_layout = True)
    # Original space
    scatter_X_raw = ax[0].scatter(X_raw[:, 0], X_raw[:, 1], X_raw[:, 2], c = X_raw[:, 3],
                                  alpha = 0.4, s = 5, cmap = cmap) # , 
    ax[0].scatter(X_mean[0], X_mean[1], X_mean[2], c = X_mean[3],
                  s = 40, zorder = 100)
    linestyle = ['solid']*DEGREE_PCA + ['dashdot']*(DEGREE_INPUT - DEGREE_PCA)
    ax[0].quiver(X_mean[0], X_mean[1], X_mean[2],                              # Starting point of vectors
              eigenvalues.sqrt()*eigenvectors[0, :]*cov_mat.diag().norm(),  # Directions of vectors
              eigenvalues.sqrt()*eigenvectors[1, :]*cov_mat.diag().norm(),
              eigenvalues.sqrt()*eigenvectors[2, :]*cov_mat.diag().norm(),
              colors = [cmap(eigenvectors[3, d].item()) for d in range(DEGREE_INPUT)],
              linewidth = 2, linestyle = linestyle, zorder = 100)
    
    # Colorbar
    fig.colorbar(scatter_X_raw, ax = ax[0],
                 use_gridspec = True, location = 'left', fraction = 0.03, pad = 0.01)
    
    # Projection onto principal components
    ax[1].scatter(z[:, 0], z[:, 1], z[:, 2],
                  s = 5, color = 'red') # , alpha = 0.4

    set_axes_equal(ax[0])
    set_axes_equal(ax[1])
fig.suptitle(f'Principle Component Analysis reducing data dimension from {DEGREE_INPUT} to {DEGREE_PCA}\n' +
             f'Retaining {(eigenvalues[0:DEGREE_PCA].sum()/eigenvalues.sum() * 100).item():.2f}% variance')

plt.show()
print()