import torch
import matplotlib.pyplot as plt
import numpy as np

# Constants
## For dummy data
NUM_OBSERVED = 8                    # Number of observable datapoints
SIGMA_NOISE_OBSERVED = 0.05         # Standard deviation of Gaussian noise added to dummy data
## For building Prior
RANGE_EXTEND = 0.2                  # How far out of the observed domain the Posterior should extend to
SAMPLE_STEP = 0.05                  # Sampling frequency of the Posterior
LENGTH = 1                          # Parameter for Kernel
VARIANCE = 0.1                      # Parameter for Kernel
SIGMA_NOISE_POSTERIOR = 1e-3        # Parameter for Posterior
## For realization
NUM_REALIZE = 5                     # Number of functions to be realized

# Generate dummy data
x_raw = 10*torch.rand(size = [NUM_OBSERVED, 1])
y_raw = torch.sin(x_raw) + torch.distributions.normal.Normal(loc = 0, scale = SIGMA_NOISE_OBSERVED).sample(x_raw.size())

def get_x_sample(x_train, range_extend, sample_step):
    # Sample X:
    range_x = x_train.max() - x_train.min()
    x_sample = torch.arange(
        x_raw.min() - range_extend*range_x,
        x_raw.max() + range_extend*range_x + sample_step,
    sample_step).unsqueeze(dim = 1)
    return x_sample
x_sample = get_x_sample(y_raw, RANGE_EXTEND, SAMPLE_STEP)

class GaussianProcess1D:
    def __init__(self, x_train:torch.Tensor, y_train:torch.Tensor, x_sample:torch.Tensor,
                 kernel_type, variance, length, sigma_noise = 0):
        self.x_train = x_train
        # Normalize to mean = 0
        self.mean_y_train = y_train.mean()
        self.y_train      = y_train - self.mean_y_train 
        self.x_sample = x_sample
        
        # For Prior (RBF)
        self.kernel_type = kernel_type
        self.variance    = variance
        self.length      = length
        self.sigma_noise = sigma_noise

    ## Covariance/Gram/Kernel matrix
    def kernel(self, input1, input2):
        if self.kernel_type == 'RBF':
            Sigma = self.variance*torch.exp(
                (-1/(2*self.length**2) * torch.sum((input1.unsqueeze(dim = 1) - input2.unsqueeze(dim = 0))**2, dim = -1)))
        return Sigma

    # Build Prior
    def get_prior(self):
        # Assuming a mean of 0 for simplicity
        self.mu_prior = torch.zeros(self.x_sample.size()).squeeze()
        self.Sigma_prior = self.kernel(self.x_sample, self.x_sample)
        # Standard deviation (confidence interval)
        sigma_prior = torch.sqrt(self.Sigma_prior.diag())
        return self.mu_prior, sigma_prior

    # Build Posterior (the Gaussian Process)
    # [y1, y2] \sim \mathcal{N}([mu1, mu2], [[Sigma11, Sigma12], [Sigma21, Sigma22]])
    def get_posterior(self):
        ## Covariance/Gram/Kernel matrix of Observed data (11), Observed data vs. Prior (12, 21), Prior (22)
        ## The squared exponential kernel is very smooth, so if your data points are close together, the
        ## covariance matrix goes singular ==> Add a small constant along the diagonal for stability
        Sigma_11 = self.kernel(self.x_train, self.x_train)
        Sigma_11 = Sigma_11 + self.sigma_noise*torch.eye(Sigma_11.size()[0])
        Sigma_21 = self.kernel(self.x_sample, self.x_train) # == Sigma_12.T
        Sigma_22 = self.Sigma_prior if (self.Sigma_prior != None) else self.kernel(self.x_sample, self.x_sample)

        ## Mean (added back the input's mean) & Variance of posterior
        self.mu_posterior = (Sigma_21 @ Sigma_11.inverse() @ self.y_train).squeeze() + self.mean_y_train 
        self.Sigma_posterior = Sigma_22 - Sigma_21 @ Sigma_11.inverse() @ Sigma_21.T
        # Standard deviation (confidence interval)
        sigma_posterior = torch.sqrt(self.Sigma_posterior.diag())
        return self.mu_posterior, sigma_posterior

    def realize(self, choice, num_realize):
        if choice == 'prior':
            mu = self.mu_prior
            Sigma = self.Sigma_prior
        elif choice == 'posterior':
            mu = self.mu_posterior
            Sigma = self.Sigma_posterior

        # Realization (Draw samples from the prior at our data points)
        ys = np.random.multivariate_normal(mean = mu, cov = Sigma, size = num_realize)
        ys = torch.tensor(ys).T

        return ys

gp = GaussianProcess1D(x_train = x_raw, y_train = y_raw, x_sample = x_sample,
                       kernel_type = 'RBF', variance = VARIANCE, length = LENGTH, sigma_noise = SIGMA_NOISE_POSTERIOR)

# Visualize Prior
fig1, ax = plt.subplots(2, 1, sharex = True, sharey = True)
# Mean ± 2 STD
mu, sigma = gp.get_prior()
ax[0].autoscale()
ax[0].plot(x_sample.squeeze(), mu,
           color = 'blue', label = 'Prior')
ax[0].fill_between(x_sample.squeeze(), mu - 2*sigma, mu + 2*sigma,
                   color = 'red', alpha = 0.5, label = "$\pm 2\sigma$")
ax[0].legend()
ax[0].set_title('Prior $\pm 2$ STD')
# Some realizations
ys = gp.realize('prior', NUM_REALIZE)
for rlz in torch.arange(NUM_REALIZE):
    ax[1].plot(x_sample, ys[:, rlz])
ax[1].set_title(f'{NUM_REALIZE} realizations of the Prior')


# Visualize Posterior
fig2, ax = plt.subplots(2, 1, sharex = True, sharey = True)
# Mean ± 2 STD
mu, sigma = gp.get_posterior()
ax[0].autoscale()
ax[0].plot(x_sample.squeeze(), mu,
           color = 'blue', label = 'Posterior')
ax[0].fill_between(x_sample.squeeze(), mu - 2*sigma, mu + 2*sigma,
                   color = 'red', alpha = 0.5, label = "$\pm 2\sigma$")
ax[0].scatter(x_raw.squeeze(), y_raw.squeeze(),
              color = 'black', zorder = 100, label = "Observed")
ax[0].legend()
ax[0].set_title('Posterior $\pm 2$ STD')
# Some realizations
ys = gp.realize('posterior', NUM_REALIZE)
for rlz in torch.arange(NUM_REALIZE):
    ax[1].plot(x_sample, ys[:, rlz])
ax[1].scatter(x_raw.squeeze(), y_raw.squeeze(),
              color = 'black', zorder = 100)
ax[1].set_title(f'{NUM_REALIZE} realizations of the Posterior')

plt.show()
print()