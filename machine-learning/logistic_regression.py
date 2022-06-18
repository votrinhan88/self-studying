# Logistic Regression & Perceptron

import torch
import matplotlib.pyplot as plt

# Constants
DATASET_SIZE = 1000
NUM_CLASSES = 2

LEARNING_RATE = 1
MOMENTUM = 0.95

# Generate dummy data
torch.manual_seed(17)

# XOR gate
# x = torch.rand([DATASET_SIZE, 2])
# y = ((x[:, 0] >= 0.5) != (x[:, 1] >= 0.5)).type(torch.int)

# Two clusters in 2D
mu = torch.tensor([[0, 0], [2, 1]]).type(torch.float)
sigma = 0.4*torch.tensor([[1, 0], [0, 1]]).type(torch.float)

num_classes = torch.tensor(NUM_CLASSES)
examples_per_class = (torch.tensor(DATASET_SIZE)/num_classes).type(torch.int)

x1 = torch.distributions.multivariate_normal.MultivariateNormal(mu[0, :], sigma).sample([examples_per_class])
x2 = torch.distributions.multivariate_normal.MultivariateNormal(mu[1, :], sigma).sample([examples_per_class])
y = torch.cat([
    torch.zeros([examples_per_class, 1]),
    torch.ones([examples_per_class, 1]),
], dim = 0)

dataset = torch.cat([torch.cat([x1, x2], dim = 0), y], dim = 1)
dataset = dataset[torch.randperm(dataset.size()[0])]
x = dataset[:, 0:num_classes]
y = dataset[:, num_classes].type(torch.int).unsqueeze(dim = 1)

fig, ax = plt.subplots()

ax.scatter(x[:, 0][y.squeeze() == 0], x[:, 1][y.squeeze() == 0], color = 'blue', s = 10)
ax.scatter(x[:, 0][y.squeeze() == 1], x[:, 1][y.squeeze() == 1], color = 'red', s = 10)



# Make model
class LogisticRegressor():
    def __init__(self, degree):
        self.degree = degree

        # Parameters (init in half interval [-0.1, 0.1))
        self.w = (0.2*torch.rand([self.degree]) - 0.1)
        self.b = (0.2*torch.rand([1]) - 0.1)

    def forward(self, input):
        z = torch.matmul(input, self.w).unsqueeze(dim = 1) + self.b
        yhat = torch.sigmoid(z)
        return yhat

class Perceptron():
    def __init__(self, degree):
        self.degree = degree

        # Parameters (init in half interval [-0.1, 0.1))
        self.w = (0.2*torch.rand([self.degree]) - 0.1)
        self.b = (0.2*torch.rand([1]) - 0.1)

    def forward(self, input):
        z = torch.matmul(input, self.w).unsqueeze(dim = 1) + self.b
        yhat = (z >= 0).type(torch.int)
        return yhat

h = LogisticRegressor(degree = NUM_CLASSES)

# Negative Log Likelihood Loss
# Reminder: PyTorch's implementation intakes log-probabilities. This intakes probabilities.
def criterion(input, target):
    log_input = torch.log(input)
    loss = -(target*log_input + (1 - target)*(1 - log_input))
    return loss

# Training loop
grad_b = torch.zeros(h.b.size())
grad_w = torch.zeros(h.w.size())

for epoch in range(10000):
    yhat = h.forward(x)
    loss = torch.mean(criterion(yhat, y))

    prev_grad_b = grad_b
    prev_grad_w = grad_w

    # Calculate gradients
    grad_b = torch.sum((yhat - y))/DATASET_SIZE
    for j in torch.arange(h.w.size()[0]):
        grad_w[j] = torch.sum((yhat - y)*x[:, j].unsqueeze(dim = 1))/DATASET_SIZE
    
    # Update gradients
    h.b = h.b - LEARNING_RATE*(MOMENTUM*prev_grad_b + grad_b)
    h.w = h.w - LEARNING_RATE*(MOMENTUM*prev_grad_w + grad_w)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss = {loss}, Parameters = {h.w, h.b}')

        if 'checkpoint_loss' in locals():
            if loss == checkpoint_loss:
                print('Early stopped.')
                break
        checkpoint_loss = loss

x1_plot = torch.arange(0, 2, 0.01)
x2_plot = -h.w[0]/h.w[1]*x1_plot - h.b/h.w[1]

ax.plot(x1_plot, x2_plot)
plt.show()