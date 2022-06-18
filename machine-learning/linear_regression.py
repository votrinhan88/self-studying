import matplotlib.pyplot as plt
import torch

# Note:
# - Gradient descent converge SUPER SLOW for engineered features of higher powers from one orgiginal
#   feature (compared to using Normal Equation)
# - The proposed linear model fails worse at when increasing dataset degree

# Constants
DATASET_SIZE = 1000
DATASET_BOUNDS = [0, 10]
DATASET_DEGREE = 3
NOISE_STD = 1e-3#.3

MODEL_DEGREE = 3

LEARNING_RATE = 3e-6
MOMENTUM = 0.95
pi = torch.acos(torch.zeros(1)).item()*2

# Generate dummy data
torch.manual_seed(17)
def make_data(size, bounds, degree, noise_std):
    x = torch.distributions.uniform.Uniform(low = bounds[0], high = bounds[1]).sample(sample_shape = [size, 1])
    gaussian_noise = torch.distributions.normal.Normal(loc = 0, scale = noise_std)
    y = x + 2*torch.sin(((degree - 1)*pi/bounds[1])*x + torch.rand([1])*2*bounds[1]) + gaussian_noise.sample([size, 1])
    return x, y

raw_x, y = make_data(DATASET_SIZE, DATASET_BOUNDS, DATASET_DEGREE, NOISE_STD)

# Create addition features of x at higher powers
def engineer_powers(input, degree):
    output = torch.ones([input.size(0), degree + 1])
    for power in torch.arange(degree + 1):
        if power == 0:
            output[:, power] = 1
        elif power >= 1:
            output[:, power] = torch.squeeze(torch.pow(input, power))
    return output

x = engineer_powers(raw_x, MODEL_DEGREE)


# Make model
class LinearRegressor():
    def __init__(self, degree):
        self.degree = degree

        # Parameters (init in half interval [-0.1, 0.1))
        self.theta = (0.2*torch.rand([self.degree + 1]) - 0.1)
        # self.theta.requires_grad = True

    def forward(self, input):
        yhat = torch.matmul(input, self.theta).unsqueeze(dim = 1)
        return yhat
        
h = LinearRegressor(degree = MODEL_DEGREE)
criterion = torch.nn.MSELoss(reduction = 'mean')

# Training loop
theta_grad = torch.zeros(h.theta.size())

for epoch in torch.arange(200000):
    pred = h.forward(x)
    loss = criterion(pred, y)

    # Calculate gradients
    momentum = MOMENTUM*theta_grad
    for power in torch.arange(h.degree + 1):
        theta_grad[power] = 2*torch.sum((pred - y)*x[:, power].unsqueeze(dim = 1), dim = 0)
    theta_grad = theta_grad/DATASET_SIZE
    
    # Update gradients
    h.theta = h.theta - LEARNING_RATE*(momentum + theta_grad)

    if epoch % 10000 == 0:
        # LEARNING_RATE = LEARNING_RATE
        print(f'Epoch {epoch}, Loss = {loss}, Parameters = {h.theta}')
print(f'Trained Parameters = {h.theta}')


raw_xplot = torch.arange(0, 10, 0.01).unsqueeze(dim = 1)
x_plot = engineer_powers(raw_xplot, MODEL_DEGREE)

plt.scatter(raw_x, y)
print(f'Gradient Descent: Loss = {criterion(h.forward(x_plot), y)}, Parameters = {h.theta}')
plt.plot(raw_xplot, h.forward(x_plot), color = 'orange')


# Normal equation
x_trans = torch.transpose(x, 0, 1)
first = torch.inverse(torch.matmul(x_trans, x))
theta_ne = first.matmul(x_trans).matmul(y)

h_ne = LinearRegressor(MODEL_DEGREE)
h_ne.theta = torch.squeeze(theta_ne)

print(f'Normal Equation: Loss = {criterion(h_ne.forward(x_plot), y)}, Parameters = {h_ne.theta}')
plt.plot(raw_xplot, h_ne.forward(x_plot), color = 'red')
plt.show()

print()