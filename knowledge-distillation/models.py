import torch

# Teacher: 2 hidden layers of 1200 nodes each
class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28*28, 1200),             # Input size of MNIST = 28*28
            torch.nn.ReLU(),
            torch.nn.Linear(1200, 1200),
            torch.nn.ReLU(),
            torch.nn.Linear(1200, 10),                # Output size of MNIST = 10
        )
        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

# Student: 2 hidden layers of 800 nodes each
class Student(torch.nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28*28, 800),              # Input size of MNIST = 28*28
            torch.nn.ReLU(),
            torch.nn.Linear(800, 800),
            torch.nn.ReLU(),
            torch.nn.Linear(800, 10),                 # Output size of MNIST = 10
        )
        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits