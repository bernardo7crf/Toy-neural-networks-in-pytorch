# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tnnpt.plotting import *

# Functions for gradient descent using torch.nn    
    
# Neural network with one hidden layer
class Hidden(nn.Module):
    def __init__(self, n_hidden, grid=False):
        super(Hidden, self).__init__()
        linear = nn.Linear(1, n_hidden)
        if (grid):
            linear.weight.data[::2].fill_(1)
            linear.weight.data[1::2].fill_(-1)
            linear.weight.data = linear.weight.data * 1e0
            linear.bias.data = -linear.weight.data[:, 0] * torch.linspace(-1, 1, n_hidden)
        self.hidden_layer = nn.Sequential(
            linear,
            nn.ReLU()
        )

    def forward(self, x):
        return self.hidden_layer(x)

# Neural network with one hidden layer
class ToyNN(nn.Module):
    def __init__(self, n_hidden, grid=False):
        super(ToyNN, self).__init__()
        self.hidden = Hidden(n_hidden, grid)
        self.output = nn.Linear(n_hidden, 1)
        if (grid):
            self.output.weight.data.fill_(np.sqrt(1 / n_hidden))

    def forward(self, x):
        hidden = self.hidden(x)
        return hidden, self.output(hidden)

# Function to take one step down gradient
def grad_step(x, y, model, loss_fn, optimizer):
    # Compute prediction error
    _, pred = model(x)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return pred, loss

# Function to do gradient descent and plot results
def grad_desc(x, y, model, n_data, n_hidden, lr=1e-3, max_it=int(1e2), plot_every=40):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    pred_vec = torch.zeros((max_it, n_data))
    loss_vec = torch.zeros(max_it)
    data_list = []

    for it in range(max_it):
#         if (it % (500) == 0):
#             print(f"Iteration {it + 1}")
        if (it % plot_every == 0):
            hidden, pred = model(x)
            x_cpu = x.cpu()
            hidden = hidden.cpu().detach().numpy()
            data = [
                go.Scatter(x=x_cpu[:, 0], y=y.cpu()[:, 0], line_color='blue'),
                go.Scatter(x=x_cpu[:, 0], y=pred.detach().cpu()[:, 0], line_color='red')
            ]
            for n in range(n_hidden):
                data = data + [
                    go.Scatter(x=x_cpu[:, 0], y=hidden[:, n], line_color='grey')
                ]
            data_list = data_list + [data]
        pred, loss = grad_step(x, y, model, loss_fn, optimizer)
        pred_vec[it, :], loss_vec[it] = pred[:, 0], loss
        
    return data_list, loss_vec
