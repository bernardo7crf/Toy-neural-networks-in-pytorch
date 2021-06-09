# Import packages
import numpy as np
import torch
from plotting import *
import torch.nn as nn
import torch.optim as optim

# Functions for gradient descent using torch.nn    
    
# Neural network with one hidden layer
class ToyNN(nn.Module):
    def __init__(self, n_hidden):
        super(ToyNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# Function to take one step down gradient
def grad_step(x, y, model, loss_fn, optimizer):
    # Compute prediction error
    pred = model(x)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return pred, loss

# Function to do gradient descent and plot results
def grad_desc(x, y, model, n_data, lr=1e-3, max_it=int(1e2), plot_every=40):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    pred_vec = torch.zeros((max_it, n_data))
    loss_vec = torch.zeros(max_it)
    data_list = []

    for it in range(max_it):
        pred, loss = grad_step(x, y, model, loss_fn, optimizer)
        pred_vec[it, :], loss_vec[it] = pred[:, 0], loss
        if (it % (2 * plot_every) == 0):
            print(f"Iteration {it + 1}")
        if (it % plot_every == 0):
            data_list = data_list + [[
                go.Scatter(x=x.cpu()[:, 0], y=y.cpu()[:, 0]),
                go.Scatter(x=x.cpu()[:, 0], y=model(x).detach().cpu()[:, 0])
            ]]

    go.Figure(data_list[0], dict(title="Initial network output")).show()
    plot_animation(y.cpu(), data_list, showlegend=False, title='Network output while training')
    plot_loss(loss_vec)