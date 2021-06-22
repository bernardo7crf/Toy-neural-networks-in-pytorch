# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Code.plotting import *

# Functions for neural ODE
    
# Create batch of input-output pairs for now
def get_batch(batch_size, x, y, device):
    s = torch.from_numpy(np.random.choice(np.arange(101, dtype=np.int64), batch_size, replace=False))
    batch_y0 = x[s]
    batch_t = torch.Tensor(np.arange(2))
    batch_y = torch.stack([x[s], y[s]], dim=0)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

# Neural net to learn function dynamics (required for adjoint method)
class ODEFunc(nn.Module):
    def __init__(self, n_hidden):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0.1)

    def forward(self, t, y):
        return self.net(y)
    
# Function to train Neural ODE
def train_node(func, x, y, device, lr=5e-3, niters=5, batch_size=101, test_freq=5, adjoint=False):
#     Setup optimizer
    optimizer = optim.RMSprop(func.parameters(), lr=lr)

    # Import ODE integrator with or without adjoint method for backpropogating gradients
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

#     Time and loss tensors
    t = torch.Tensor(np.arange(2)).to(device)
    loss_vec = torch.zeros(niters)

#     Plot initial network and ODE outputs
    nn_data_list, ode_data_list = make_output_data(x, y, t, func, odeint, device)
    plot_outputs(nn_data_list[0], ode_data_list[0], "Initial")

#     Loop over iterations of gradient descent for ODE solver and neural net
    for itr in range(niters):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(batch_size, x, y, device)
        pred_y = odeint(func, batch_y0, t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        loss_vec[itr] = loss # Need separate to avoid random torch error
        optimizer.step()

#         Save network and ODE output for animation at regular intervals
        if itr % test_freq == 0:
            print(f"Iteration {itr}")
            with torch.no_grad():
                nn_data, ode_data = make_output_data(x, y, t, func, odeint, device)
                nn_data_list = nn_data_list + nn_data
                ode_data_list = ode_data_list + ode_data

#     Plot final ODE and network outputs
    nn_data, ode_data = make_output_data(x, y, t, func, odeint, device)
    nn_data_list = nn_data_list + nn_data
    ode_data_list = ode_data_list + ode_data
    plot_outputs(nn_data[0], ode_data[0], "Final")

#     Plot training animations and loss
    plot_animation(y, nn_data_list)
    plot_animation(y, ode_data_list, title="ODE output while training")
    plot_loss(loss_vec)
    
