# Import packages
import numpy as np
import torch

# Function to initialise parameters and plot intial neuron outputs
def init_par(n_hidden, x, y):
# #     He initialisation - good for relus, keeps variance same across layers.
#     W0 = torch.normal(
#         mean=torch.zeros(n_hidden), 
#         std=2 * torch.ones(n_hidden)
#     )
#     W1 = torch.normal(
#         mean=torch.zeros(n_hidden), 
#         std=2 * torch.ones(n_hidden) / np.sqrt(n_hidden), 
#     )
# #     Biases should be zero but seems to do better by seperating knots with 
# #     non-zero
#     n_bss = n_hidden + 1
#     bss = 0.1 * torch.ones(n_bss)
#     par = torch.cat((W0, W1, bss))

#     Pytorch default U(-sqrt(k), sqrt(k)), where k = number of inputs to layer
    W0 = torch.rand(n_hidden) * 2 - 1
    W1 = (torch.rand(n_hidden) * 2 - 1) / np.sqrt(n_hidden)
    b0 = torch.rand(n_hidden) * 2 - 1
    b1 = (torch.rand(1) * 2 - 1) / np.sqrt(n_hidden)
    par = torch.cat((W0, W1, b0, b1))
    
    return par

# Function to split parameters into weight matrices and bias vectors
def par_split(par, n_hidden):
    splits = n_hidden * (np.arange(3) + 1)
    W0 = par[:splits[0]].reshape(1, n_hidden)
    W1 = par[splits[0]:splits[1]].reshape(n_hidden, 1)
    b0 = par[splits[1]:splits[2]].reshape(1, n_hidden)
    b1 = par[splits[2]:].reshape(1, 1)
    return(W0, b0, W1, b1)

