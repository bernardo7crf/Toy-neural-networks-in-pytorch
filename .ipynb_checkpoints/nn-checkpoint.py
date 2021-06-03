# Import packages
import numpy as np
import torch
from tools import *
from plotting import *

# Function to initialise parameters and plot intial neuron outputs
def init_par(n_hidden, x, y):
    n_wts = n_hidden * 2
    n_bss = n_hidden + 1
#     Random uniform - bad
#     par = torch.rand((n_par, ), requires_grad=True)
#     He initialisation - good for relus, keeps variance same across layers.
#     Biases should be zero...  have to seperate.
    wts = torch.normal(
        mean=torch.zeros((n_wts, )), 
#         std=2 * torch.ones((n_par, )) / torch.sqrt(n_neurons_prev_layer), 
        std=2 * torch.ones((n_wts, ))
    )
#     bss = torch.zeros((n_bss, ))
    bss = 0.1 * torch.ones((n_bss, ))
    par = torch.cat((wts, bss))
    par.requires_grad = True
    plot_neurons(par, x, y, n_hidden, title="Initial neuron outputs")
    return par

# Loss function
def loss(par, x, y, n_hidden):
    W0, b0, W1, b1 = par_split(par, n_hidden)
    A0 = x @ W0 + b0
    N0 = torch.maximum(A0, torch.zeros_like(A0))
    N1 = N0 @ W1 + b1
    return (y - N1)**2

# Gradient descent function
def fit_nn(par, x, y, n_hidden, batch_p=0.1, lr=1e-5, 
           tol=1e-6, max_it=int(5e2), its_per_frm=10):
#     Loss, iteration counter, number of reinitialised parameters, 
#     frames for animation, random number generator
    loss_vec = torch.zeros(max_it)
    count = 0
    tot_reinit = 0
    data_list = []
    rng = np.random.default_rng()
    
    # Loop until change in loss smaller than tolerance or iteration limit reached
    while True:
        # Display progress
        if (count % 50 == 0):
            print(f"Iteration {count}")

        # Subsample data for stochastic gradient descent
        n_samp = int(101 * batch_p)
        idx = rng.choice(np.arange(101), n_samp, replace=False)
        x_samp = x[idx, 0].reshape(n_samp, 1)
        y_samp = y[idx, 0].reshape(n_samp, 1)

        # Make an animation frame at regular intervals
        if (count % its_per_frm == 0):
            data_list = data_list + [
                make_plot_data(
                    par, x, y, n_hidden, x_samp=x_samp, y_samp=y_samp, 
                    include_samps=True
                )
            ]
            
        # Update loss and step down gradient
        loss_vec[count] = torch.sum(loss(par, x_samp, y_samp, n_hidden))
        loss_vec[count].backward(retain_graph=True)
        with torch.no_grad():
            par = par - lr * par.grad
            par.requires_grad_(True)
            
#         Stop if loss not reduced
#         if (loss_vec[count - 1] - loss_vec[count] < tol):
#             print(f"Loss change smaller than tolerance on iteration {count}")
#             print(f"Loss change {loss_vec[count] - loss_vec[count - 1]}")
#             break

#         Increment counter and check if done
        count = count + 1
        if (count == max_it):
            print(f"Reached maximum number of iterations")
            break


    # Plot loss, learning, and final outputs
    plot_neurons(par, x, y, n_hidden, title="Final neuron outputs")
    plot_animation(par, x, y, n_hidden, data_list)
    plot_loss(loss_vec)

#     Return updated parameters
    return(par)