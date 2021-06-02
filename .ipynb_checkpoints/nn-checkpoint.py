# Import packages
import numpy as np
import torch
from tools import *
from plotting import *

# Function to initialise parameters and plot intial neuron outputs
def init_par(n_par, x, y, n_hidden):
    par = torch.rand((n_par, ), requires_grad=True)
    plot_neurons(par, x, y, n_hidden)
    return par

# Loss function
def loss(par, x, y, n_hidden):
    W0, b0, W1, b1 = par_split(par, n_hidden)
    A0 = x @ W0 + b0
    N0 = torch.maximum(A0, torch.zeros_like(A0))
    N1 = N0 @ W1 + b1
    return (y - N1)**2

# Add neuron with node at point of greatest loss
def add_neuron(par, lr, n_hidden):
    max_loss_idx = loss_vec[count] == np.max(loss_vec[count])
    max_loss_x = x_samp[np.argmin(max_loss_idx)]
    par = np.concatenate((
        np.array([0.01]), # New input weight
        par[:n_hidden], 
        np.array(-0.01 * max_loss_x), # New input bias
        par[n_hidden:(2 * n_hidden)],
        np.array([0.01]), par[(2 * n_hidden):])) # New output weight
    lr = lr * n_hidden / (n_hidden + 1) # Reduce learning rate
    n_hidden = n_hidden + 1 # Increase number of neurons
    return par, lr, n_hidden

# Gradient descent function
def grad_desc(par, x, y, n_hidden, lr = 1e-5, tol = 1e-6, max_it = int(5e2)):
    # Loss, iteration counter, number of reinitialised parameters, frames for animation
    loss_vec = torch.zeros(max_it)
    loss_vec[0] = torch.sum(loss(par, x, y, n_hidden))
    count = 0
    tot_reinit = 0
    frame_list = []
    
    # Loop until change in loss smaller than tolerance or iteration limit reached
    while True:
        # Make an animation frame at regular intervals
        if (count % 50 == 0):
            frame_list = frame_list + [make_frame(par, x, y, n_hidden)]
            
        # Display progress
        if (count % 50 == 0):
            print(f"count {count}")

        # Subsample data for stochastic gradient descent
        n_samp = 101
        rng = np.random.default_rng()
        idx = rng.choice(np.arange(101), n_samp, replace=False)
        x_samp = x[idx, 0].reshape(n_samp, 1)
        y_samp = y[idx, 0].reshape(n_samp, 1)

#         # Add neuron with node at point of greatest loss at regular intervals
#         if (count % 100 == 0):
#             par, lr, n_hidden = add_neuron(par, lr, n_hidden)

        # Find gradient over whole dataset
        loss_vec[count].backward(retain_graph=True)
        
        # Step down gradient
        with torch.no_grad():
            par = par - lr * par.grad
            par.requires_grad_(True)
            
        # If gradient of loss w.r.t. parameter is zero reinitialise it
        # if (np.sum(par_g == 0) > 0):
        #     n_reinit = np.sum(par_g == 0)
        #     new_par, key = init_par(key, n_reinit)
        #     par = par.at[np.where(par_g == 0)].set(new_par)
        #     tot_reinit = tot_reinit + n_reinit

        # Update loss and iteration counter
        count = count + 1
        loss_vec[count] = torch.sum(loss(par, x_samp, y_samp, n_hidden))

        # If stopping condition met print it and stop
        # if (loss_vec[count - 1] - loss_vec[count] < tol):
        #     print(f"Loss change smaller than tolerance on iteration {count}")
        #     print(f"Loss change {loss_vec[count] - loss_vec[count - 1]}")
        #     break
        if (count + 1 == max_it):
            print(f"Reached maximum number of iterations")
            break

    # Print number of parameters reinitialised due to zero gradients
#     print(f"Params reinitialised due to zero gradients {tot_reinit} times")
    print(f"Final number of hidden neurons {n_hidden}")

    # Plot loss, learning, and final outputs
    plot_loss(loss_vec)
    plot_animation(par, x, y, n_hidden, frame_list)
    plot_neurons(par, x, y, n_hidden)

    # Return updated parameters
    return(par, n_hidden)