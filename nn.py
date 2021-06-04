# Import packages
import numpy as np
import torch
import torch.distributions as td    # PyTorch's probability distributions package
from tqdm.notebook import trange    # progress bars
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
def grad_desc(par, x, y, n_hidden, batch_p=0.1, lr=1e-5, 
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
    plot_animation(y, data_list)
    plot_loss(loss_vec)

#     Return updated parameters
    return(par)

# Evidence lower bound 
def elbo(
    q_samples: torch.Tensor, x, y, y_dist: td.Distribution, q_dist: td.Distribution,
    n_samples: int, n_hidden: int
) -> torch.Tensor:
    y_given_q_samps = torch.zeros((n_samples, 101))
    for s in range(n_samples):
        W0, b0, W1, b1 = par_split(q_samples[s, :], n_hidden)
        A0 = x @ W0 + b0
        N0 = torch.maximum(A0, torch.zeros_like(A0))
        N1 = N0 @ W1 + b1
        y_given_q_samps[s, :] = N1[:, 0]
        
    n_par = 3 * n_hidden + 1
    prior_dist = td.MultivariateNormal(
        loc=torch.zeros(n_par), covariance_matrix=100 * torch.eye(n_par)
    )

    return (q_dist.log_prob(q_samples) - y_dist.log_prob(y_given_q_samps) - 
            prior_dist.log_prob(q_samples)).mean()

# Variational inference
def var_inf(
    q_mean, log_q_sd, x, y, n_hidden, n_iterations = 500, plot_every = 100, 
    n_samples = 20
):
    y_dist = td.MultivariateNormal(loc=y[:, 0], covariance_matrix=0.01 * torch.eye(101))
    opt = torch.optim.Adam([q_mean, log_q_sd], lr=1e-1)
    losses = torch.zeros(n_iterations)
    pbar = trange(n_iterations)
    data_list = []

    for t in pbar:
        opt.zero_grad()
        q_dist = td.MultivariateNormal(loc=q_mean, covariance_matrix=torch.diag(torch.exp(log_q_sd)))
        q_samples = q_dist.rsample([n_samples])

        if (t % plot_every == 0):
            data = make_samps_data(y_dist, q_samples, x, y, n_hidden, n_samples)
            data_list = data_list + [data]
#             plot_samps(data)

        loss = elbo(q_samples, x, y, y_dist, q_dist, n_samples, n_hidden)
        loss.backward()
        opt.step()
        losses[t] = loss.item()
        pbar.set_postfix(loss=loss.item())
        
    plot_animation(y, data_list, showlegend=False)