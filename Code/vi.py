# Import packages
import numpy as np
import torch
import torch.distributions as td    # PyTorch's probability distributions package
from tqdm.notebook import trange    # progress bars
import torch.nn as nn
import torch.optim as optim
from Code.tools import *
from Code.plotting import *

# Functions for variational inference

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
    n_samples = 20, lr = 1e-2
):
    y_dist = td.MultivariateNormal(loc=y[:, 0], covariance_matrix=0.01 * torch.eye(101))
    opt = torch.optim.Adam([q_mean, log_q_sd], lr=lr)
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
    
