# Import packages
import numpy as np

# Function to split parameters into weight matrices and bias vectors
def par_split(par, n_hidden):
    splits = n_hidden * (np.arange(3) + 1)
    W0 = par[:splits[0]].reshape(1, n_hidden)
    b0 = par[splits[0]:splits[1]].reshape(1, n_hidden)
    W1 = par[splits[1]:splits[2]].reshape(n_hidden, 1)
    b1 = par[splits[2]:].reshape(1, 1)
    return(W0, b0, W1, b1)

