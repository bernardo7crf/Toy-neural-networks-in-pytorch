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

#         Add neuron with node at point of greatest loss at regular intervals
#         if (count % 100 == 0):
#             par, lr, n_hidden = add_neuron(par, lr, n_hidden)

#         If gradient of loss w.r.t. parameter is zero reinitialise it
#         if (np.sum(par_g == 0) > 0):
#             n_reinit = np.sum(par_g == 0)
#             new_par, key = init_par(key, n_reinit)
#             par = par.at[np.where(par_g == 0)].set(new_par)
#             tot_reinit = tot_reinit + n_reinit

