# Import packages
import numpy as np
import plotly.graph_objects as go
import torch

# Function to split parameters into matrices and vectors
def par_split(par, n_hidden_ns):
    splits = n_hidden_ns * (np.arange(3) + 1)
    W0 = par[:splits[0]].reshape(1, n_hidden_ns)
    b0 = par[splits[0]:splits[1]].reshape(1, n_hidden_ns)
    W1 = par[splits[1]:splits[2]].reshape(n_hidden_ns, 1)
    b1 = par[splits[2]:].reshape(1, 1)
    return(W0, b0, W1, b1)

# Function to initialise weights and biases
def init_par(n_par, x, y, n_hidden_ns):
    par = torch.rand((n_par, ), requires_grad=True)

    # Plot initial neuron outputs
    plot_neurons(par, x, y, n_hidden_ns)

    # Return parameters and new RNG key
    return par

# Loss function
def loss(x, par, n_hidden_ns, y):
    W0, b0, W1, b1 = par_split(par, n_hidden_ns)
    A0 = x @ W0 + b0
    N0 = torch.maximum(A0, torch.zeros_like(A0))
    N1 = N0 @ W1 + b1
    return ((y - N1)**2)

# Function to combine lines for neuron output animation
def make_plot_data(par, x, y, n_hidden_ns):
    # Find neuron outputs
    W0, b0, W1, b1 = par_split(par, n_hidden_ns)
    A0 = x @ W0 + b0
    N0 = torch.maximum(A0, torch.zeros_like(A0))
    N1 = N0 @ W1 + b1
    N1_np = N1.detach().numpy()

    # Make plot for target function
    data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]

    # Make plots for neurons in hidden layer
    # for n in np.arange(n_hidden_ns):
    #     data = data + [go.Scatter(
    #             x=x[:, 0], y=N0[:, n], line_color='grey', 
    #             name=f"N0{n} = {W0[0, n]:.2f} * x + {b0[0, n]:.2f}")]

    # Make plot for network output
    data = data + [go.Scatter(x=x[:, 0], y=N1_np[:, 0], line_color='red', name=f"N10")]

    return data

# Function to plot neuron outputs
def plot_neurons(par, x, y, n_hidden_ns):
    # Find neuron outputs
    W0, b0, W1, b1 = par_split(par, n_hidden_ns)
    A0 = x @ W0 + b0
    N0 = torch.maximum(A0, torch.zeros_like(A0))
    N1 = N0 @ W1 + b1
    W0_np = W0.detach().numpy()
    b0_np = b0.detach().numpy()
    W1_np = W1.detach().numpy()
    b1_np = b1.detach().numpy()
    N0_np = N0.detach().numpy()
    N1_np = N1.detach().numpy()

    # Make plot for target function
    data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]

    # Make plots for neurons in hidden layer
    for n in np.arange(n_hidden_ns):
        data = data + [go.Scatter(
            x=x[:, 0], y=N0_np[:, n], line_color='grey', 
            name=f"N0{n} = {W0_np[0, n]:.2f} * x_np + {b0_np[0, n]:.2f}")]

    # Make plot for network output
    data = data + [go.Scatter(x=x[:, 0], y=N1_np[:, 0], line_color='red', name=f"N10")]    

    # Setup layout
    layout = dict(
            title='Two-layer neuron outputs', xaxis_title="x", 
            yaxis_title='outputs', autosize=False, width=600, height=400
    )

    # Make plot
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(hoverinfo='skip')
    fig.show()

# Function to make frame for animation
def make_frame(par, x, y, n_hidden_ns):
    # Make plot data
    data = make_plot_data(par, x, y, n_hidden_ns)
                                                        
    # Setup layout
    layout = dict(
            title='Two-layer neuron outputs', xaxis_title="x", 
            yaxis_title='outputs', autosize=False, width=600, height=400
    )

    return go.Frame(data=data, layout=layout)

# Function to plot animation of neuron outputs
def plot_animation(par, x, y, n_hidden_ns, frame_list):
    # Buttons for animation
    play_but = dict(label="Play", method="animate", 
                                    args=[None, {"transition": {"duration": 0},
                                                 "frame": {"duration": 500}}])
    pause_but = dict(label="Pause", method="animate",
                                    args=[None, {"frame": {"duration": 0, "redraw": False},
                                                 "mode": "immediate", 
                                                 "transition": {"duration": 0}}]) 
    
    # Make animation
    fig = go.Figure(
        data = make_plot_data(par, x, y, n_hidden_ns), 
        layout = go.Layout(
                autosize=False, width=600, height=400, xaxis_title="x", 
                title='Learning animation', yaxis_title='outputs', 
                updatemenus=[dict(type="buttons", buttons=[play_but, pause_but])]
        ),
        frames = frame_list
    )
    fig.update_traces(hoverinfo='skip')
    fig.show()

# Function to plot loss
def plot_loss(loss_vec):
    layout = dict(
            title='Loss over gradient descent', xaxis_title="Iteration", 
            yaxis_title='Loss', autosize=False, width=600, height=400
    )
    fig = go.Figure(data=go.Scatter(y=loss_vec.detach()), layout=layout)
    fig.update_traces(hoverinfo='skip')
    fig.show()

# Gradient descent function
def grad_desc(par, x, y, n_hidden_ns, lr = 1e-5, tol = 1e-6, max_it = int(5e2)):
    # Loss, iteration counter, frames for animation
    loss_vec = torch.zeros(max_it)
    loss_vec[0] = torch.sum(loss(x, par, n_hidden_ns, y))
    count = 0
    tot_reinit = 0
    frame_list = []
    
    # Loop until change in loss smaller than tolerance or iteration limit reached
    while True:
        # Make an animation frame at regular intervals
        if (count % 1 == 0):
            frame_list = frame_list + [make_frame(par, x, y, n_hidden_ns)]
            
        if (count % 50 == 0):
            print(f"count {count}")

        n_samp = 101
        rng = np.random.default_rng()
        idx = rng.choice(np.arange(101), n_samp, replace=False)
        x_samp = x[idx, 0].reshape(n_samp, 1)
        y_samp = y[idx, 0].reshape(n_samp, 1)

        # # Add neuron with node at point of greatest loss at regular intervals
        # if (count % 100 == 0):
        #     # loss = vloss_jit(x_samp, par, n_hidden_ns, y_samp)
        #     loss = vloss(x_samp, par, n_hidden_ns, y_samp)
        #     max_loss_x = x_samp[np.argmin(loss == np.max(loss))]
        #     # New weight = 1, new bias = -max_loss_x, new output weight = 1
        #     par = np.concatenate((np.array([0.01]), par[:n_hidden_ns], 
        #                                                    np.array(-0.01 * max_loss_x), par[n_hidden_ns:(2 * n_hidden_ns)],
        #                                                    np.array([0.01]), par[(2 * n_hidden_ns):]))
        #     lr = lr * n_hidden_ns / (n_hidden_ns + 1)
        #     n_hidden_ns = n_hidden_ns + 1

#         print(count)
#         print(par)
#         print(loss_vec[count])

        # Find gradient over whole dataset
        loss_vec[count].backward(retain_graph=True)
        
        # Step down gradient
        with torch.no_grad():
#             par.sub_(lr * par.grad)
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
        loss_vec[count] = torch.sum(loss(x_samp, par, n_hidden_ns, y_samp))

        # If stopping condition met print it and stop
        # if (loss_vec[count - 1] - loss_vec[count] < tol):
        #     print(f"Loss change smaller than tolerance on iteration {count}")
        #     print(f"Loss change {loss_vec[count] - loss_vec[count - 1]}")
        #     break
        if (count + 1 == max_it):
            print(f"Reached maximum number of iterations")
            break

    # Print number of parameters reinitialised due to zero gradients
    print(f"Params reinitialised due to zero gradients {tot_reinit} times")
    print(f"Final number of hidden neurons {n_hidden_ns}")

    # Plot loss
    plot_loss(loss_vec)

    # Plot learning animation
    plot_animation(par, x, y, n_hidden_ns, frame_list)
    
    # Plot final outputs of all neurons
    plot_neurons(par, x, y, n_hidden_ns)

    # Return updated parameters
    return(par, n_hidden_ns)