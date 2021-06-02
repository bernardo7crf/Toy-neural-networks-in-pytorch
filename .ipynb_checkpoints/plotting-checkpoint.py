# Import packages
import numpy as np
import torch
import plotly.graph_objects as go
from tools import *

# Function to combine lines for neuron output animation
def make_plot_data(par, x, y, n_hidden, include_hidden=False):
    # Find neuron outputs
    par_np = par.detach().numpy()
    W0, b0, W1, b1 = par_split(par_np, n_hidden)
    A0 = x @ W0 + b0
    N0 = torch.maximum(A0, torch.zeros_like(A0))
    N1 = N0 @ W1 + b1

    # Make plot data for target function, neurons in hidden layer, and output
    data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]
    if (include_hidden):
        for n in np.arange(n_hidden):
            data = data + [go.Scatter(
                x=x[:, 0], y=N0[:, n], line_color='grey', 
                name=f"N0{n} = {W0[0, n]:.2f} * x + {b0[0, n]:.2f}"
            )]
    data = data + [go.Scatter(x=x[:, 0], y=N1[:, 0], line_color='red', name=f"N10")]    
    return data

# Function to plot neuron outputs
def plot_neurons(par, x, y, n_hidden):
    data = make_plot_data(par, x, y, n_hidden, include_hidden=True)
    layout = dict(
        title='Two-layer neuron outputs', xaxis_title="x", 
        yaxis_title='outputs', autosize=False, width=600, height=400
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(hoverinfo='skip')
    fig.show()

# Function to make frame for animation
def make_frame(par, x, y, n_hidden):
    data = make_plot_data(par, x, y, n_hidden)
    layout = dict(
            title='Two-layer neuron outputs', xaxis_title="x", 
            yaxis_title='outputs', autosize=False, width=600, height=400
    )
    return go.Frame(data=data, layout=layout)

# Function to plot animation of neuron outputs
def plot_animation(par, x, y, n_hidden, frame_list):
    # Buttons for animation
    play_but = dict(
        label="Play", method="animate", 
        args=[None, {"transition": {"duration": 0}, "frame": {"duration": 500}}]
    )
    pause_but = dict(
        label="Pause", method="animate",
        args=[None, {"frame": {"duration": 0, "redraw": False},
                     "mode": "immediate", "transition": {"duration": 0}}]
    ) 
    
    # Make animation
    fig = go.Figure(
        data = make_plot_data(par, x, y, n_hidden), 
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