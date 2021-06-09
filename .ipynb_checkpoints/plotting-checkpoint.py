# Import packages
import numpy as np
import torch
import torch.distributions as td    # PyTorch's probability distributions package
import plotly.graph_objects as go
from tools import *

# Function to combine lines for neuron output animation
def make_plot_data(
    par, x, y, n_hidden, include_hidden=False, x_samp=None, y_samp=None, 
    include_samps=False
):
    # Find neuron outputs
    x, y = x.cpu(), y.cpu()
    par_np = par.detach().cpu().numpy()
    W0, b0, W1, b1 = par_split(par_np, n_hidden)
    A0 = x @ W0 + b0
    N0 = torch.maximum(A0, torch.zeros_like(A0))
    N1 = N0 @ W1 + b1

    # Make plot data for target function, neurons in hidden layer, and output
    data = [go.Scatter(x=x[:, 0], y=y[:, 0], name="y")]
    if (include_samps):
        x_samp, y_samp = x_samp.cpu(), y_samp.cpu()
        data = data + [go.Scatter(
            x=x_samp[:, 0], y=y_samp[:, 0], mode='markers', marker_color='black',
            name=f"batch samples"
        )]
    if (include_hidden):
        for n in np.arange(n_hidden):
            data = data + [go.Scatter(
                x=x[:, 0], y=N0[:, n], line_color='grey', 
                name=f"N0{n} = {W0[0, n]:.2f} * x + {b0[0, n]:.2f}"
            )]
    data = data + [go.Scatter(x=x[:, 0], y=N1[:, 0], line_color='red', name=f"N10")]    
    return data

# Function to plot neuron outputs
def plot_neurons(par, x, y, n_hidden, title):
    data = make_plot_data(par, x, y, n_hidden, include_hidden=True)
    layout = dict(
        title=title, xaxis_title="x", 
        yaxis_title='outputs', autosize=False, width=600, height=400
    )
    fig = go.Figure(data=data, layout=layout)
    y_np = y.detach().cpu().numpy()
    fig.update_yaxes(range=[np.min(y_np) - 0.1, np.max(y_np) + 0.1])
    fig.update_traces(hoverinfo='skip')
    fig.show()

# Function to plot animation of neuron outputs
def plot_animation(y, data_list, showlegend=True, title='Network output while training'):
#     Buttons for animation
    play_but = dict(
        label="Play", method="animate", 
        args=[None, {
            "frame": {"duration": 100, "redraw": False},
            "fromcurrent": True, 
            "transition": {"duration": 0,"easing": "quadratic-in-out"}
        }]
    )
    pause_but = dict(
        label="Pause", method="animate",
        args=[[None], {
            "frame": {"duration": 0, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 0}
        }]
    ) 
    
#     Frames for animations
    frame_list = []
    for data in data_list:
        frame_list = frame_list + [go.Frame(data=data)]
    
#     Make animation
    fig = go.Figure(
        data = data_list[0], 
        layout = go.Layout(
            autosize=False, width=600, height=400, xaxis_title="x", 
            title=title, yaxis_title='outputs', 
            updatemenus=[{
                "type": "buttons", 
                "buttons": [play_but, pause_but],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        ),
        frames = frame_list
    )
    y_np = y.detach().cpu().numpy()
    fig.update_yaxes(range=[np.min(y_np) - 0.1, np.max(y_np) + 0.1])
    fig.update_traces(hoverinfo='skip')
    fig.update_layout(showlegend=showlegend)
    fig.show()

# Function to plot loss
def plot_loss(loss_vec):
    layout = dict(
        title='Batch loss while training', xaxis_title="Iteration", 
        yaxis_title='Loss', autosize=False, width=600, height=400,
    )
    fig = go.Figure(data=go.Scatter(y=loss_vec.detach()), layout=layout)
    fig.update_traces(hoverinfo='skip')
    fig.show()
    
def make_samps_data(y_dist, q_samples, x, y, n_hidden, n_samples):
    # Plot posterior predictive samples of y and network outputs
    y_samples = y_dist.rsample(sample_shape=torch.Size([50]))
    data = []
    for s in range(n_samples):
        W0, b0, W1, b1 = par_split(q_samples[s, :], n_hidden)
        A0 = x @ W0 + b0
        N0 = torch.maximum(A0, torch.zeros_like(A0))
        N1 = N0 @ W1 + b1
        data = data + [
            go.Scatter(
                x=x[:, 0], y=y_samples[s, :].numpy(), mode='markers', 
                marker_color='blue', opacity=0.1
            ), go.Scatter(
                x=x[:, 0], y=N1[:, 0].detach(), mode='markers', 
                marker_color='red', opacity=0.1
            )
        ]
    return data
    
def plot_samps(data):
    layout = dict(
        title="Samples from target and neural network", xaxis_title="x", 
        yaxis_title='outputs', autosize=False, width=600, height=400
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(hoverinfo='skip')
    fig.update_layout(showlegend=False)
    fig.show()
    
# Make data to plot output with implied target function
def make_one_output_data(x, device, output, target):
    return [
        go.Scatter(x=x[:, 0], y=output, line_color='red'), 
        go.Scatter(x=x[:, 0], y=target, line_color='blue')
    ]

def make_output_data(x, y, t, func, odeint, device):
    ode_out = odeint(func, x.to(device), t).cpu().detach().numpy()[1, :][:, 0]
    return [make_one_output_data(
        x, device, output=func(0, x.to(device)).cpu().detach()[:, 0], 
        target=y[:, 0] - x[:, 0]
    )], [make_one_output_data(x, device, output=ode_out, target=y[:, 0])]

def plot_outputs(nn_data, ode_data, title_val):
    layout = dict(
        title=f"{title_val} network output", xaxis_title="x", 
        yaxis_title='outputs', autosize=False, width=600, height=400
    )
    fig = go.Figure(data=nn_data, layout=layout)
    fig.update_traces(hoverinfo='skip')
    fig.show()
    layout.update({'title': f"{title_val} ODE output"})
    fig = go.Figure(data=ode_data, layout=layout)
    fig.update_traces(hoverinfo='skip')
    fig.show()
    