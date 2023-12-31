import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# P3
from p3.core.utility import resample_df, create_buffered_range_df
from p3.core.variable_definitions import get_display_str, get_hover_str
from p3.plotting.utility import (
    create_specs,
    create_subplots,
    get_colors,
    get_plotly_colors,
    update_subplot_title_fontsize,
    hex_to_rgb,
    get_subplot_rows_cols
)


DEFAULT_LINE_COLOR = get_plotly_colors()['cyan']
    
def plot_on_flat_plane(data, **kwargs):

    if isinstance(data, list):
        return _plot_on_flat_plane_multi(data, **kwargs)
    elif isinstance(data, pd.DataFrame):
        return _plot_on_flat_plane(data, **kwargs)
    
def _plot_on_flat_plane_multi(dfs, fig=None, **kwargs):
    colors = get_colors(len(dfs))
    for df in dfs:
        fig = _plot_on_flat_plane(df, fig=fig, line_color=next(colors), **kwargs)
    return fig

def _plot_on_flat_plane(
        df, 
        line_color: str = DEFAULT_LINE_COLOR,
        animate: bool = False,
        fig=None, 
        row=1,
        col=1,
        **kwargs
    ):

    fig = fig or make_subplots()
    fig.add_trace(
        go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='lines',
            line_color=line_color,
            opacity=0.7,
        ), row=row, col=col
    )
    fig.update_xaxes(
        title=dict(text=get_display_str('x')),
        tickfont_size=10,
        showspikes=True,
        spikethickness=1,
        spikecolor=hex_to_rgb(line_color, opacity=0.5),
        range=create_buffered_range_df(df, 'x', 0.05), 
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title=dict(text=get_display_str('y')),
        tickfont_size=10,
        showspikes=True,
        spikethickness=1,
        spikecolor=hex_to_rgb(line_color, opacity=0.5),
        range=create_buffered_range_df(df, 'y', 0.05), 
        row=row,
        col=col,
    )

    if animate:
        fig = animate_trajectory(df, 'x', 'y', fig=fig, row=row, col=col, line_color=line_color, **kwargs)

    fig.update_layout(
        showlegend=False,
        template='plotly_dark',
    )
    return fig

def plot_flight_state(data, vars_to_plot, **kwargs):

    if isinstance(data, list):
        return _plot_flight_state_multi(data, vars_to_plot, **kwargs)
    elif isinstance(data, pd.DataFrame):
        return _plot_flight_state(data, vars_to_plot, **kwargs)
    
def _plot_flight_state_multi(dfs, vars_to_plot, fig=None):
    colors = get_colors(len(dfs))
    for df in dfs:
        fig = _plot_flight_state(df, vars_to_plot, fig=fig, line_color=next(colors))
    return fig

def _plot_flight_state(
    df, 
    vars_to_plot,
    line_color: str = DEFAULT_LINE_COLOR,
    fig = None,
    rows = None,
    cols = None,
):

    if fig is None:
        fig, rows, cols = create_subplots(
            len(vars_to_plot),
            subplot_titles=[get_display_str(attr) for attr in vars_to_plot]
        )
    elif rows is None or cols is None:
        rows, cols = get_subplot_rows_cols(fig)

    colors = [line_color] * len(vars_to_plot)

    x_range = create_buffered_range_df(df, 't', 0.05)

    # Add traces for all variables
    for i, attr in enumerate(vars_to_plot):
        fig.add_trace(
            go.Scatter(
                x=df['t'],
                y=df[attr],
                mode='lines',
                name=attr,
                hovertemplate=get_hover_str(attr),
                line_color=colors[i],
                opacity=0.7,
            ), row=rows[i], col=cols[i]
        )
        fig.update_xaxes(
            tickfont_size=10,
            row=rows[i],
            col=cols[i],
            showspikes=True,
            spikethickness=1,
            spikecolor=hex_to_rgb(colors[i], opacity=0.5),
            range=x_range,
            autorange=False
        )

        fig.update_yaxes(
            tickfont_size=10,
            row=rows[i],
            col=cols[i],
            showspikes=True,
            spikethickness=1,
            spikecolor=hex_to_rgb(colors[i], opacity=0.5),
            range=create_buffered_range_df(df, attr, 0.05), 
            autorange=False
        )

    fig.update_layout(showlegend=False, template='plotly_dark')
    fig = update_subplot_title_fontsize(fig, 12)
    return fig

def _create_point_scatter(df, x_var, y_var, line_color):
    return go.Scatter(
        x=[df[x_var].iloc[0]],
        y=[df[y_var].iloc[0]],
        mode='markers',
        marker=dict(
            size=10,
            color='black',
            line_color=line_color,
            line_width=2.0,
        )
    )

def animate_trajectory(
    df,
    x_var,
    y_var,
    speeds = [1, 2, 4, 8],
    allow_resample: bool = True,
    resample_dt: float = 0.1,
    n_samples: int = 100,
    fig: go.Figure = None,
    row=1,
    col=1,
    line_color: str = DEFAULT_LINE_COLOR,
    **kwargs
):
    """
    Visualizes a trajectory from a DataFrame with x and y columns and animates it using Plotly's slider.

    Parameters:
        df (pd.DataFrame): DataFrame containing x and y columns.
    """

    # Resample the data if necessary
    if allow_resample:
        df = resample_df(df, 't', resample_dt, n_samples=n_samples)

    # Create figure if not provided
    fig = fig or make_subplots()
    
    # Dummy trace (i.e. the trace to be animated)
    fig.add_trace(_create_point_scatter(df, x_var, y_var, line_color), row=row, col=col)

    # Create data for each frame (i.e., each step in the trajectory)
    frames = [go.Frame(
                data=[go.Scatter(
                    x=[df['x'].iloc[i]],
                    y=[df['y'].iloc[i]],
                    mode='markers')],

                # Note: important! Tells Ploty to **only** animate the trace we just made
                traces=[len(fig.data) - 1],
                name=f"{df['t'].iloc[i]:.2f} s"
            ) for i in range(len(df))]

    fig.frames = frames

    # Create the figure with an initial frame and the frames for animation
    transition_time = df['t'].iloc[-1] / len(df) * 1000
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': transition_time / speed, 'redraw': False}, 
                        'transition' : {'duration' : transition_time / speed, 'easing': 'linear'}, 
                        'fromcurrent': True
                        }],
                    'label': f'{speed}x',
                    'method': 'animate'
                } for speed in speeds
            ] + [
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 't = ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                        'transition': {'duration': 0}}],
                        'label': f.name,
                        'method': 'animate'} for k, f in enumerate(frames)]
        }],
        template='plotly_dark'
    )
    fig.update_xaxes(range=create_buffered_range_df(df, 'x', 0.05), autorange=False)
    fig.update_yaxes(range=create_buffered_range_df(df, 'y', 0.05), autorange=False)
    return fig

