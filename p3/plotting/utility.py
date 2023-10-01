from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import hex_to_rgb as hex_to_rgb_plotly
import numpy as np

def add_kilo(x, precision):
    if abs(x) > 1000:
        return f"{x / 1000:0.{precision}}"
    else:
        return f"{x:0.}"
        
def update_font_size(fig, font_size):
    """
    Update the font size of all axis labels and ticks in a Plotly figure.
    
    Parameters:
    - fig: The Plotly figure to update.
    - font_size: The desired font size.
    """
    
    # Update the x-axis
    fig.update_layout(
        xaxis=dict(
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size)
        )
    )
    
    # Update the y-axis
    fig.update_layout(
        yaxis=dict(
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size)
        )
    )
    
    # If the figure has a secondary y-axis, update that as well
    if 'yaxis2' in fig['layout']:
        fig.update_layout(
            yaxis2=dict(
                title_font=dict(size=font_size),
                tickfont=dict(size=font_size)
            )
        )

    return fig

def create_specs(n_row, n_col, specs=None):
    specs = specs or {}
    specs = [[specs] * n_col] * n_row
    return specs

def find_factors(N):
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            x = i
            y = N // i
    return x, y

# Returns two lists corresponding to all row-column
# pairs in the figure
def get_subplot_rows_cols(fig):
    row_cols = np.array(list(product(*fig._get_subplot_rows_columns())))
    return row_cols[:, 0], row_cols[:, 1]


def get_square_subplot_shape(N):
    base = int(N**0.5)
    if base * base == N:
        return base, base
    for cols in range(base + 1, N + 1):
        rows, remainder = divmod(N, cols)
        if rows >= remainder:
            return rows, cols
    return N, 1


def get_subplot_shape(N, reserve_top_left=False):
    if reserve_top_left:
        # Find shape for N + 4 (2x2 reserved)
        rows, cols = get_square_subplot_shape(N + 4)
    else:
        rows, cols = get_square_subplot_shape(N)
    return rows, cols


def create_subplots(N, reserve_top_left=False, **kwargs):
    rows, cols = get_subplot_shape(N + 4, reserve_top_left)

    # If reserved, initialize the specs
    if reserve_top_left:
        specs = []
        for i in range(rows):
            row = []
            for j in range(cols):
                if i == 0 and j == 0:  # This is the top-left subplot
                    row.append({'type': 'xy', 'rowspan': 2, 'colspan': 2})
                elif i < 2 and j < 2:  # This is one of the other 3 top-left subplots
                    row.append(None)
                else:
                    row.append({})
            specs.append(row)

        fig = make_subplots(rows=rows, cols=cols, specs=specs, **kwargs)
    else:
        fig = make_subplots(rows=rows, cols=cols, **kwargs)
    
    # If reserved, skip the top-left 2x2 cells
    if reserve_top_left:
        subplot_coords = [(i, j) for i in range(1, rows + 1) for j in range(1, cols + 1) 
                          if not (i <= 2 and j <= 2)][:N]
    else:
        subplot_coords = [(i, j) for i in range(1, rows + 1) for j in range(1, cols + 1)][:N]

    rows_list, cols_list = zip(*subplot_coords)
    
    return fig, list(rows_list), list(cols_list)


def get_colors(n: int, return_as=iter):
    """
    Get a list of n colors from the Plotly default color cycle.
    
    Parameters:
    - n: The number of colors to return.
    
    Returns:
    - colors: A list of n colors.
    """

    if n == 1:
        return return_as(px.colors.qualitative.Plotly[:1])
    elif n < 10:
        return return_as(px.colors.qualitative.Plotly[:n])
    elif n < 24:
        return return_as(px.colors.qualitative.Light24[:n])
    else:
        # Create gradient from blue to red for n colors
        colors = []
        for i in range(n):
            colors.append(f"hsl({i * 360 / n}, 100%, 50%)")
        return return_as(colors)
    
def get_plotly_colors():
    """
    Get a dictionary of colors from the Plotly default color cycle.
    
    Returns:
    - plotly_colors: A dictionary of colors.
    """
    # Get the built-in Plotly color sequence from plotly.express
    colors = px.colors.qualitative.Plotly

    return {
        "blue": colors[0],
        "red": colors[1],
        "green": colors[2],
        "purple": colors[3],
        "orange": colors[4],
        "cyan": colors[5],
        "salmon": colors[6],
        "lime": colors[7],
        "pink": colors[8],
        "yellow": colors[9]
    }

def update_subplot_title_fontsize(fig, font_size):
    """
    Update the font size of subplot titles in a Plotly figure.

    Parameters:
    - fig: A Plotly figure object created using make_subplots.
    - font_size: Desired font size for subplot titles.

    Returns:
    - Updated Plotly figure with the desired font size for subplot titles.
    """
    for annotation in fig['layout']['annotations']: 
        annotation['font'] = dict(size=font_size)
    return fig

def hex_to_rgb(hex_code, opacity=1.0):
    """
    Convert a hex color code to an RGBA string using Plotly's color library.

    Parameters:
    - hex_code (str): The hex color code to convert. Should start with '#'.
    - opacity (float, optional): A value between 0 (fully transparent) and 1 (fully opaque).

    Returns:
    - str: The RGBA representation of the color.
    """
    
    r, g, b = hex_to_rgb_plotly(hex_code)
    return f'rgba({r}, {g}, {b}, {opacity})'


def _add_expanding_slider(fig, num_points_per_step=10):
    """
    Add a slider to a Plotly scatter plot.
    
    :param fig: Existing Plotly figure
    :param num_points_per_step: Number of points to show per slider step
    :return: Updated Plotly figure with a slider
    """
    
    # Extract data
    x_values = fig.data[0]['x']
    y_values = fig.data[0]['y']
    
    # Create steps for the slider
    steps = []
    for i in range(0, len(x_values), num_points_per_step):
        step = {
            'args': [{
                'x': [x_values[:i+num_points_per_step]],
                'y': [y_values[:i+num_points_per_step]]
            }],
            'label': str(i+num_points_per_step),
            'method': 'restyle'
        }
        steps.append(step)

    # Create the slider
    slider = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Number of points:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': steps
    }

    # Add the slider to the figure
    fig.update_layout(sliders=[slider])

    return fig

DEFAULT_LINE_COLOR = get_plotly_colors()['cyan']

def _add_point_slider(
    fig,
    num_points_per_step=1,
    line_color=DEFAULT_LINE_COLOR,
    transition_time = 1000,
    speeds=[1, 2, 4, 8],
    row=1,
    col=1
):
    """
    Add a slider to a Plotly scatter plot to place a dot at specified points.
    
    :param fig: Existing Plotly figure
    :param num_points_per_step: Number of points to step per slider position
    :return: Updated Plotly figure with a slider
    """

    # Create steps for the slider
    max_steps = max([len(fig.data[trace_index]['x']) for trace_index in range(len(fig.data))])
    n_traces = len(fig.data)
    rows, cols = get_subplot_rows_cols(fig)
    steps = []
    all_x_vals = []
    all_y_vals = []
    traces = []
    for i, (row, col) in enumerate(zip(rows, cols)):
            
        # Extract data
        x_values = fig.data[i]['x']
        y_values = fig.data[i]['y']
        
        all_x_vals.append(x_values)
        all_y_vals.append(y_values)

        # Ensure the base line remains and add a scatter dot trace
        fig.add_trace(
            go.Scatter(
                x=[x_values[0]],
                y=[y_values[0]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='black',
                    line_color=line_color,
                    line_width=2.0,
                )
            ), row=row, col=col
        )
        traces.append(len(fig.data) - 1)

    frames = []
    for i in range(0, max_steps, num_points_per_step):
        frame_data = []
        for x_values, y_values, trace_idx in zip(all_x_vals, all_y_vals, traces):
            frame_data.append(go.Scatter(x=[x_values[i]], y=[y_values[i]], mode='markers'))
        frames.append(go.Frame(data=frame_data, name=str(i), traces=traces))


    # Create the slider (existing code)
    slider = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Point Index:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 0, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate", "transition": {"duration": 0}}],
                   "label": f.name, "method": "animate"} for f in frames]
    }

    # Add play and pause buttons
    play_button = {
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
    }

    # Add the slider, frames, and the play button to the figure
    fig.frames = frames
    fig.update_layout(sliders=[slider], updatemenus=[play_button])

    return fig
