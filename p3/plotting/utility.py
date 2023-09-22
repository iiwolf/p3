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

# Example usage:
# fig = make_subplots(rows=2, cols=2)
# rows, cols = get_subplot_rows_cols(fig)
# print(rows, cols)

def create_subplots(N, **kwargs):
    # For N <= 2, it's just one row
    if N <= 2:
        rows, cols = 1, N
    # For N = 3, it's a row of three
    elif N == 3:
        rows, cols = 1, 3
    else:
        rows, cols = find_factors(N)
        
        # In case the number is prime or the layout could be improved
        if rows == 1:
            cols = (N // 2) + (N % 2)
            rows = 2
    
    fig = make_subplots(rows=rows, cols=cols, **kwargs)
    
    # Return expanded rows and cols
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
        return return_as(px.colors.qualitative.Plotly[0])
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