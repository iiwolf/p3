# P3
from p3.core.variable_definitions import get_display_str, FLIGHT_DASH_VARS
from p3.plotting.utility import create_subplots
from p3.plotting.plot_trajectories import plot_on_flat_plane, plot_flight_state

def plot_flight_dash(df, vars_to_plot=None, fig=None, row=1, col=1):

    if vars_to_plot is None:
        vars_to_plot = FLIGHT_DASH_VARS

    if fig is None:
        fig, rows, cols = create_subplots(
            len(vars_to_plot),
            subplot_titles=[None] + [get_display_str(attr) for attr in vars_to_plot],
            reserve_top_left=True
        )

    fig = plot_on_flat_plane(df, fig=fig, row=1, col=1)
    fig = plot_flight_state(df, vars_to_plot, fig=fig, rows=rows, cols=cols)

    return fig