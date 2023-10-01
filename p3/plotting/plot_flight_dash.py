# P3
from p3.plotting.utility import create_subplots
from p3.plotting.plot_trajectories import plot_on_flat_plane, plot_flight_state
def plot_flight_dash(df, vars_to_plot, fig=None, row=1, col=1):
    if fig is None:
        fig, rows, cols = create_subplots(len(vars_to_plot), reserve_top_left=True)

    fig = plot_on_flat_plane(df, fig=fig, row=1, col=1)
    fig = plot_flight_state(df, vars_to_plot, fig=fig, rows=rows, cols=cols)

    return fig