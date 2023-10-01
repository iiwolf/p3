import pandas as pd
from unit_test import UnitTest
class TestPlotTrajectory(UnitTest):

    def test_plot_flight_state(self):
        from p3.plotting.plot_trajectories import plot_flight_state
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_flight_state(df, df.columns.difference(['t']))
        fig.show()

    def test_plot_on_flat_plane(self):
        from p3.plotting.plot_trajectories import plot_on_flat_plane
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_on_flat_plane(df)
        fig.show()

    def test_plot_on_flat_plane_animated(self):
        from p3.plotting.plot_trajectories import plot_on_flat_plane
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_on_flat_plane(df, animate=True)
        fig.show()

    def test_plot_on_flat_plane_w_point_slider(self):
        from p3.plotting.plot_trajectories import plot_on_flat_plane, _add_point_slider
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_on_flat_plane(df)
        fig = _add_point_slider(fig)
        fig.show()

    def test_plot_on_flat_plane_w_point_slider_animated(self):
        from p3.plotting.plot_trajectories import plot_on_flat_plane, animate_trajectory_slider
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_on_flat_plane(df)
        fig = animate_trajectory_slider(df, 'x', 'y', fig=fig)
        fig.show()

    def test_plot_flight_state_w_point_slider(self):
        from p3.plotting.plot_trajectories import plot_flight_state
        from p3.plotting.utility import _add_point_slider
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_flight_state(df, df.columns.difference(['t']))
        fig = _add_point_slider(fig)
        fig.show()

    def test_trajectory_animation(self):
        from p3.plotting.plot_trajectories import animate_trajectory
        df = pd.read_csv(self.sample_trajectory_data)
        # df = df[df['t'] < 20]
        fig = animate_trajectory(df, 'x', 'y')
        fig.show()

    def test_chatgpt(self):
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots

        def add_slider_to_subplots(fig):
            """
            Add a slider to each subplot in a Plotly figure.
            
            :param fig: Existing Plotly figure with subplots
            :return: Updated Plotly figure with sliders for each subplot
            """
            
            # Determine number of rows and cols based on subplot traces
            rows, cols = 3, 4
            # Iterate over each subplot to add a scatter dot trace
            for r in range(1, rows+1):
                for c in range(1, cols+1):
                    subplot_data = list(fig.select_traces(row=r, col=c))
                    
                    if len(subplot_data) == 0:
                        continue

                    x_values = subplot_data[0]['x']
                    y_values = subplot_data[0]['y']

                    # Initially add scatter dot for the first data point
                    fig.add_trace(go.Scatter(x=[x_values[0]], y=[y_values[0]], mode='markers', name=f'Slider Dot ({r},{c})'), row=r, col=c)
            
            # Create steps for the slider
            steps = []
            for i in range(len(x_values)):  # Assumes all subplots have the same number of data points
                step_args = [{
                    f"x[{trace_idx}]": [x_values[i]],
                    f"y[{trace_idx}]": [y_values[i]]
                } for trace_idx, trace in enumerate(fig.data) if trace_idx % 2 == 1]  # Modify only the scatter dot traces, which are at odd indices

                step = {
                    'args': [step_args, [f"Scatter Dot {i}"]],
                    'label': str(i),
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
                    'prefix': 'Point Index:',
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

        # Sample usage for a 3x4 grid of subplots
        fig = make_subplots(rows=3, cols=4)
        for r in range(1, 4):
            for c in range(1, 5):
                fig.add_trace(go.Scatter(x=list(range(100)), y=list(range(100)), mode='lines', name=f'Base Line ({r},{c})'), row=r, col=c)

        fig_with_slider = add_slider_to_subplots(fig)
        fig_with_slider.show()


    def test_chatgpt_2(self):
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots

        # Sample usage for a 3x4 grid of subplots
        n_row, n_col = 3, 5
        fig = make_subplots(rows=n_row, cols=n_col)
        for r in range(1, n_row):
            for c in range(1,n_col):
                fig.add_trace(go.Scatter(x=list(range(100)), y=list(range(100)), mode='lines', name=f'Base Line ({r},{c})'), row=r, col=c)

        fig_with_slider = add_slider_to_subplots(fig)
        fig_with_slider.show()
