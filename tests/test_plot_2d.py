import pandas as pd
from unit_test import UnitTest
class TestPlotTrajectory(UnitTest):

    def test_plot_flight_state(self):
        from p3.plotting.plot_2d import plot_flight_state
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_flight_state(df, df.columns.difference(['t']))
        fig.show()

    def test_plot_on_flat_plane(self):
        from p3.plotting.plot_2d import plot_on_flat_plane
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_on_flat_plane(df)
        fig.show()

    def test_plot_flight_state_multi_y(self):
        from p3.plotting.plot_2d import plot_flight_state_multi_y
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_flight_state_multi_y(df)
        fig.show()

    def test_trajectory_animation(self):
        import plotly.express as px

        df = pd.read_csv(self.sample_trajectory_data).iloc[::10]

        # Create animation
        fig = px.scatter(
            df,
            x='x',
            y='y',
            animation_frame=df.index,
            range_x=[min(df['x']) - 1, max(df['x']) + 1], 
            range_y=[min(df['y']) - 1, max(df['y']) + 1]
        )

        # Show figure
        fig.show()
