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
        from p3.plotting.plot_2d import animate_trajectory
        df = pd.read_csv(self.sample_trajectory_data)
        # df = df[df['t'] < 20]
        fig = animate_trajectory(df, 'x', 'y')
        fig.show()
