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
        from p3.plotting.plot_trajectories import plot_on_flat_plane
        from p3.plotting.utility import _add_point_slider
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
