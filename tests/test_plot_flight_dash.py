import pandas as pd
from unit_test import UnitTest
class TestPlotTrajectory(UnitTest):

    def test_plot_flight_dash(self):
        from p3.plotting.plot_flight_dash import plot_flight_dash
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_flight_dash(df)
        fig.show()

    def test_plot_flight_dash_slider(self):
        from p3.plotting.plot_flight_dash import plot_flight_dash
        from p3.plotting.utility import _add_point_slider
        df = pd.read_csv(self.sample_trajectory_data)
        # fig = plot_flight_dash(df, vars_to_plot=['x', 'y', 'vx', 'vy'])
        fig = plot_flight_dash(df)
        fig = _add_point_slider(fig)
        fig.show()
