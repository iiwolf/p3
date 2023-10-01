import pandas as pd
from unit_test import UnitTest
class TestPlotTrajectory(UnitTest):

    def test_plot_flight_dash(self):
        from p3.plotting.plot_flight_dash import plot_flight_dash
        df = pd.read_csv(self.sample_trajectory_data)
        fig = plot_flight_dash(df, df.columns.difference(['t']))
        fig.show()

