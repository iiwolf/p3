import numpy as np
import pandas as pd

def resample_df(df, time_col, dt, n_samples=None):
    """
    Interpolates a DataFrame using pandas resample based on a desired time step 
    and then optionally downsamples to a specified number of samples.

    Parameters:
    - df: pandas DataFrame containing the time series data
    - time_col: str, name of the column containing time values (assumed to be in seconds)
    - dt: float, desired time step for interpolation (also in seconds)
    - n_samples: int, optional, number of samples to be taken from the interpolated data

    Returns:
    - new_df: pandas DataFrame with interpolated values
    """

    # Convert the time column to Timedelta and set as index
    df = df.set_index(pd.to_timedelta(df[time_col], unit='s'))

    # Drop the original time column
    df = df.drop(columns=[time_col])

    # Resample and interpolate
    new_df = df.resample(f'{int(dt*1e9)}N').first().interpolate()

    # If n_samples is provided, downsample the interpolated data
    if n_samples is not None:
        idx = np.linspace(0, len(new_df) - 1, n_samples, dtype=int)
        new_df = new_df.iloc[idx]

    # Reset the index to have time in seconds (or another unit if you prefer)
    new_df[time_col] = new_df.index.total_seconds()
    new_df = new_df.reset_index(drop=True)

    return new_df
