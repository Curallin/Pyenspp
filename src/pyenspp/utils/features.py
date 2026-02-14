import xarray as xr
import numpy as np
import pandas as pd
# Temporal encoding
def doy_sin_cos(dates):
    doy = pd.DatetimeIndex(dates).dayofyear.values
    sin = np.sin(2 * np.pi * doy / 365.0)
    cos = np.cos(2 * np.pi * doy / 365.0)
    return sin, cos