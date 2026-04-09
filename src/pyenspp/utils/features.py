import xarray as xr
import numpy as np
import pandas as pd
# Temporal encoding
def doy_sin_cos(dates):
    """
    Convert dates to day of year sine and cosine values.
    
    Parameters
    ----------
    dates : np.array or single date
        Array of dates or a single date value.
    
    Returns
    -------
    sin : np.array or float
        Sine values of day of year.
    cos : np.array or float
        Cosine values of day of year.
    """
    # Convert single value to list
    if not isinstance(dates, (list, np.ndarray, pd.DatetimeIndex)):
        dates = [dates]
    
    doy = pd.DatetimeIndex(dates).dayofyear.values
    sin = np.sin(2 * np.pi * doy / 365.0)
    cos = np.cos(2 * np.pi * doy / 365.0)
    
    # Return scalar if input was scalar
    if len(sin) == 1:
        return sin[0], cos[0]
    return sin, cos

# Mean difference
def get_md(ens):
    '''
    MD (Mean Difference) refers to the mean difference between the ensemble members. It is a measure of the spread of the ensemble members.

    Parameters
    ----------
    ens: (number, ...) - number 是集合成员维度(放在第一个维度！), 后续可以是一个维度比如time, 也可以是多个维度比如time, step, lat, lon等
    Returns
    md: (...) number维度去除, 后续维度不变
    '''
    # 获取成员数量 M
    M = ens.shape[0]
    # 使用排序法优化计算（比两两相减更快，且节省内存）
    ens_sorted = np.sort(ens, axis=0)
    # 统计权重公式：MD = 2/(M^2) * sum((2i - M - 1) * x_i)
    i = np.arange(1, M + 1).reshape(M, *([1] * (ens.ndim - 1)))
    md = (2 / (M**2)) * np.sum((2 * i - M - 1) * ens_sorted, axis=0)
    return md
