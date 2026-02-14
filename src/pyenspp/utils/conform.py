import xarray as xr

# ============================================================================
# Standardize observation and forecast data
# ============================================================================
OBS_PRECIP_NAMES = ["tp", "precip", "rainfall", "pr", "rain", "precipitation", "prec"]

def conform_obs(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize observational precipitation data.

    - Ensure Dataset has 'longitude', 'latitude', 'time' dimensions.
    - Automatically rename 'lon' -> 'longitude', 'lat' -> 'latitude'.
    - Detect precipitation variable from common names and rename to 'prec'.
    - Raise error if required dims or variable are missing.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray

    Returns
    -------
    xr.Dataset
        Standardized dataset with single 'prec' variable.
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "prec")

    rename_dict = {}
    if "lon" in ds.coords:
        rename_dict["lon"] = "longitude"
    if "lat" in ds.coords:
        rename_dict["lat"] = "latitude"
    if rename_dict:
        ds = ds.rename(rename_dict)
        print("Coordinates automatically renamed")

    required_dims = ["longitude", "latitude", "time"]
    missing_dims = [d for d in required_dims if d not in ds.dims]
    if missing_dims:
        raise ValueError(f"Missing required dims: {missing_dims}")

    # Detect precipitation variable
    found_var = next((v for v in ds.data_vars if v.lower() in OBS_PRECIP_NAMES), None)
    if found_var is None:
        raise ValueError(f"No recognizable precipitation variable found. Allowed: {OBS_PRECIP_NAMES}")

    ds = ds[[found_var]].rename({found_var: "prec"})
    print(f"Variable '{found_var}' renamed to 'prec' and retained only this variable")

    return ds


def conform_forecast(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize forecast precipitation data.

    - Ensure Dataset has 'longitude', 'latitude', 'time', 'step', 'number' dims.
    - Automatically rename 'lon'/'lat'.
    - Detect precipitation variable from common names and rename to 'prec'.
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "prec")

    rename_dict = {}
    if "lon" in ds.coords:
        rename_dict["lon"] = "longitude"
    if "lat" in ds.coords:
        rename_dict["lat"] = "latitude"
    if rename_dict:
        ds = ds.rename(rename_dict)
        print("Coordinates automatically renamed")

    required_dims = ["longitude", "latitude", "time", "step", "number"]
    missing_dims = [d for d in required_dims if d not in ds.dims]
    if missing_dims:
        raise ValueError(f"Missing required dims: {missing_dims}")

    found_var = next((v for v in ds.data_vars if v.lower() in OBS_PRECIP_NAMES), None)
    if found_var is None:
        raise ValueError(f"No recognizable precipitation variable found. Allowed: {OBS_PRECIP_NAMES}")

    ds = ds[[found_var]].rename({found_var: "prec"})
    print(f"Variable '{found_var}' renamed to 'prec' and retained only this variable")

    return ds


# ============================================================================
# Convert cumulative precipitation to step-wise increments
# ============================================================================
def cum_to_step(ds, var_name: str = 'prec'):
    """
    Convert cumulative precipitation to per-step increments.

    - step=0 is set to 0, subsequent steps = current - previous.
    - Supports xr.DataArray or xr.Dataset.
    - Clips negative values caused by floating point errors.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
    var_name : str
        Name of the precipitation variable.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Incremental precipitation.
    """
    if isinstance(ds, xr.Dataset):
        tp = ds[var_name]
    elif isinstance(ds, xr.DataArray):
        tp = ds
    else:
        raise TypeError("Input must be xr.DataArray or xr.Dataset")

    diff = tp.diff(dim='step', label='lower')
    zero = xr.zeros_like(tp.isel(step=0))
    incremental = xr.concat([zero, diff], dim='step')
    incremental = incremental.assign_coords(step=tp.step)
    incremental = incremental.clip(min=0)

    if isinstance(ds, xr.Dataset):
        ds_new = ds.copy()
        ds_new[var_name] = incremental
        return ds_new
    else:
        incremental.name = var_name
        return incremental