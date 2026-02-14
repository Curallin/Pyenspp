import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

# ============================================================================
# Compute area-weighted mean precipitation
# ============================================================================
def area_mean_precip(
    data: xr.DataArray | xr.Dataset,
    region: gpd.GeoDataFrame,
    var_name: str = "prec",
    use_cos_lat_weight: bool = True,
    skipna: bool = True,
) -> xr.DataArray:
    """
    Compute area-weighted mean precipitation over a given region.

    - Supports any xarray object with 'latitude' and 'longitude' dimensions.
    - Preserves all other dimensions (e.g., time, step, member).
    - Accounts for grid area and Earth curvature if use_cos_lat_weight=True.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Precipitation data. If Dataset, var_name is used.
    region : geopandas.GeoDataFrame
        Target region geometry.
    var_name : str
        Variable name for Dataset input.
    use_cos_lat_weight : bool
        Apply cos(latitude) weighting (recommended).
    skipna : bool
        Ignore NaNs in the weighted sum (recommended).

    Returns
    -------
    xr.DataArray
        Area-averaged precipitation, dims excluding latitude/longitude.
    """
    # Convert Dataset to DataArray if needed
    if isinstance(data, xr.Dataset):
        if var_name not in data:
            raise ValueError(f"Variable '{var_name}' not found. Available: {list(data.data_vars)}")
        da = data[var_name]
    elif isinstance(data, xr.DataArray):
        da = data
        var_name = da.name or var_name
    else:
        raise TypeError("Input must be xarray DataArray or Dataset")
    
    # Ensure lat/lon are last dims
    da = da.transpose(
        *[d for d in da.dims if d not in ('latitude', 'longitude')],
        'latitude',
        'longitude'
    )

    if not {'latitude', 'longitude'} <= set(da.dims):
        raise ValueError("Data must have 'latitude' and 'longitude' dims")
    
    # Ensure region is in EPSG:4326
    if region.crs.to_epsg() != 4326:
        region = region.to_crs(epsg=4326)

    lat = da['latitude']
    lon = da['longitude']
    is_1d = lat.ndim == 1 and lon.ndim == 1
    if is_1d:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d = lon.values
        lat2d = lat.values

    # Grid resolution
    dlon = float(np.nanmean(np.diff(lon.values))) if is_1d else float(lon2d[0,1] - lon2d[0,0])
    dlat = float(np.nanmean(np.diff(lat.values))) if is_1d else float(lat2d[1,0] - lat2d[0,0])

    geom = unary_union(region.geometry)
    bounds = geom.bounds

    # Compute coverage fraction per grid cell
    coverage = np.zeros_like(lat2d, dtype=np.float32)
    mask_bbox = (
        (lon2d - dlon/2 <= bounds[2]) &
        (lon2d + dlon/2 >= bounds[0]) &
        (lat2d - dlat/2 <= bounds[3]) &
        (lat2d + dlat/2 >= bounds[1])
    )
    i_idx, j_idx = np.where(mask_bbox)
    for i, j in zip(i_idx, j_idx):
        cell = box(
            lon2d[i,j] - dlon/2, lat2d[i,j] - dlat/2,
            lon2d[i,j] + dlon/2, lat2d[i,j] + dlat/2
        )
        if geom.intersects(cell):
            inter_area = geom.intersection(cell).area
            cell_area = cell.area
            coverage[i,j] = inter_area / cell_area if cell_area > 0 else 0

    # Apply latitude weighting
    if use_cos_lat_weight:
        weights = coverage * np.cos(np.deg2rad(lat2d))
    else:
        weights = coverage

    total_w = weights.sum()
    if total_w <= 0:
        raise ValueError("No overlapping area between region and grid")
    weights_norm = weights / total_w

    weights_da = xr.DataArray(
        weights_norm,
        dims=da.dims[-2:],
        coords={'latitude': da.coords['latitude'], 'longitude': da.coords['longitude']}
    )

    # Weighted mean
    weighted = da * weights_da
    mean = weighted.sum(dim=['latitude', 'longitude'], skipna=skipna)
    mean.name = var_name
    mean.attrs.update({
        'long_name': f"Area-weighted mean {var_name}",
        'units': da.attrs.get('units', 'unknown'),
        'method': 'cos_lat_weighted' if use_cos_lat_weight else 'coverage_only',
        'region_crs': region.crs.to_string() if region.crs else 'unknown',
        'skipna': str(skipna)
    })

    return mean


def build_area_weights(
    lat: xr.DataArray,
    lon: xr.DataArray,
    region: gpd.GeoDataFrame,
    use_cos_lat_weight: bool = True,
):
    """
    Build area weights for a given region (computed once).

    Returns a DataArray of weights (latitude x longitude).
    """
    if region.crs.to_epsg() != 4326:
        region = region.to_crs(epsg=4326)

    geom = unary_union(region.geometry)
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)

    dlon = float(np.nanmean(np.diff(lon.values)))
    dlat = float(np.nanmean(np.diff(lat.values)))

    coverage = np.zeros_like(lat2d, dtype=np.float32)
    bounds = geom.bounds
    mask_bbox = (
        (lon2d - dlon/2 <= bounds[2]) &
        (lon2d + dlon/2 >= bounds[0]) &
        (lat2d - dlat/2 <= bounds[3]) &
        (lat2d + dlat/2 >= bounds[1])
    )

    idx_i, idx_j = np.where(mask_bbox)
    for i, j in zip(idx_i, idx_j):
        cell = box(
            lon2d[i, j] - dlon/2, lat2d[i, j] - dlat/2,
            lon2d[i, j] + dlon/2, lat2d[i, j] + dlat/2
        )
        if geom.intersects(cell):
            coverage[i, j] = geom.intersection(cell).area / cell.area

    weights = coverage * np.cos(np.deg2rad(lat2d)) if use_cos_lat_weight else coverage
    if weights.sum() <= 0:
        raise ValueError("No overlapping area between region and grid")
    weights /= weights.sum()

    return xr.DataArray(
        weights,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="area_weight",
        attrs={"method": "coverage_coslat"},
    )


def area_mean_with_weights(
    da: xr.DataArray,
    weights: xr.DataArray,
    skipna: bool = True,
):
    """
    Compute area-weighted mean using precomputed weights.
    """
    weighted = da * weights
    return weighted.sum(dim=("latitude", "longitude"), skipna=skipna)