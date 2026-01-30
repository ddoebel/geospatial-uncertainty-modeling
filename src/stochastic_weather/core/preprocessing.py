import numpy as np
import xarray as xr

def calculate_anomalies(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate seasonal anomalies by subtracting the monthly mean.
    
    Args:
        ds: Input dataset with a time dimension.
        
    Returns:
        Dataset of anomalies.
    """
    return ds.groupby("time.month") - ds.groupby("time.month").mean("time")

def apply_cosine_weighting(ds: xr.DataArray) -> xr.DataArray:
    """
    Apply latitude-dependent cosine weighting to account for grid cell area.
    
    Args:
        ds: DataArray with a latitude dimension.
        
    Returns:
        Weighted DataArray.
    """
    weights = np.sqrt(np.abs(np.cos(np.deg2rad(ds.latitude))))
    return ds * weights
