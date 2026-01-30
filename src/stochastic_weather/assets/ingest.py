import xarray as xr
from dagster import asset
import os

@asset
def raw_weather_data() -> xr.Dataset:
    """
    Load raw weather data from NetCDF without modification.
    """
    # Using absolute path logic or relative to project root
    # For now, keeping it similar to original but more robust
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = os.path.join(project_root, "data/ERA5_LowRes_Monthly_evap.nc")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    ds = xr.open_dataset(data_path)
    ds.load()
    return ds
