import numpy as np
import xarray as xr
from dagster import asset

@asset
def validated_data(raw_weather_data: xr.Dataset) -> xr.Dataset:
    """
    Perform structural and physical validation on the raw weather data.
    """
    ds = raw_weather_data

    # --- Structural checks ---
    if "time" not in ds.dims:
        raise ValueError("Missing time dimension")
        
    time_index = ds.time.to_index()
    if not time_index.is_monotonic_increasing:
        raise ValueError("Time dimension is not monotonic")
    if time_index.duplicated().any():
        raise ValueError("Duplicate timestamps found")

    # --- Physical checks ---
    for var in ds.data_vars:
        data = ds[var]
        
        # Check for non-finite values in core stats
        if not np.isfinite(data.mean()) or not np.isfinite(data.std()):
            raise ValueError(f"{var}: non-finite statistics detected")
            
        # Specific physical range check for evaporation (pev)
        if var == "pev":
            # Daily accumulated evaporation in meters water equivalent
            max_val = float(data.max())
            if max_val > 0.02:  # 20 mm/day is extreme for monthly average
                raise ValueError(f"{var}: implausibly high evaporation ({max_val} m)")

    return ds
