import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

def plot_spatial_uncertainty(uncertainty_ds: xr.Dataset, lead_time: int = 6, title: str = None):
    """
    Plot the spatial uncertainty map.
    
    Args:
        uncertainty_ds: Dataset containing 'forecast_uncertainty' variable.
        lead_time: Lead time in months (for title).
        title: Optional title for the plot.
    """
    if "forecast_uncertainty" not in uncertainty_ds:
        # If it's a DataArray, convert to Dataset or handle it
        if isinstance(uncertainty_ds, xr.DataArray):
            da = uncertainty_ds
        else:
            raise ValueError("Dataset must contain 'forecast_uncertainty'")
    else:
        da = uncertainty_ds["forecast_uncertainty"]

    plt.figure(figsize=(10, 6))
    da.plot(cmap="viridis")
    
    if title is None:
        title = f"Reconstructed Forecast Uncertainty (Lead Time: {lead_time} months)"
    
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    return plt.gcf()

def plot_eof_modes(eof_modes: xr.DataArray, n_modes: int = 4):
    """
    Plot the spatial patterns of the EOF modes.
    """
    n_modes = min(n_modes, len(eof_modes.mode))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in range(n_modes):
        mode_data = eof_modes.sel(mode=i).unstack("space")
        mode_data.plot(ax=axes[i], cmap="RdBu_r")
        axes[i].set_title(f"EOF Mode {i+1}")
        
    plt.tight_layout()
    return fig
