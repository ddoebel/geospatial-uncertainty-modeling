import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import io
import base64
from dagster import asset, MetadataValue, Output
from ..core.uncertainty import calculate_ou_variance, reconstruct_spatial_variance
from ..core.visualization import plot_spatial_uncertainty

@asset
def forecast_variance(ou_parameters: xr.Dataset) -> xr.DataArray:
    """
    Compute lead-time-dependent variance for each EOF mode.
    """
    lead_times = np.arange(0, 25)  # 0 to 24 months
    n_modes = ou_parameters.sizes["mode"]
    
    variance_data = np.zeros((len(lead_times), n_modes))
    
    for i, k in enumerate(ou_parameters.mode):
        lam = ou_parameters["lambda"].sel(mode=k).item()
        sigma = ou_parameters["sigma"].sel(mode=k).item()
        variance_data[:, i] = calculate_ou_variance(lam, sigma, lead_times)
        
    return xr.DataArray(
        variance_data,
        dims=("lead_time", "mode"),
        coords={"lead_time": lead_times, "mode": ou_parameters.mode},
        name="forecast_variance"
    )

@asset
def spatial_uncertainty_map(context, forecast_variance: xr.DataArray, eof_modes: xr.DataArray) -> Output:
    """
    Reconstruct the spatial uncertainty map at a fixed lead time (e.g., 6 months).
    """
    lead_time = 6
    var_at_t = forecast_variance.sel(lead_time=lead_time).values
    
    # Reconstruct using patterns
    spatial_var_values = reconstruct_spatial_variance(var_at_t, eof_modes.values)
    
    # Create DataArray and unstack to lat/lon
    spatial_var = xr.DataArray(
        spatial_var_values,
        dims=("space",),
        coords={"space": eof_modes.space},
        name="forecast_uncertainty"
    ).unstack("space")
    
    # Create the plot and attach it to metadata
    fig = plot_spatial_uncertainty(spatial_var, lead_time=lead_time)
    
    # Save plot to buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)
    
    return Output(
        value=spatial_var.to_dataset(),
        metadata={
            "plot": MetadataValue.md(f"![plot](data:image/png;base64,{image_data})"),
            "lead_time_months": lead_time
        }
    )
