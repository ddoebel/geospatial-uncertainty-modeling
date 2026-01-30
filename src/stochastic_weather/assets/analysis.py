import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import io
import base64
from dagster import asset, multi_asset, AssetOut, MetadataValue, Output
from ..core.preprocessing import calculate_anomalies, apply_cosine_weighting
from ..core.eof import compute_pca
from ..core.ou_logic import estimate_integral_scale, get_ou_parameters
from ..core.visualization import plot_eof_modes

@multi_asset(
    outs={
        "eof_coefficients": AssetOut(description="Time series of EOF coefficients (Principal Components)"),
        "eof_modes": AssetOut(description="Spatial patterns of EOF modes"),
    }
)
def eof_analysis(context, validated_data: xr.Dataset):
    """
    Perform EOF decomposition on the weighted anomalies of the 'pev' variable.
    """
    # Select variable and preprocess
    da = validated_data["pev"]
    anomalies = calculate_anomalies(da)
    weighted_anomalies = apply_cosine_weighting(anomalies)
    
    # Flatten spatial dimensions for PCA
    X = weighted_anomalies.stack(space=("latitude", "longitude"))
    # Dropping NaNs if any (e.g. over land if it was ocean only, or vice versa)
    X_clean = X.dropna("space")
    
    n_modes = 4
    A, Phi = compute_pca(X_clean.values, n_modes)
    
    # Create xarray objects for the outputs
    coeffs = xr.DataArray(
        A, 
        dims=("time", "mode"),
        coords={"time": X_clean.time, "mode": np.arange(n_modes)},
        name="eof_coefficients"
    )
    
    modes = xr.DataArray(
        Phi, 
        dims=("mode", "space"),
        coords={"mode": np.arange(n_modes), "space": X_clean.space},
        name="eof_modes"
    )

    # Visualization
    fig = plot_eof_modes(modes)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)

    return Output(coeffs, output_name="eof_coefficients"), Output(
        value=modes,
        metadata={
            "eof_modes_plot": MetadataValue.md(f"![plot](data:image/png;base64,{image_data})")
        },
        output_name="eof_modes"
    )

@asset
def ou_parameters(eof_coefficients: xr.DataArray) -> xr.Dataset:
    """
    Fit Ornstein-Uhlenbeck processes to each EOF coefficient time series.
    """
    lambdas = []
    sigmas = []
    
    for k in eof_coefficients.mode:
        ts = eof_coefficients.sel(mode=k).values
        tau = estimate_integral_scale(ts)
        variance = np.var(ts)
        
        params = get_ou_parameters(tau, variance)
        lambdas.append(params["lambda"])
        sigmas.append(params["sigma"])
        
    return xr.Dataset(
        data_vars={
            "lambda": (["mode"], lambdas),
            "sigma": (["mode"], sigmas)
        },
        coords={"mode": eof_coefficients.mode},
        attrs={
            "units": "1/month",
            "description": "Fitted parameters of the Ornstein-Uhlenbeck process"
        }
    )
