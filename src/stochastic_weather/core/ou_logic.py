import numpy as np
from statsmodels.tsa.stattools import acf
from typing import Dict

def estimate_integral_scale(data: np.ndarray, dx: float = 1.0) -> float:
    """
    Estimate the integral time scale of a time series.
    
    Args:
        data: 1D array of time series data.
        dx: Time step between observations.
        
    Returns:
        The integral time scale (tau).
    """
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return 0.0
        
    # Subtract mean to ensure we work with anomalies
    data = data - np.mean(data)
    
    # Calculate Autocorrelation Function
    acf_vals = acf(data, nlags=len(data)//2, fft=True)
    
    # Find the first zero-crossing as a cutoff for integration
    cutoff = np.where(acf_vals < 0)[0]
    cutoff = cutoff[0] if cutoff.size > 0 else len(acf_vals)
    
    # Integrate using trapezoidal rule
    tau = np.trapezoid(acf_vals[:cutoff], dx=dx)
    return tau

def get_ou_parameters(tau: float, variance: float) -> Dict[str, float]:
    """
    Derive Ornstein-Uhlenbeck parameters from integral scale and variance.
    
    Args:
        tau: Integral time scale.
        variance: Variance of the process.
        
    Returns:
        Dictionary containing 'lambda' (decay rate) and 'sigma' (noise strength).
    """
    if tau <= 0:
        return {"lambda": np.nan, "sigma": np.nan}
        
    lam = 1.0 / tau
    sigma = np.sqrt(2 * lam * variance)
    
    return {
        "lambda": lam,
        "sigma": sigma
    }

def validate_ou_fit(lam: float, sigma: float, acf_empirical: np.ndarray, dt: float = 1.0) -> float:
    """
    Validate the OU fit by comparing empirical vs theoretical autocorrelation.
    
    Args:
        lam: Decay rate.
        sigma: Noise strength.
        acf_empirical: Empirical ACF values.
        dt: Time step.
        
    Returns:
        Mean Squared Error (MSE) between empirical and theoretical ACF.
    """
    lags = np.arange(len(acf_empirical))
    acf_theoretical = np.exp(-lam * lags * dt)
    
    mse = np.mean((acf_empirical - acf_theoretical) ** 2)
    return mse
