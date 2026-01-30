import numpy as np

def calculate_ou_variance(lam: float, sigma: float, t: np.ndarray) -> np.ndarray:
    """
    Calculate the variance of an OU process at given lead times.
    
    V(t) = (sigma^2 / 2*lambda) * (1 - exp(-2*lambda*t))
    
    Args:
        lam: Decay rate.
        sigma: Noise strength.
        t: Lead times.
        
    Returns:
        Variance at lead times.
    """
    if lam <= 0:
        return np.full_like(t, np.nan)
    return (sigma ** 2 / (2 * lam)) * (1 - np.exp(-2 * lam * t))

def reconstruct_spatial_variance(mode_variances: np.ndarray, eof_patterns: np.ndarray) -> np.ndarray:
    """
    Project EOF mode variances back to spatial grid.
    
    Total Variance = sum_k (Var_k * Phi_k^2)
    
    Args:
        mode_variances: Variances per mode (n_modes,).
        eof_patterns: EOF spatial patterns (n_modes, n_space).
        
    Returns:
        Spatial variance (n_space,).
    """
    # mode_variances: (n_modes, 1) to broadcast over eof_patterns: (n_modes, n_space)
    return np.sum(mode_variances[:, np.newaxis] * (eof_patterns ** 2), axis=0)
