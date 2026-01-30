import numpy as np
from typing import Tuple

def compute_pca(data: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis (EOF decomposition) using SVD.
    
    Args:
        data: 2D array (time, space).
        n_modes: Number of modes to retain.
        
    Returns:
        Tuple of (Principal Components, EOF Patterns).
    """
    # U: (time, time), S: (min(time, space),), Vh: (space, space)
    U, S, Vh = np.linalg.svd(data, full_matrices=False)
    
    # Coefficients (Principal Components)
    A = U[:, :n_modes] * S[:n_modes]
    
    # Modes (EOF Patterns)
    Phi = Vh[:n_modes, :]
    
    return A, Phi
