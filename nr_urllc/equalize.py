import numpy as np
from typing import Optional

def zf(Y: np.ndarray, H_est: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Zero-Forcing equalizer: Y_eq = Y / H_est."""
    return (Y / (H_est + eps)).astype(np.complex64)


def mmse(Y: np.ndarray, H_est: np.ndarray, noise_var: Optional[float]) -> np.ndarray:
    """Diagonal MMSE equalizer per RE. If noise_var is None, fallback to ZF.
    G = H* / (|H|^2 + sigma2)
    """
    if noise_var is None:
        return zf(Y, H_est)
    denom = (np.abs(H_est) ** 2) + float(noise_var)
    G = np.conjugate(H_est) / denom
    return (G * Y).astype(np.complex64)