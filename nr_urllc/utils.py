import numpy as np


def get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def complex_exp(theta: np.ndarray, dtype=np.complex64) -> np.ndarray:
    return np.exp(1j * theta).astype(dtype)
