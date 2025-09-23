import numpy as np


def generate_pilots(
    n_subcarriers: int, seed: int = 0, dtype=np.complex64
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([1, -1, 1j, -1j], size=n_subcarriers).astype(dtype)
