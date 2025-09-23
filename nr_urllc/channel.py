import numpy as np


def awgn(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator, dtype=np.complex64
) -> np.ndarray:
    """Add AWGN to a signal."""
    snr_linear = 10 ** (snr_db / 10)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / snr_linear
    noise = rng.normal(
        scale=np.sqrt(noise_power / 2), size=signal.shape
    ) + 1j * rng.normal(scale=np.sqrt(noise_power / 2), size=signal.shape)
    return (signal + noise).astype(dtype)
