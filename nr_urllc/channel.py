import numpy as np
from typing import Sequence, Tuple

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

def add_awgn(signal, snr_db, rng=None, dtype=np.complex64):
    if rng is None:
        rng = np.random.default_rng()
    return awgn(signal, snr_db, rng, dtype)

def flat_rayleigh(S: int, rng: np.random.Generator | None = None, dtype=np.complex64) -> np.ndarray:
    """One complex scalar per OFDM symbol (flat fading). Shape [S]. CN(0,1)."""
    if rng is None:
        rng = np.random.default_rng()
    h = (rng.normal(size=S) + 1j * rng.normal(size=S)) / np.sqrt(2.0)
    return h.astype(dtype)


def tdl_fir_from_profile(
    delays: Sequence[int],  # in samples
    powers_db: Sequence[float],  # power per tap in dB (will be normalized)
    rng: np.random.Generator | None = None,
    dtype=np.complex64,
) -> np.ndarray:
    """Build a random complex FIR h[n] with taps at given integer delays.
    Complex Gaussian taps with specified average power. Output length = max(delay)+1.
    """
    if rng is None:
        rng = np.random.default_rng()
    delays = np.asarray(delays, dtype=int)
    p_lin = 10.0 ** (np.asarray(powers_db, dtype=float) / 10.0)
    p_lin = p_lin / p_lin.sum()
    L = int(delays.max()) + 1
    h = np.zeros(L, dtype=np.complex64)
    for d, p in zip(delays, p_lin):
        amp = np.sqrt(p)
        tap = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
        h[d] = (amp * tap).astype(np.complex64)
    return h


def apply_fir_per_symbol(x_time: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve each OFDM symbol row with FIR h and crop to input length.
    x_time: [S, N] complex (with CP). Returns [S, N].
    """
    S, N = x_time.shape
    y = np.zeros_like(x_time, dtype=np.complex64)
    for s in range(S):
        full = np.convolve(x_time[s], h, mode="full").astype(np.complex64)
        y[s] = full[:N]
    return y
