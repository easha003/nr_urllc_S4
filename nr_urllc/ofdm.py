import numpy as np


def modulate(
    symbols: np.ndarray, n_subcarriers: int, cp_len: int, dtype=np.complex64
) -> np.ndarray:
    """OFDM modulation with cyclic prefix."""
    time_domain = np.fft.ifft(symbols.astype(dtype), n=n_subcarriers)
    cp = time_domain[-cp_len:]
    return np.concatenate([cp, time_domain]).astype(dtype)


def demodulate(
    signal: np.ndarray, n_subcarriers: int, cp_len: int, dtype=np.complex64
) -> np.ndarray:
    """OFDM demodulation removing cyclic prefix."""
    no_cp = signal[cp_len : cp_len + n_subcarriers]
    return np.fft.fft(no_cp.astype(dtype), n=n_subcarriers)
