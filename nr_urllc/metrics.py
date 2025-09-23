import numpy as np


def ber_count(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    return np.mean(tx_bits != rx_bits)


def sinr_db(signal_power: float, noise_power: float) -> float:
    return 10 * np.log10(signal_power / noise_power)
