import numpy as np


def crc_check(bits: np.ndarray) -> bool:
    """Stub for CRC check (accepts all packets for now)."""
    return True


def encode(bits: np.ndarray) -> np.ndarray:
    """Stub for FEC encoder (pass-through)."""
    return bits


def decode(bits: np.ndarray) -> np.ndarray:
    """Stub for FEC decoder (pass-through)."""
    return bits
