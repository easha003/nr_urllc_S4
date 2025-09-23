import numpy as np
from . import channel, metrics, fec, utils


def run(cfg: dict) -> dict:
    """Run a minimal QPSK+AWGN simulation."""
    rng = utils.get_rng(cfg["seed"])
    n_bits = cfg.get("n_bits", 1000)
    snr_db = cfg.get("snr_db", 10)

    # QPSK mapping
    bits = rng.integers(0, 2, size=n_bits)
    symbols = (1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2])
    symbols = symbols.astype(np.complex64)

    # Channel
    rx_symbols = channel.awgn(symbols, snr_db, rng)

    # Hard decision QPSK demod
    rx_bits = np.zeros_like(bits)
    rx_bits[0::2] = (rx_symbols.real < 0).astype(int)
    rx_bits[1::2] = (rx_symbols.imag < 0).astype(int)

    ber = metrics.ber_count(bits, rx_bits)

    return {
        "success": True,
        "reps_used": 1,
        "sinr_db": snr_db,
        "latency_ms": 0.0,
        "crc_ok": fec.crc_check(rx_bits),
        "meta": {"seed": cfg["seed"], "snr_db": snr_db},
        "ber": ber,
    }
