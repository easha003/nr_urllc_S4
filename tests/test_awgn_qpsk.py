import numpy as np
from nr_urllc import simulate


def qpsk_theory_ber(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))


def test_awgn_qpsk_ber():
    cfg = {"seed": 0, "n_bits": 1_000_000, "snr_db": 3}  # moderate SNR
    result = simulate.run(cfg)
    sim_ber = result["ber"]
    theory_ber = qpsk_theory_ber(cfg["snr_db"])
    # Use relative difference
    assert abs(sim_ber - theory_ber) / theory_ber <= 0.5
