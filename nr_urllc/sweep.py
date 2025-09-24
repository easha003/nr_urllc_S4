# nr_urllc/sweep.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Callable, Optional
import math
import numpy as np

from . import utils
from . import ofdm

# ---------- dataclasses ----------

@dataclass
class SweepPoint:
    snr_db: float
    n_bits: int
    n_errs: int
    ber: float

@dataclass
class SweepResult:
    success: bool
    meta: Dict[str, Any]
    points: List[SweepPoint]

    @property
    def ber_curve(self) -> Dict[str, float]:
        return {f"{p.snr_db:.1f}": p.ber for p in self.points}

    @property
    def n_bits_curve(self) -> Dict[str, int]:
        return {f"{p.snr_db:.1f}": p.n_bits for p in self.points}

    @property
    def n_errs_curve(self) -> Dict[str, int]:
        return {f"{p.snr_db:.1f}": p.n_errs for p in self.points}


# ---------- theory helpers (array-safe) ----------

def ber_mqam_theory(ebn0_db: float | np.ndarray, M: int) -> np.ndarray:
    if M == 4:
        return utils.ber_qpsk_theory(ebn0_db)
    g = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    k = int(math.log2(M))
    return (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * utils.qfunc(np.sqrt(3.0 * k / (M - 1) * g))


# ---------- planning bits per SNR ----------

def _align_up(n: int, align: int) -> int:
    if align <= 1:
        return int(n)
    return int(((n + align - 1) // align) * align)

def plan_bits_per_snr(
    snr_db_list: List[float],
    M: int,
    target_errs: int = 100,
    min_bits: int = 20_000,
    max_bits: int = 2_000_000,
    floor_ber: float = 1e-6,
    align_to: Optional[int] = None,
) -> Dict[float, int]:
    """
    Choose n_bits per SNR to expect ~target_errs errors, capped into [min_bits, max_bits].
    If align_to is provided (e.g., k for SC; K*k for OFDM), n_bits is rounded up to a multiple of it.
    """
    plan: Dict[float, int] = {}
    # Treat 'snr' as Eb/N0 for SC; Es/N0->Eb/N0 consistency is handled by the measure fn.
    for snr in snr_db_list:
        ber_est = float(ber_mqam_theory(snr, M))
        ber_est = max(ber_est, floor_ber)  # avoid exploding counts at very high SNR
        n = math.ceil(target_errs / ber_est)
        n = max(min_bits, min(n, max_bits))
        if align_to is not None:
            n = _align_up(n, align_to)
        plan[float(snr)] = int(n)
    return plan


# ---------- measurement backends ----------

def measure_sc_awgn_ber(
    ebn0_db: float,
    M: int,
    n_bits: int,
    rng: np.random.Generator,
    code_rate: float = 1.0,
) -> tuple[int, int, float]:
    """Single-carrier BER under AWGN using utils.mod/demod and sigma from Eb/N0."""
    k = int(math.log2(M))
    tx_bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    tx_syms = utils.mod(tx_bits, M)
    sigma_RI = utils.ebn0_db_to_sigma_sc(ebn0_db, M=M, code_rate=code_rate, Es_sym=1.0)
    n = rng.normal(0.0, sigma_RI, tx_syms.shape) + 1j * rng.normal(0.0, sigma_RI, tx_syms.shape)
    rx_syms = tx_syms + n
    rx_bits = utils.demod(rx_syms, M)
    n_eval = rx_bits.size  # should equal n_bits, but guard anyway
    n_errs = int(np.count_nonzero(rx_bits != tx_bits[:n_eval]))
    ber = n_errs / float(n_eval)
    return n_errs, n_eval, ber


def measure_ofdm_awgn_ber(
    snr_db: float,
    M: int,
    n_bits_target: int,
    rng: np.random.Generator,
    nfft: int,
    cp: float,
    n_subcarriers: int,
    minislot_symbols: int,
    code_rate: float = 1.0,
    ifft_norm: str = "numpy",
) -> tuple[int, int, float]:
    """
    OFDM BER: builds enough OFDM symbols to meet/exceed n_bits_target.
    Returns (n_errs, n_bits_eval, ber).
    """
    k = int(math.log2(M))
    bits_per_ofdm_symbol = n_subcarriers * k
    n_syms = math.ceil(n_bits_target / bits_per_ofdm_symbol)
    # optional: force to a multiple of minislot size
    if minislot_symbols > 1:
        n_syms = _align_up(n_syms, minislot_symbols)

    n_bits_eval = n_syms * bits_per_ofdm_symbol

    # Make used-tone matrix [S, K]
    tx_bits = rng.integers(0, 2, size=n_bits_eval, dtype=np.int8)
    syms = utils.mod(tx_bits, M).reshape(n_syms, n_subcarriers)

    x = ofdm.tx(syms, nfft=nfft, cp=cp, n_subcarriers=n_subcarriers)  # time-domain with CP
    sigma_RI = utils.ebn0_db_to_sigma_ofdm_time(
        snr_db, M=M, code_rate=code_rate, nfft=nfft, ifft_norm=ifft_norm, Es_sub=1.0
    )
    noise = rng.normal(0.0, sigma_RI, x.shape) + 1j * rng.normal(0.0, sigma_RI, x.shape)
    y = x + noise
    Y_used = ofdm.rx(y, nfft=nfft, cp=cp, n_subcarriers=n_subcarriers)  # [S, K] used tones
    rx_bits = utils.demod(Y_used.reshape(-1), M)

    n_errs = int(np.count_nonzero(rx_bits != tx_bits))
    ber = n_errs / float(n_bits_eval)
    return n_errs, n_bits_eval, ber


# ---------- top-level sweep helpers ----------

def autoramp_sc_qpsk_sweep(
    ebn0_db_list: List[float],
    seed: int,
    M: int = 4,
    target_errs: int = 100,
    min_bits: int = 20_000,
    max_bits: int = 2_000_000,
) -> SweepResult:
    rng = utils.get_rng(seed)
    k = int(math.log2(M))
    plan = plan_bits_per_snr(
        ebn0_db_list, M=M, target_errs=target_errs, min_bits=min_bits, max_bits=max_bits, align_to=k
    )
    points: List[SweepPoint] = []
    for snr in ebn0_db_list:
        n_bits = plan[snr]
        n_errs, n_eval, ber = measure_sc_awgn_ber(snr, M, n_bits, rng)
        points.append(SweepPoint(snr_db=float(snr), n_bits=n_eval, n_errs=n_errs, ber=ber))
    meta = {"seed": seed, "M": M, "mode": "sc_awgn", "target_errs": target_errs}
    return SweepResult(success=True, meta=meta, points=points)


def autoramp_ofdm_qpsk_sweep(
    snr_db_list: List[float],
    seed: int,
    *,
    M: int = 4,
    nfft: int,
    cp: float,
    n_subcarriers: int,
    minislot_symbols: int,
    target_errs: int = 100,
    min_bits: int = 20_000,
    max_bits: int = 2_000_000,
) -> SweepResult:
    rng = utils.get_rng(seed)
    k = int(math.log2(M))
    align_to = n_subcarriers * k  # make bitcount align with whole OFDM symbols
    plan = plan_bits_per_snr(
        snr_db_list, M=M, target_errs=target_errs, min_bits=min_bits, max_bits=max_bits, align_to=align_to
    )
    points: List[SweepPoint] = []
    for snr in snr_db_list:
        n_bits = plan[snr]
        n_errs, n_eval, ber = measure_ofdm_awgn_ber(
            snr, M, n_bits, rng, nfft=nfft, cp=cp, n_subcarriers=n_subcarriers, minislot_symbols=minislot_symbols
        )
        points.append(SweepPoint(snr_db=float(snr), n_bits=n_eval, n_errs=n_errs, ber=ber))
    meta = {
        "seed": seed, "M": M, "mode": "ofdm_awgn",
        "nfft": nfft, "cp": cp, "n_subcarriers": n_subcarriers, "minislot_symbols": minislot_symbols,
        "target_errs": target_errs
    }
    return SweepResult(success=True, meta=meta, points=points)
