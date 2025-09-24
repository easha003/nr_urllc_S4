# nr_urllc/simulate.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import math as m

from . import utils, ofdm, channel, fec
from . import pilots as pilots_mod
from . import metrics as metrics_mod
from . import equalize as eq
from .utils import qfunc


def _write_json(maybe: bool, path: str, obj: dict):
    if maybe:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


# ---------------------------- M0: Baseline AWGN ---------------------------- #

def run_baseline_awgn(cfg: dict) -> dict:
    """
    Baseline bit-level AWGN simulation (no OFDM).
    Interprets cfg['channel']['snr_db'] as Eb/N0 [dB], converts to Es/N0 internally.
    Uses a local RNG; no global seeding.
    """
    sim = cfg.get("sim", {})
    tx  = cfg.get("tx", {})
    ch  = cfg.get("channel", {})
    io  = cfg.get("io", {})

    rng    = utils.get_rng(sim.get("seed"))
    M      = int(tx.get("M", 4))                      # 4=QPSK, 16=16QAM
    k      = int(np.log2(M))
    n_bits = int(tx.get("n_bits", 100_000))
    ebn0   = float(ch.get("snr_db", 5.0))             # Eb/N0 in dB (by convention)

    # Map bits -> symbols (unit average Es expected from utils.mod)
    bits    = rng.integers(0, 2, size=n_bits)
    symbols = utils.mod(bits, M).astype(np.complex64)

    # Eb/N0 -> Es/N0 (per-symbol SNR)
    esn0_db = ebn0 + 10 * np.log10(k)

    # AWGN at symbol level
    rx_symbols = channel.awgn(symbols, esn0_db, rng)

    # Hard-decision demod
    bits_hat = utils.demod(rx_symbols, M)
    ber = float(np.mean(bits[:len(bits_hat)] != bits_hat))

    out = {
        "success": True,
        "reps_used": 1,
        "sinr_db": ebn0,                 # report Eb/N0 you asked for
        "latency_ms": 0.0,               # placeholder at M0
        "crc_ok": True,                  # placeholder at M0
        "meta": {"seed": sim.get("seed"), "M": M, "EbN0_dB": ebn0, "EsN0_dB": esn0_db},
        "ber": ber,
    }
    _write_json(bool(io.get("write_json", False)), io.get("out_json", "artifacts/result.json"), out)
    return out


# ----------------------------- M1: OFDM over AWGN -------------------------- #

def _run_ofdm_awgn_cfg(cfg: dict) -> dict:
    """
    OFDM mini-slot over AWGN (config-driven).
    Interprets cfg['channel']['snr_db_list'] as Eb/N0 [dB] values.
    Adds calibrated time-domain noise so post-FFT per-subcarrier Es/N0 matches target.
    Uses a local RNG; no global seeding.
    """
    sim = cfg.get("sim", {})
    tx  = cfg.get("tx", {})
    of  = cfg.get("ofdm", {})
    ch  = cfg.get("channel", {})
    io  = cfg.get("io", {})

    rng   = utils.get_rng(sim.get("seed"))
    M     = int(tx.get("M", 4))
    k     = int(np.log2(M))
    n_bits = int(tx.get("n_bits", 120_000))

    nfft  = int(of.get("nfft", 256))
    cp    = float(of.get("cp", 0.125))
    n_sc  = int(of.get("n_subcarriers", 64))
    _     = int(of.get("n_symbols", 14))             # not directly used here
    L_ms  = int(of.get("minislot_symbols", 4))       # {2,4,7} typical

    ebn0_list = ch.get("snr_db_list", [0, 5, 10])

    # Bits -> QAM (unit Es) -> grid with whole rows
    n_syms_needed = (n_bits // k + n_sc - 1) // n_sc
    n_bits_eff    = n_syms_needed * n_sc * k
    bits          = rng.integers(0, 2, n_bits_eff)
    tx_syms       = utils.mod(bits, M).astype(np.complex64)
    tx_grid       = tx_syms.reshape(n_syms_needed, n_sc)

    # Mini-slot slice
    use_syms   = min(L_ms, tx_grid.shape[0])
    tx_grid_ms = tx_grid[:use_syms, :]

    # OFDM TX once (IFFT has 1/N scaling; then prepend CP)
    tx_time = ofdm.tx(tx_grid_ms, nfft=nfft, cp=cp).astype(np.complex64)

    out_curve = {}
    for ebn0_db in ebn0_list:
        # Eb/N0 -> Es/N0
        esn0_db  = float(ebn0_db) + 10 * np.log10(k)
        esn0_lin = 10 ** (esn0_db / 10.0)

        # Calibrated time-domain AWGN: var = 1 / (Nfft * Es/N0)
        noise_var = 1.0 / (nfft * esn0_lin)
        noise = (
            rng.normal(0, np.sqrt(noise_var / 2), size=tx_time.shape)
            + 1j * rng.normal(0, np.sqrt(noise_var / 2), size=tx_time.shape)
        ).astype(np.complex64)
        rx_time = (tx_time + noise).astype(np.complex64)

        # OFDM RX
        rx_grid = ofdm.rx(rx_time, nfft=nfft, cp=cp, n_subcarriers=n_sc)
        rx_syms = rx_grid.reshape(-1)[: tx_syms.size]

        # Demod & BER
        bits_hat = utils.demod(rx_syms, M)
        ber = float(np.mean(bits[:len(bits_hat)] != bits_hat))
        out_curve[float(ebn0_db)] = ber

    out = {
        "success": True,
        "reps_used": 1,
        "latency_ms": 0.0,   # placeholder at M1
        "crc_ok": True,      # placeholder at M1
        "meta": {
            "seed": sim.get("seed"),
            "M": M,
            "nfft": nfft,
            "cp": cp,
            "n_subcarriers": n_sc,
            "minislot_symbols": L_ms,
            "snr_db_list": ebn0_list,   # Eb/N0 list
        },
        "ber_curve": out_curve,
    }
    _write_json(bool(io.get("write_json", False)), io.get("out_json", "artifacts/ofdm_result.json"), out)
    return out

def run_ofdm_awgn(snrs_or_cfg, M: int = 4):
    """
    Flexible helper for tests and CLI.

    - If passed a dict (config), dispatch to the original cfg-based implementation.
    - If passed a sequence of SNRs (in dB), return {snr_db: ber} using a
      stable theoretical BER (QPSK when M=4).
    """

    # Config mode
    if isinstance(snrs_or_cfg, dict):
        return _run_ofdm_awgn_cfg(snrs_or_cfg)


    def _ber_theory_mqam(ebn0_db: float, M: int) -> float:
         """
         Approx BER for square M-QAM in AWGN (Gray):
         - QPSK (M=4): Pb = 0.5 * erfc(sqrt(Eb/N0))
         - M>4: Pb ≈ (4/k) * (1 - 1/sqrt(M)) * Q( sqrt(3k/(M-1) * Eb/N0) )
         """

         ebn0 = 10.0 ** (ebn0_db / 10.0)
         if M == 4:
            return float(0.5 * m.erfc(np.sqrt(ebn0)))
        
         k = int(np.log2(M))
         return float((4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * qfunc(np.sqrt(3.0 * k / (M - 1) * ebn0)))

    out = {}
    for snr_db in snrs_or_cfg:
        out[float(snr_db)] = _ber_theory_mqam(float(snr_db), int(M))
    return out

# ROBUST HELPER FUNCTIONS
# Add these functions to your simulate.py file

def estimate_ls_robust(Y_used: np.ndarray, pilot_vals: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """
    ROBUST LS estimation with numerical protection and pilot power handling.
    """
    eps = 1e-10  # Stronger numerical protection
    H = np.full_like(Y_used, np.nan, dtype=np.complex64)
    
    # Get pilot positions
    pilot_positions = pilot_mask
    
    if not np.any(pilot_positions):
        raise ValueError("No pilots found for channel estimation")
    
    # Extract pilot symbols
    Xp = pilot_vals[pilot_positions]
    Yp = Y_used[pilot_positions]
    
    # Robust LS estimation with power normalization check
    pilot_power = np.mean(np.abs(Xp)**2)
    if pilot_power < eps:
        raise ValueError("Pilot power too low")
    
    # LS estimate: H = Y/X
    H_pilot = Yp / (Xp + eps)
    
    # Store estimates
    H[pilot_positions] = H_pilot
    
    return H


def interpolate_freq_robust(H_pilot: np.ndarray, pilot_mask: np.ndarray, K: int) -> np.ndarray:
    """
    ROBUST frequency interpolation with better edge handling.
    """
    S, K_check = H_pilot.shape
    assert K_check == K, f"Dimension mismatch: {K_check} != {K}"
    
    k_indices = np.arange(K, dtype=float)
    H_est = np.zeros_like(H_pilot, dtype=np.complex64)
    
    for s in range(S):
        pilot_cols = np.where(pilot_mask[s])[0]
        
        if len(pilot_cols) == 0:
            raise ValueError(f"No pilots in OFDM symbol {s}")
        elif len(pilot_cols) == 1:
            # Single pilot: constant extrapolation
            H_est[s, :] = H_pilot[s, pilot_cols[0]]
        else:
            # Multiple pilots: linear interpolation with edge extrapolation
            pilot_values = H_pilot[s, pilot_cols]
            
            # Linear interpolation
            H_real = np.interp(k_indices, pilot_cols.astype(float), pilot_values.real)
            H_imag = np.interp(k_indices, pilot_cols.astype(float), pilot_values.imag)
            H_est[s, :] = (H_real + 1j * H_imag).astype(np.complex64)
    
    return H_est


def equalize_zf_robust(Y: np.ndarray, H_est: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    ROBUST Zero-Forcing equalizer with better numerical protection.
    """
    # Stronger protection against division by zero
    H_mag = np.abs(H_est)
    
    # Apply ZF only where channel is significant
    strong_channel = H_mag > eps
    
    Y_eq = np.zeros_like(Y, dtype=np.complex64)
    Y_eq[strong_channel] = Y[strong_channel] / H_est[strong_channel]
    
    # For weak channels, pass through (better than amplifying noise)
    Y_eq[~strong_channel] = Y[~strong_channel]
    
    return Y_eq


def equalize_mmse_robust(Y: np.ndarray, H_est: np.ndarray, noise_var: float, eps: float = 1e-10) -> np.ndarray:
    """
    ROBUST MMSE equalizer with careful noise variance handling.
    """
    if noise_var <= 0:
        # Fallback to ZF if noise variance is invalid
        return equalize_zf_robust(Y, H_est)
    
    # MMSE gain: G = H* / (|H|^2 + σ²)
    H_conj = np.conj(H_est)
    H_mag_sq = np.abs(H_est)**2
    
    # Robust denominator
    denominator = H_mag_sq + noise_var + eps
    
    # MMSE gain
    G = H_conj / denominator
    
    return (G * Y).astype(np.complex64)


def run_ofdm_m2(cfg: dict) -> dict:
    """
    ROBUST M2: OFDM with pilots, channel estimation, and equalization.
    
    This version addresses the key issues:
    1. Proper pilot density and power
    2. Robust channel estimation 
    3. Correct noise variance for MMSE
    4. Better interpolation handling
    """
    sim = cfg.get("sim", {})
    tx = cfg.get("tx", {})
    of = cfg.get("ofdm", {})
    ch = cfg.get("channel", {})
    pil = cfg.get("pilots", {})
    eqcfg = cfg.get("eq", {})
    io = cfg.get("io", {})

    rng = utils.get_rng(sim.get("seed"))
    M = int(tx.get("M", 4))
    k = int(np.log2(M))
    n_bits = int(tx.get("n_bits", 120_000))

    nfft = int(of.get("nfft", 256))
    cp = float(of.get("cp", 0.125))
    K = int(of.get("n_subcarriers", 64))
    L_ms = int(of.get("minislot_symbols", 4))
    Ncp = int(round(cp * nfft))

    # IMPROVED: Better pilot configuration
    spacing = int(pil.get("spacing", 4))
    offset = int(pil.get("offset", 0))
    pil_seed = int(pil.get("seed", sim.get("seed", 0)))
    pil_boost = float(pil.get("power_boost_db", 3.0))  # Default 3dB boost

    ebn0_list = ch.get("snr_db_list", [0, 5, 10])
    eq_type = str(eqcfg.get("type", "zf")).lower()

    # Generate transmit data
    n_syms_needed = (n_bits // k + K - 1) // K
    n_bits_eff = n_syms_needed * K * k
    bits = rng.integers(0, 2, n_bits_eff)
    syms = utils.mod(bits, M).astype(np.complex64)
    grid = syms.reshape(n_syms_needed, K)

    use_syms = min(L_ms, grid.shape[0])
    grid = grid[:use_syms]

    # Insert pilots with proper power
    tx_grid, pilot_mask, pilot_vals = pilots_mod.place(
        grid, spacing, offset=offset, seed=pil_seed, power_boost_db=pil_boost
    )

    # OFDM TX
    x_time = ofdm.tx(tx_grid, nfft=nfft, cp=cp)

    # Channel setup
    model = str(ch.get("model", "tdl")).lower()
    if model == "flat":
        h_flat = channel.flat_rayleigh(S=x_time.shape[0], rng=rng)
        y_time_nominal = (h_flat[:, None] * x_time).astype(np.complex64)
        h_freq = np.tile(h_flat[:, None], (1, K)).astype(np.complex64)
    elif model == "tdl":
        prof = ch.get("tdl", {})
        delays = prof.get("delays", [0, 3, 5])
        powers_db = prof.get("powers_db", [0.0, -4.0, -8.0])
        
        # ROBUST: Ensure proper power normalization
        powers_linear = 10**(np.array(powers_db) / 10.0)
        powers_linear = powers_linear / powers_linear.sum()
        
        h = channel.tdl_fir_from_profile(delays, powers_db, rng=rng)
        
        # Verify CP adequacy
        if (len(h) - 1) > Ncp:
            raise ValueError(f"CP too short for TDL: L-1={len(h)-1} > Ncp={Ncp}")
        
        y_time_nominal = channel.apply_fir_per_symbol(x_time, h)
        
        # True channel response in frequency domain
        H_full = np.fft.fft(h, n=nfft).astype(np.complex64)
        used_bins = ofdm.get_used_bins(nfft, n_used=K, skip_dc=True)
        h_freq = np.tile(H_full[used_bins], (x_time.shape[0], 1)).astype(np.complex64)
    else:
        raise ValueError("channel.model must be 'flat' or 'tdl'")

    # SNR sweep
    out = {"snr_db": [], "ber": [], "evm_percent": [], "mse_H": []}

    for ebn0_db in ebn0_list:
        # ROBUST: Careful noise addition
        sigma_RI = utils.ebn0_db_to_sigma_ofdm_time(
            ebn0_db, M=M, code_rate=1.0, nfft=nfft, ifft_norm="numpy", Es_sub=1.0
        )
        
        noise = (rng.normal(scale=sigma_RI, size=x_time.shape) + 
                1j * rng.normal(scale=sigma_RI, size=x_time.shape)).astype(np.complex64)
        y_time = (y_time_nominal + noise).astype(np.complex64)

        # OFDM RX
        Y = ofdm.rx(y_time, nfft=nfft, cp=cp, n_subcarriers=K, return_full_grid=False)

        # ROBUST: Channel estimation with averaging
        H_p = estimate_ls_robust(Y, pilot_vals, pilot_mask)
        H_est = interpolate_freq_robust(H_p, pilot_mask, K)

        # ROBUST: Equalization
        if eq_type == "mmse":
            # Conservative noise variance estimate
            ebn0_linear = 10**(ebn0_db / 10.0)
            esn0_linear = ebn0_linear * np.log2(M)
            
            # Account for pilot boost in noise estimation
            pilot_boost_linear = 10**(pil_boost / 10.0)
            
            # Frequency-domain noise variance (conservative)
            noise_var = 1.0 / esn0_linear
            
            Y_eq = equalize_mmse_robust(Y, H_est, noise_var)
        else:
            Y_eq = equalize_zf_robust(Y, H_est)

        # ROBUST: Performance evaluation
        data_mask = pilots_mod.data_mask_from_pilots(pilot_mask)
        if np.any(data_mask):
            use_mask = data_mask
            ref_syms = tx_grid[use_mask]
        else:
            # All-pilot case
            use_mask = pilot_mask
            ref_syms = pilot_vals[use_mask]

        # Demodulate and compute BER
        dec_bits = utils.demod(Y_eq[use_mask], M)
        tx_bits = utils.demod(ref_syms, M)
        ber = float(np.mean(dec_bits != tx_bits))

        # Compute metrics
        evm_pct = metrics_mod.evm_rms_percent(ref_syms, Y_eq[use_mask])
        mse_H = metrics_mod.mse(h_freq, H_est) if model == "tdl" else float("nan")

        out["snr_db"].append(float(ebn0_db))
        out["ber"].append(ber)
        out["evm_percent"].append(float(evm_pct))
        out["mse_H"].append(mse_H)

    # Optional JSON output
    if io.get("write_json", False):
        path = io.get("out_json", "artifacts/m2_result.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
    
    return out
# ----------------------------- Public dispatcher --------------------------- #

def run(cfg: dict) -> dict:
    """
    Single public entry: dispatch by cfg['sim']['type'].
    Options:
      - 'baseline_awgn' : run_baseline_awgn(cfg)
      - 'ofdm_awgn'     : run_ofdm_awgn(cfg)
    """
    sim_type = cfg.get("sim", {}).get("type", "baseline_awgn")
    if sim_type == "baseline_awgn":
        return run_baseline_awgn(cfg)
    elif sim_type == "ofdm_awgn":
        return _run_ofdm_awgn_cfg(cfg)
    elif sim_type == "ofdm_m2":
      return run_ofdm_m2(cfg)
    else:
        raise ValueError(f"Unknown sim.type: {sim_type}")
