# scripts/run_sims.py
import argparse, yaml, json
from nr_urllc import simulate
from nr_urllc.sweep import autoramp_ofdm_qpsk_sweep, autoramp_sc_qpsk_sweep

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--out", default="artifacts/result.json")
    args = p.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg.get("sim", {})            # nested
    tx_cfg  = cfg.get("tx", {})
    ofdm    = cfg.get("ofdm", {})
    ch      = cfg.get("channel", {})
    auto    = cfg.get("autoramp", {})

    mode = str(sim_cfg.get("type", "")).lower()
    use_autoramp = bool(sim_cfg.get("autoramp", False))

    if mode == "ofdm_awgn" and use_autoramp:
        res = autoramp_ofdm_qpsk_sweep(
            snr_db_list=ch["snr_db_list"],
            seed=sim_cfg["seed"],
            M=tx_cfg.get("M", 4),
            nfft=ofdm["nfft"],
            cp=ofdm["cp"],
            n_subcarriers=ofdm["n_subcarriers"],
            minislot_symbols=ofdm["minislot_symbols"],
            target_errs=auto.get("target_errs", 100),
            min_bits=auto.get("min_bits", 20_000),
            max_bits=auto.get("max_bits", 2_000_000),
        )
        result = {
            "success": res.success,
            "reps_used": 1,
            "latency_ms": 0.0,
            "crc_ok": True,
            "meta": res.meta,
            "ber_curve": res.ber_curve,
            "n_bits_curve": res.n_bits_curve,
            "n_errs_curve": res.n_errs_curve,
        }

    elif mode == "sc_awgn" and use_autoramp:
        res = autoramp_sc_qpsk_sweep(
            ebn0_db_list=ch["ebn0_db_list"],
            seed=sim_cfg["seed"],
            M=tx_cfg.get("M", 4),
            target_errs=auto.get("target_errs", 100),
            min_bits=auto.get("min_bits", 20_000),
            max_bits=auto.get("max_bits", 2_000_000),
        )
        result = {
            "success": res.success,
            "reps_used": 1,
            "latency_ms": 0.0,
            "crc_ok": True,
            "meta": res.meta,
            "ber_curve": res.ber_curve,
            "n_bits_curve": res.n_bits_curve,
            "n_errs_curve": res.n_errs_curve,
        }
    elif mode == "ofdm_m2":
    # no autoramp for M2 (yet) → delegate to simulate.run
        result = simulate.run(cfg)
    else:
        # fallback to your existing pipeline
        result = simulate.run(cfg)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("Simulation result saved to", args.out)

if __name__ == "__main__":
    main()

