
# scripts/nr_step1_smoketest.py
import argparse, json, yaml, os
from pathlib import Path
from nr_urllc import urllc as urllc_mod
from nr_urllc import nr_timing as nr_time_mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default="artifacts/nr_step1_meta.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    dummy = {"meta": {}}
    # attach step0 if available
    try:
        urllc_mod.attach_and_maybe_write(dummy, cfg)
    except Exception as e:
        print("[warn] URLLC attach failed:", e)
    # attach step1
    nr_time_mod.attach_step1_meta(dummy, cfg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dummy["meta"], f, indent=2)
    print("[ok] wrote", args.out)

if __name__ == "__main__":
    main()
