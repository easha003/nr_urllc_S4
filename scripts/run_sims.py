import argparse
import yaml
import json
from nr_urllc import simulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--out", default="artifacts/result.json", help="Output path")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    result = simulate.run(cfg)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("Simulation result saved to", args.out)
