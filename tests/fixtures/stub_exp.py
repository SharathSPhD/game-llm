"""Stub experiment for testing Studio API.

Minimal experiment script that writes fake results.json without GPU.

Usage:
    python tests/fixtures/stub_exp.py --config config.yaml --output /tmp/results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def main() -> None:
    """Run stub experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device (ignored)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to log file
    log_file = output_dir / "run.log"
    with open(log_file, "w") as f:
        f.write("Stub experiment starting...\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Output: {args.output}\n")
        f.flush()
        time.sleep(0.1)

        # Load config
        with open(args.config) as cfg_f:
            config = yaml.safe_load(cfg_f) or {}
        f.write(f"Loaded config with keys: {list(config.keys())}\n")
        f.flush()

        # Simulate work
        for i in range(3):
            f.write(f"Step {i+1}/3: processing...\n")
            f.flush()
            time.sleep(0.05)

    # Write fake results.json
    results = {
        "experiment": "stub_exp",
        "config_hash": "abc123",
        "git_commit": "deadbeef",
        "metrics": {
            "final_loss": 2.5,
            "final_accuracy": 0.42,
            "wall_time": 0.5,
        },
    }
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
