#!/usr/bin/env bash
# Daemonized exp10 full run — survives terminal/Claude restarts.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/exp10_full/run.log
mkdir -p results/exp10_full
nohup setsid .venv/bin/python experiments/exp05_eqlm_pretrain.py \
  --config configs/exp10_full.yaml --output results/exp10_full \
  >> "$LOG" 2>&1 < /dev/null &
echo "exp10 running (pid $!), log: $LOG"
