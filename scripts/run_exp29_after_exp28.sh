#!/usr/bin/env bash
# Cycle 29 stability probe, queued behind exp28 so the GPU lock holds.
set -u
cd "$(dirname "$0")/.."
until grep -q "EXP28 SECOND FAMILY COMPLETE" results/scale/exp28.log 2>/dev/null; do
  sleep 120
done
echo "$(date -Is) === exp29 stability probe start ==="
.venv-scale/bin/python experiments/exp29_stability_probe.py
echo "$(date -Is) === exp29 rc=$? ==="
echo "EXP29 STABILITY PROBE COMPLETE"
