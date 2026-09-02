#!/usr/bin/env bash
# SPEC 0017 confirmation seeds: fresh prompt draws never seen by the offline
# analysis that selected the mechanism.
set -u
cd "$(dirname "$0")/.."
for SEED in 45 46 47; do
  echo "$(date -Is) === exp23 confirm seed $SEED ==="
  .venv-scale/bin/python experiments/exp23_cross_examination.py --seed $SEED --out results/scale/exp23_confirm
  echo "$(date -Is) === seed $SEED rc=$? ==="
done
echo "EXP23 CONFIRMATION COMPLETE"
