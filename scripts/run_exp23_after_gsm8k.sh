#!/usr/bin/env bash
# Queue the cross-examination arena behind the GSM8K re-measurement.
# The GPU lock is one job at a time, so this waits for the marker the GSM8K
# sweep writes rather than starting alongside it.
set -u
cd /home/sharaths/projects/game-llm
until grep -q "GSM8K FIXED SWEEP COMPLETE" results/scale/gsm8k_fixed.log 2>/dev/null; do
  sleep 60
done
for SEED in 42 43 44; do
  echo "$(date -Is) === exp23 seed $SEED ==="
  .venv-scale/bin/python experiments/exp23_cross_examination.py --seed $SEED
  echo "$(date -Is) === exp23 seed $SEED rc=$? ==="
done
echo "EXP23 CROSS-EXAMINATION COMPLETE"
