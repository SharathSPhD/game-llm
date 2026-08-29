#!/usr/bin/env bash
# The paradigm test: one model, matched parameters, matched average compute,
# adaptive depth against a fixed stack. Queued directly behind exp29.
set -u
cd /home/sharaths/projects/game-llm
until grep -q "EXP29 STABILITY PROBE COMPLETE" results/scale/exp29.log 2>/dev/null; do
  sleep 60
done
echo "$(date -Is) === exp31 adaptive depth start ==="
.venv-scale/bin/python experiments/exp31_adaptive_depth.py \
  --eqlm results/scale/ckpt/eqlm_anytime_seed42.pt \
  --explicit results/scale/ckpt/explicit_baseline_seed42.pt
echo "$(date -Is) === exp31 rc=$? ==="
echo "EXP31 ADAPTIVE DEPTH COMPLETE"
