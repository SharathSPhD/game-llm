#!/usr/bin/env bash
# Fairness-audit baselines for the council side-result, queued behind the
# paradigm test so the core architecture question gets the GPU first.
set -u
cd /home/sharaths/projects/game-llm
until grep -q "EXP31 ADAPTIVE DEPTH COMPLETE" results/scale/exp31.log 2>/dev/null; do
  sleep 120
done
echo "$(date -Is) === exp30 fair baselines start ==="
.venv-scale/bin/python experiments/exp30_fair_baselines.py
echo "$(date -Is) === exp30 rc=$? ==="
echo "EXP30 FAIR BASELINES COMPLETE"
