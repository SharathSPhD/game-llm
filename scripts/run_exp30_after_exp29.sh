#!/usr/bin/env bash
# Fairness-audit baselines, queued behind the stability probe.
set -u
cd /home/sharaths/projects/game-llm
until grep -q "EXP29 STABILITY PROBE COMPLETE" results/scale/exp29.log 2>/dev/null; do
  sleep 120
done
echo "$(date -Is) === exp30 fair baselines start ==="
.venv-scale/bin/python experiments/exp30_fair_baselines.py
echo "$(date -Is) === exp30 rc=$? ==="
echo "EXP30 FAIR BASELINES COMPLETE"
