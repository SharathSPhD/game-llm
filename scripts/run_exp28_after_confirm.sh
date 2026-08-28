#!/usr/bin/env bash
# Second-family generality test, queued behind the SPEC 0017 confirmation so
# the single-GPU lock is respected.
set -u
cd /home/sharaths/projects/game-llm
until grep -q "EXP23 CONFIRMATION COMPLETE" results/scale/exp23_confirm.log 2>/dev/null; do
  sleep 120
done
echo "$(date -Is) === exp28 second-family start ==="
.venv-scale/bin/python experiments/exp28_second_family.py
echo "$(date -Is) === exp28 rc=$? ==="
echo "EXP28 SECOND FAMILY COMPLETE"
