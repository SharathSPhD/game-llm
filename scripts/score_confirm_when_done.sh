#!/usr/bin/env bash
# Score the SPEC 0017 confirmation the moment generation completes.
set -u
cd "$(dirname "$0")/.."
until grep -q "EXP23 CONFIRMATION COMPLETE" results/scale/exp23_confirm.log 2>/dev/null; do
  sleep 60
done
.venv/bin/python experiments/exp27_anchored_vote.py --preregistered \
  --root results/scale/exp23_confirm \
  --out results/scale/exp27_confirmation.json > results/scale/exp27_confirmation.log 2>&1
echo "CONFIRMATION SCORED rc=$?"
