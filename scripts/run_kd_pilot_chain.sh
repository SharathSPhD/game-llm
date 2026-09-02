#!/usr/bin/env bash
# SPEC 0021 pilot chain: wait for cache -> ship -> smoke -> full pilot on 5090.
set -u
cd "$(dirname "$0")/.."
until grep -q "wrote .*tokens" results/scale/kd_cache.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) cache ready, shipping to 5090"
rsync -aL experiments/exp38_kd_pilot.py ss@192.168.0.204:~/fusion-project/kinetic_exp10/experiments/
rsync -aL --progress data/cache/kd_fineweb_qwen.pt ss@192.168.0.204:~/fusion-project/kinetic_exp10/data/cache/ 2>&1 | tail -1
echo "$(date -Is) smoke"
ssh ss@192.168.0.204 "docker exec rtx5090-train bash -lc 'cd /fusion-project/kinetic_exp10 && python experiments/exp38_kd_pilot.py --steps 30 --batch 4 --out results/exp38_smoke.json' 2>&1 | tail -4"
if ssh ss@192.168.0.204 "test -f ~/fusion-project/kinetic_exp10/results/exp38_smoke.json"; then
  echo "SMOKE OK -> full pilot"
  ssh ss@192.168.0.204 "cd ~/fusion-project/kinetic_exp10 && setsid nohup docker exec rtx5090-train bash -lc 'cd /fusion-project/kinetic_exp10 && python experiments/exp38_kd_pilot.py --steps 4000 --batch 8 --out results/exp38_kd_pilot.json' > exp38_5090.log 2>&1 < /dev/null &"
  echo "PILOT LAUNCHED"
else
  echo "SMOKE FAILED"
fi
