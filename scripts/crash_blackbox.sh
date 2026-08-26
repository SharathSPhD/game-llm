#!/usr/bin/env bash
# Black-box recorder: 30s samples of power/thermal/load, synced to disk so the
# final pre-crash sample survives a hard power loss.
OUT="${1:-/home/sharaths/projects/game-llm/results/blackbox.log}"
while true; do
  {
    echo "=== $(date -Is)"
    cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ' '; echo " (thermal mC)"
    nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null
    awk '{print "load:", $1, $2, $3}' /proc/loadavg
    free -m | awk '/Mem:/{print "memMB used:", $3}'
  } >> "$OUT" 2>&1
  sync "$OUT" 2>/dev/null || sync
  sleep 30
done
