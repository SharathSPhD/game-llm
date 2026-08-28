#!/usr/bin/env bash
# Thermal governor: keeps a process below the GB10's ~91C hardware trip by
# pausing (SIGSTOP) above PAUSE_MC and resuming (SIGCONT) below RESUME_MC.
# Usage: thermal_governor.sh <pgrep-pattern> [pause_mC] [resume_mC]
PATTERN="$1"; PAUSE=${2:-85000}; RESUME=${3:-78000}
# The pattern appears in this script's own command line, so pgrep -f matches the
# governor itself. Selecting that PID makes the governor pause itself and leave
# the workload running — silently unprotected. Exclude self and parent.
SELF=$$; PARENT=$PPID
LOG=/home/sharaths/projects/game-llm/results/thermal_governor.log
state=running
while true; do
  T=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  PID=$(pgrep -f "$PATTERN" | grep -vx -e "$SELF" -e "$PARENT" | head -1)
  if [ -n "$PID" ] && [ -n "$T" ]; then
    if [ "$state" = running ] && [ "$T" -gt "$PAUSE" ]; then
      kill -STOP "$PID" && state=paused
      echo "$(date -Is) PAUSE at ${T}mC (pid $PID)" >> "$LOG"; sync "$LOG" 2>/dev/null
    elif [ "$state" = paused ] && [ "$T" -lt "$RESUME" ]; then
      kill -CONT "$PID" && state=running
      echo "$(date -Is) RESUME at ${T}mC (pid $PID)" >> "$LOG"; sync "$LOG" 2>/dev/null
    fi
  fi
  sleep 10
done
