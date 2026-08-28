#!/usr/bin/env bash
# Thermal governor: keeps one compute job below the platform's thermal limit by
# pausing it (SIGSTOP) above PAUSE_MC and resuming it (SIGCONT) below RESUME_MC.
#
# Usage:  thermal_governor.sh <pid> [pause_mC] [resume_mC]
#
# It takes a PID, not a pattern, and that is deliberate. Pattern matching has now
# failed three times in this project, each time silently: `pgrep -f` matches any
# command line containing the string, which includes this script's own arguments,
# the shell wrapper that launched the job, and — most damagingly — unrelated
# monitors that merely mention the job's name inside a nested command. On the
# third occurrence the governor stopped a remote-training watchdog while the GPU
# job it was meant to guard ran to 90C unprotected. A PID names exactly one
# process, so the failure mode does not exist.
#
# To guard a job you are launching, capture its PID:
#     nohup my_job & echo $! > job.pid
#     nohup bash scripts/thermal_governor.sh "$(cat job.pid)" &
set -u

PID="${1:-}"
PAUSE=${2:-82000}
RESUME=${3:-74000}
LOG=/home/sharaths/projects/game-llm/results/thermal_governor.log

if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <pid> [pause_mC] [resume_mC]" >&2
  exit 2
fi
if ! kill -0 "$PID" 2>/dev/null; then
  echo "thermal_governor: pid $PID is not running" >&2
  exit 2
fi

CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | cut -c1-120)
echo "$(date -Is) GUARD pid $PID pause=${PAUSE}mC resume=${RESUME}mC :: $CMD" >> "$LOG"

state=running
while kill -0 "$PID" 2>/dev/null; do
  T=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  if [ -n "$T" ]; then
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

# A job that exits while paused would leave a stopped process behind, so release
# it unconditionally on the way out.
kill -CONT "$PID" 2>/dev/null
echo "$(date -Is) DONE pid $PID exited" >> "$LOG"
