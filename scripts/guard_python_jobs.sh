#!/usr/bin/env bash
# Attach a PID-based thermal governor to each compute process of a multi-stage
# job as it appears.
#
# A sweep that loads several models in turn is a different process each time, so
# a single governor guards only the first stage and then exits. This supervisor
# closes that gap without reintroducing the pattern-matching bug: it matches only
# the interpreter's own command line (never a shell, never a monitor that merely
# mentions the job), resolves the match to a PID once, and hands that PID to the
# governor, which never matches anything again.
#
# Usage: guard_python_jobs.sh <cmdline-regex> [pause_mC] [resume_mC]
set -u
RE="$1"; PAUSE=${2:-82000}; RESUME=${3:-74000}
HERE=$(dirname "$(readlink -f "$0")")
declare -A guarded=()
while true; do
  for pid in $(pgrep -f "$RE"); do
    exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null)
    case "$exe" in *python*) ;; *) continue ;; esac   # compute jobs only
    [ -n "${guarded[$pid]:-}" ] && continue
    # A second supervisor, or a restart of this one, must not attach a second
    # governor to the same process: two governors race, one resuming what the
    # other paused, which leaves the job running hot while both logs claim it is
    # paused. Check the live process table, not just this instance's memory.
    pgrep -f "thermal_governor.sh $pid( |$)" >/dev/null 2>&1 && continue
    guarded[$pid]=1
    nohup bash "$HERE/thermal_governor.sh" "$pid" "$PAUSE" "$RESUME" >/dev/null 2>&1 &
  done
  sleep 15
done
