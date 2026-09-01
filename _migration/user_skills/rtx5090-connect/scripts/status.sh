#!/usr/bin/env bash
# Usage: status.sh <remote_job_name> [tail_lines] [user@host]
# Tails a job's log and reports current GPU utilization, so you can tell a
# genuinely running job apart from one that's silently stalled.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

JOB_NAME="${1:?usage: status.sh <remote_job_name> [tail_lines] [user@host]}"
TAIL_LINES="${2:-40}"
[ -n "${3:-}" ] && RTX5090_HOST="$3"

echo "== log tail: $JOB_NAME =="
rtx_ssh "tail -n $TAIL_LINES $RTX5090_PROJECT_DIR/$JOB_NAME/train.log 2>/dev/null" \
  || echo "(no train.log found -- job may not have started, or used a different log path)"

echo
echo "== gpu right now =="
rtx_ssh "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv"

echo
echo "== process alive inside container? =="
rtx_ssh "cd $RTX5090_COMPOSE_DIR && docker compose exec -T train pgrep -af python" \
  || echo "(no python process found in container -- job may have finished or crashed; check the log above)"
