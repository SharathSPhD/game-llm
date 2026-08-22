#!/usr/bin/env bash
# Start (or restart) the Kinetic AI backend on the GB10.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .gateway.env
pkill -f "uvicorn app.server:app" 2>/dev/null || true
sleep 1
LOG="${1:-/tmp/kinetic_backend.log}"
nohup setsid env \
  GATEWAY_SECRET="$GATEWAY_SECRET" \
  ALLOWED_ORIGINS="$ALLOWED_ORIGINS" \
  RESULTS_DIR="$(pwd)/results" \
  .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8097 \
  > "$LOG" 2>&1 < /dev/null &
echo "backend starting (log: $LOG)"
