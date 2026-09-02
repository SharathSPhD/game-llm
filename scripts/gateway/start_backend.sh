#!/usr/bin/env bash
# Start (or restart) the Kinetic AI backend on whichever host the serving profile
# names (KINETIC_SERVE_PROFILE, configs/serve/profiles/; ADR 0010). The profile
# supplies port, origins and the device policy; .gateway.env supplies secrets.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .gateway.env
export KINETIC_SERVE_PROFILE="${KINETIC_SERVE_PROFILE:-rtx5090}"
PORT="$(.venv/bin/python -m kinetic_ai.serve.profile port)"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-$(.venv/bin/python -m kinetic_ai.serve.profile allowed_origins)}"
echo "profile $KINETIC_SERVE_PROFILE: port $PORT, device $(.venv/bin/python -m kinetic_ai.serve.profile resolved_device)"
pkill -f "uvicorn app.server:app" 2>/dev/null || true
sleep 1
LOG="${1:-/tmp/kinetic_backend.log}"
nohup setsid env \
  GATEWAY_SECRET="$GATEWAY_SECRET" \
  ALLOWED_ORIGINS="$ALLOWED_ORIGINS" \
  KINETIC_SERVE_PROFILE="$KINETIC_SERVE_PROFILE" \
  RESULTS_DIR="$(pwd)/results" \
  .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
echo "backend starting (log: $LOG)"
