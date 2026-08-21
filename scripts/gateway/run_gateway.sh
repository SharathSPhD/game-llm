#!/usr/bin/env bash
# Permanent-URL gateway: FastAPI backend + cloudflared quick tunnel, with the
# tunnel URL pushed to Cloudflare KV so the stable Worker
# (kinetic.sharath-sathish.workers.dev) always proxies to the current backend.
# The public URL never changes; only the KV value does.

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-8097}"
CF_ACCT="${CF_ACCT:-139e7fa343ba8a280261f8d7d92cdeda}"
KV_ID="${KV_ID:-3070c81152aa4526bf63899da5584155}"
: "${CF_EMAIL:?export CF_EMAIL}"; : "${CF_KEY:?export CF_KEY}"

cd "$REPO"

# Start FastAPI server (assumes venv at .venv)
echo "Starting FastAPI backend on port $PORT..."
nohup .venv/bin/python -m uvicorn app.server:app \
  --host 127.0.0.1 --port "$PORT" \
  > /tmp/kinetic_gateway.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "FastAPI backend ready"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "FastAPI backend failed to start"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
  fi
  sleep 2
done

# Start cloudflared tunnel
echo "Starting cloudflared tunnel..."
nohup cloudflared tunnel --url "http://localhost:$PORT" \
  > /tmp/kinetic_tunnel.log 2>&1 &
TUNNEL_PID=$!

# Extract tunnel URL from logs
URL=""
for i in $(seq 1 30); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/kinetic_tunnel.log | head -1)
  [ -n "$URL" ] && break
  sleep 2
done

if [ -z "$URL" ]; then
  echo "Failed to obtain tunnel URL"
  kill $SERVER_PID $TUNNEL_PID 2>/dev/null || true
  exit 1
fi

# Update KV with the tunnel URL
echo "Updating Cloudflare KV with backend URL: $URL"
curl -s -X PUT \
  "https://api.cloudflare.com/client/v4/accounts/$CF_ACCT/storage/kv/namespaces/$KV_ID/values/gateway_url" \
  -H "X-Auth-Email: $CF_EMAIL" -H "X-Auth-Key: $CF_KEY" \
  --data "$URL" >/dev/null

if [ $? -eq 0 ]; then
  echo "✓ Gateway is live!"
  echo "  Public URL:  https://kinetic.sharath-sathish.workers.dev"
  echo "  Backend:     $URL"
  echo "  Local:       http://localhost:$PORT"
else
  echo "✗ Failed to update KV"
  kill $SERVER_PID $TUNNEL_PID 2>/dev/null || true
  exit 1
fi

# Keep running
wait $SERVER_PID $TUNNEL_PID
