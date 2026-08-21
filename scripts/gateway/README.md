# Kinetic AI gateway — permanent public URL

**Public URL (stable, never changes):** https://kinetic.kinetic-ai.workers.dev

## Architecture

```
Web app (Vercel)  ──┐
                     │  ┌─────────────────────┐
                     └─▶│ Cloudflare Worker   │
                        │  (worker.js)        │
                     ┌─▶│  reads KV           │
                     │  └─────────────────────┘
API requests ───────┘          │
                               │ current tunnel URL
                        ┌──────▼──────────┐
                        │ cloudflared     │
                        │ quick tunnel    │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  GB10           │
                        │  FastAPI        │
                        │  app/server.py  │
                        └─────────────────┘
```

The Worker (`worker.js`, deployed to `kinetic.kinetic-ai.workers.dev`) is
the permanent front door. Behind it, an ephemeral cloudflared tunnel reaches the
GB10; its URL is stored in KV. On any gateway restart, the KV value is refreshed,
but the public URL is unaffected.

## Starting the Gateway

### Prerequisites

- `cloudflared` CLI installed (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
- Cloudflare account with:
  - KV namespace bound (see `run_gateway.sh` for KV_ID)
  - Worker deployed (see deployment below)
- FastAPI app running on port 8097 (controlled by PORT env var)
- Venv at `.venv` with FastAPI + uvicorn

### Start Gateway

```bash
export CF_EMAIL='sharath.sathish@outlook.com'
export CF_KEY='<your cloudflare global api key>'
bash scripts/gateway/run_gateway.sh
```

This:
1. Starts the FastAPI app on localhost:8097
2. Launches a cloudflared tunnel to it
3. Extracts the ephemeral tunnel URL
4. Writes that URL to Cloudflare KV
5. Logs the public URL (which never changes)

Example output:
```
✓ Gateway is live!
  Public URL:  https://kinetic.kinetic-ai.workers.dev
  Backend:     https://abc-def-ghi.trycloudflare.com
  Local:       http://localhost:8097
```

### Logs

- FastAPI: `/tmp/kinetic_gateway.log`
- Tunnel: `/tmp/kinetic_tunnel.log`

## Environment Variables

All are optional except CF_EMAIL and CF_KEY:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CF_EMAIL` | *(required)* | Cloudflare account email |
| `CF_KEY` | *(required)* | Cloudflare global API key |
| `CF_ACCT` | `139e7fa...` | Cloudflare account ID |
| `KV_ID` | `3070c81...` | KV namespace ID |
| `PORT` | `8097` | FastAPI server port |

## Deploying the Worker

One-time setup:

```bash
# Install Wrangler (Cloudflare CLI)
npm install -g wrangler

# Login
wrangler login

# Deploy the worker (from this directory)
wrangler deploy --name kinetic
```

This creates the Worker at `kinetic.kinetic-ai.workers.dev` and binds it
to the KV namespace. The Worker pulls the backend URL from KV and proxies `/api/*`
requests to it.

## Manually Update KV

If you need to point the gateway at a different backend:

```bash
curl -X PUT \
  "https://api.cloudflare.com/client/v4/accounts/$CF_ACCT/storage/kv/namespaces/$KV_ID/values/gateway_url" \
  -H "X-Auth-Email: $CF_EMAIL" -H "X-Auth-Key: $CF_KEY" \
  --data 'https://<new-tunnel-url>.trycloudflare.com'
```

## API Endpoints

All endpoints are served at the public URL. Examples:

```bash
# Health check (no auth required)
curl https://kinetic.kinetic-ai.workers.dev/health

# Solve equilibrium (requires Bearer token)
curl -X POST https://kinetic.kinetic-ai.workers.dev/api/solve \
  -H "Authorization: Bearer $GATEWAY_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"game":"rps","method":"mmd_fixed","lr":0.1,"tau":1.0,"steps":100,"seed":42}'
```

See `app/server.py` for full endpoint documentation.

## Troubleshooting

**Gateway not responding:**
1. Check cloudflared logs: `tail -f /tmp/kinetic_tunnel.log`
2. Check FastAPI logs: `tail -f /tmp/kinetic_gateway.log`
3. Verify KV contains correct URL: `curl https://api.cloudflare.com/client/v4/accounts/$CF_ACCT/storage/kv/namespaces/$KV_ID/values/gateway_url -H "X-Auth-Email: $CF_EMAIL" -H "X-Auth-Key: $CF_KEY"`

**Tunnel URL keeps changing:**
This is normal. Each `run_gateway.sh` creates a new ephemeral tunnel URL, but
the KV is updated automatically. The public Worker URL never changes.

**KV namespace not found:**
Verify KV_ID is correct in `run_gateway.sh` or your Cloudflare dashboard.
