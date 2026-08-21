# EqLM Web Frontend — Deployment Guide

## Overview

EqLM web is a Next.js 14 (app router) + React 18 + TailwindCSS frontend for the Equilibrium Lab research platform. It provides interactive pages for running solvers, tracing QRE paths, auctions, and managing training jobs.

**Stack:**
- Next.js 14 (app router)
- React 18
- TailwindCSS 3
- TypeScript 5
- @supabase/ssr (optional)
- Node.js 18+

**Deployment:** Vercel (root: `apps/web/`)

---

## Local Development

### Setup

```bash
# From apps/web directory
npm install

# Create .env.local (copy from .env.example)
cp .env.example .env.local

# Optional: configure gateway
export GATEWAY_URL=http://localhost:8000
export GATEWAY_SECRET=dev-secret

# Optional: configure Supabase (auth soft-disables if unset)
export NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
export NEXT_PUBLIC_SUPABASE_ANON_KEY=your-key
```

### Run

```bash
npm run dev
# Open http://localhost:3000
```

### Build

```bash
# Type check
npm run type-check

# Build
npm run build

# Start production server
npm start

# Smoke test (requires build)
npm run check-pages
```

---

## Environment Variables

### Public (visible in browser)

- `NEXT_PUBLIC_SUPABASE_URL` — Supabase project URL (optional)
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` — Supabase anon key (optional)

### Server-side only (never sent to browser)

- `GATEWAY_URL` — Backend API URL (e.g., `http://backend:8000`)
- `GATEWAY_SECRET` — Bearer token for `/api/proxy/*` forwarding

### Behavior

- **When `GATEWAY_URL` is set:** All API requests are forwarded to the gateway with `Authorization: Bearer {GATEWAY_SECRET}`
- **When `GATEWAY_URL` is unset:** App enters **REPLAY_MODE**, serving canned demo responses (flags `replay: true`)
- **When Supabase keys are unset:** Auth softly disables; all pages remain readable

---

## API Proxy Routes

The frontend forwards requests to the backend via Next.js API routes:

| Route | Backend | Method | Auth |
|-------|---------|--------|------|
| `/api/proxy/api/solve` | `/api/solve` | POST | Bearer |
| `/api/proxy/api/qre_path` | `/api/qre_path` | POST | Bearer |
| `/api/proxy/api/auction` | `/api/auction` | POST | Bearer |
| `/api/proxy/api/jobs` | `/api/jobs` | POST | Bearer |
| `/api/proxy/api/jobs/{id}` | `/api/jobs/{id}` | GET | Bearer |
| `/api/proxy/api/results` | `/api/results` | GET | Bearer |
| `/api/health` | `/health` | GET | None |

See `app/api/proxy/[...path]/route.ts` for implementation.

---

## Backend Gateway Setup

The backend server must expose the following endpoints. See `app/server.py` in the repo root for the reference implementation.

**Health check (no auth):**
```bash
GET /health
→ { "status": "ok", "version": "0.1.0-...", "gpu_available": true/false }
```

**Solver (requires Bearer auth):**
```bash
POST /api/solve
Body: { game, method, lr, tau, steps, seed }
→ SolveResponse
```

**QRE path (requires Bearer auth):**
```bash
POST /api/qre_path
Body: { game, lambda_min, lambda_max, n_points }
→ QREPathResponse
```

**Auction (requires Bearer auth):**
```bash
POST /api/auction
Body: { bids, agent_distributions, auction_type, vocab_size, seed }
→ AuctionResponse
```

**Job submission (requires Bearer auth):**
```bash
POST /api/jobs
Body: { type, params }
→ { job_id }
```

**Job status (requires Bearer auth):**
```bash
GET /api/jobs/{job_id}
→ { job_id, status, result }
```

---

## Deployment to Vercel

### Step 1: Connect Repository

```bash
npm i -g vercel

# From repo root, authenticate and link
vercel link
```

### Step 2: Configure Root Directory

In Vercel project settings:
- **Root Directory:** `apps/web`
- **Build Command:** `npm run build`
- **Output Directory:** `.next`
- **Install Command:** `npm install`

### Step 3: Set Environment Variables

In Vercel dashboard (Project → Settings → Environment Variables):

```
GATEWAY_URL=https://your-backend-api.com
GATEWAY_SECRET=<bearer token>
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
```

### Step 4: Deploy

```bash
# Automatic deployments via GitHub
git push

# Or manual
vercel --prod
```

---

## Replay Mode (Offline Demo)

When `GATEWAY_URL` is unset, the frontend enters **replay mode**:

- API routes serve canned demo responses (realistic synthetic data)
- All responses include `replay: true` flag
- Health check shows "offline" status in header
- Each page remains functional with demo data

**Demo data sources:**
- `lib/replay-data.ts` — Pre-generated solve trajectories, QRE paths, auction results
- Realistic convergence curves (log-linear decay)
- Consistent with backend response shapes

---

## Monitoring

### Pages Status

Check each page at:
- `/` — Overview
- `/lab` — Equilibrium Lab (solver runs)
- `/qre` — QRE Explorer
- `/auction` — Auction Playground
- `/studio` — Training Studio
- `/findings` — Research Findings

### Health Dot

The navigation bar shows a health status indicator:
- **Green dot + "Live"** — Backend is reachable (version and GPU status in tooltip)
- **Red dot + "Demo"** — Backend unreachable, using replay demo data

Health check refreshes every 30 seconds.

### Error Handling

- **Missing env vars:** Auth softly disables, pages remain readable
- **Gateway unreachable:** Requests fall back to replay mode, badge shows "Demo"
- **Invalid response:** Error message displayed in UI; user can retry

---

## Building & Testing

### Type Checking

```bash
npm run type-check
```

### Smoke Test

Verifies each page returns HTTP 200:

```bash
npm run build
npm run check-pages
```

### Linting

```bash
npm run lint
```

---

## File Structure

```
apps/web/
├── app/
│   ├── layout.tsx          # Root layout with nav, footer, theme
│   ├── page.tsx            # Home page (/)
│   ├── api/
│   │   ├── health/route.ts # GET /api/health
│   │   └── proxy/[...path]/route.ts  # POST/GET proxy to gateway
│   ├── lab/
│   │   ├── page.tsx
│   │   └── components/EquilibriumLab.tsx
│   ├── qre/page.tsx
│   ├── auction/page.tsx
│   ├── studio/page.tsx
│   ├── findings/page.tsx
│   └── components/
│       ├── NavLinks.tsx
│       ├── ThemeToggle.tsx
│       └── HealthDot.tsx
├── lib/
│   ├── config.ts           # Env loader
│   ├── replay-data.ts      # Demo responses
│   └── findings.ts         # Research findings data
├── styles/
│   └── globals.css         # Dark scientific aesthetic
├── public/                 # Static assets (currently empty)
├── scripts/
│   └── check-pages.mjs     # Smoke test script
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.js
├── next.config.js
├── middleware.ts           # Supabase SSR auth
├── .env.example
└── DEPLOY.md              # This file
```

---

## Performance & Architecture

### Performance Targets

- **Page load:** < 2s (cached)
- **API call:** < 100ms (gateway latency dependent)
- **NashConv chart:** Renders 500 points in < 200ms (SVG, no heavy charting lib)
- **Lighthouse:** Target 90+ across all categories

### API Forwarding

API routes at `/api/proxy/*` forward to `GATEWAY_URL` with:
- Server-side Bearer auth (GATEWAY_SECRET never sent to browser)
- Request/response streaming
- Error handling with fallback to replay

### Replay Mode

Canned responses are generated on-the-fly:
- Realistic decay curves (log-linear convergence)
- Smooth QRE homotopy paths
- Truthfulness-validated auction results
- All flagged `replay: true` for transparency

---

## Troubleshooting

### Pages return 404

Check that all routes are defined in `app/` (not in deleted `pages/` dir).

### Build fails with TypeScript errors

```bash
npm run type-check
# Fix errors shown, then rebuild
npm run build
```

### API calls failing with "Gateway unreachable"

- Check `GATEWAY_URL` and `GATEWAY_SECRET` env vars
- Verify backend is running: `curl -i http://backend:8000/health`
- In Vercel dashboard, check that env vars are set and redeployed

### Theme not persisting

Clear browser storage:
```javascript
localStorage.removeItem('eqlm-theme')
```

Then reload and choose a theme.

### Auth pages gated when shouldn't be

Ensure Supabase env vars are set if you want auth enabled. If unset, auth soft-disables and all pages are readable.

---

## References

- **Next.js Docs:** https://nextjs.org/docs
- **Supabase SSR:** https://supabase.com/docs/guides/auth/auth-helpers/nextjs
- **TailwindCSS:** https://tailwindcss.com/docs
- **Backend API:** `../../app/server.py` (reference FastAPI implementation)
- **Findings Data:** `../../research/memory/findings.md` (source for F1–F8)

---

## Support & Contact

- **Issues:** https://github.com/sharaths/game-llm/issues
- **Email:** sharath.ai.colab@gmail.com
- **Tarka Verification:** Results in `/findings` are independently recomputed and verified before sign-off

---

**Last updated:** 2026-08-21  
**Version:** 0.1.0  
**Node requirement:** 18+
