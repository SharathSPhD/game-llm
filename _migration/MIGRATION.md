# GB10 → 5090 Migration — restart guide

**Date:** 2026-09-01
**Reason:** GB10 (DGX Spark, spark-5208, S/N 1984025003541) going for RMA
(confirmed hardware cooling defect). All work relocated to the RTX 5090
workstation (`ss@192.168.0.204`, ss-Fusion-75).

---

## TL;DR — how to resume work on the 5090

```bash
ssh ss@192.168.0.204
export PATH="$HOME/.local/bin:$PATH"        # claude lives in ~/.local/bin
export CLAUDE_CONFIG_DIR=/home/ss/.claude-gb10   # <-- the migrated GB10 sessions
cd /home/ss/projects/<project>
claude          # first run only: /login (browser OAuth) — see "Login" below
claude --resume # lists your migrated sessions for this project
```

## What moved where

| Item | GB10 (old) | 5090 (new) |
|------|-----------|------------|
| Projects | `/home/sharaths/projects/` | `/home/ss/projects/` |
| Claude config+sessions | `/home/sharaths/.claude/` | `/home/ss/.claude-gb10/` |
| `.claude.json` | `/home/sharaths/.claude.json` | `/home/ss/.claude-gb10/.claude.json` |
| Full `.claude` backup | — | `/home/ss/gb10-rma-backup/dotclaude/` (byte copy) |
| Dormant project archive | — | `/home/ss/gb10-archive/` (25 projects, pre-existing) |

**Important:** the 5090's own user `ss` has its *own* separate Claude setup at
`/home/ss/.claude/` — it was NOT touched. Always export
`CLAUDE_CONFIG_DIR=/home/ss/.claude-gb10` to reach the migrated GB10 sessions.

## Login (one-time, unavoidable)

OAuth credentials do NOT transfer between machines (refresh tokens are
device-bound; the copied token reads as expired). On first `claude` run on the
5090 you must `/login` once via browser. After that, all migrated sessions,
memory, and history are available. Everything else migrated as files.

## Session continuity — how it was done

Claude keys sessions by a path-encoded directory name. Because projects moved
from `/home/sharaths` to `/home/ss`, every session dir was renamed
(`-home-sharaths-projects-X` → `-home-ss-projects-X`) and every embedded
absolute path inside the session `.jsonl` (the `cwd` field and all references)
was rewritten `/home/sharaths → /home/ss`. 30 project dirs, 1534 files,
~141k lines. Originals on GB10 are untouched (fully reversible).

## What was intentionally LEFT BEHIND (not migrated)

Per operator decisions, to keep the 5090 healthy (700G+ free floor):

- `PSALM-integration/data/checkpoints/` (384G) — BabyLM, done. `submission/`
  (leaderboard entry) WAS kept.
- `prabhasa-samskrutam/data/checkpoints/` (135G) — the scientifically
  load-bearing m2/m3 arms were PUBLISHED to HF instead (see below).
- `gptbert_gb10_run/checkpoints/` (114G) — old gptbert run.
- `pramana/hf_upload_full`, `hf_upload`, `tmp_gguf` (~49G) — HF staging copies.
- All `.venv/`, `node_modules/`, `__pycache__/`, `.mypy_cache/` — regenerable.

## HuggingFace-published weights (prabhasa HORD)

Real trained checkpoints published PUBLIC under pseudonym p-s (account qbz506):
- `qbz506/p-s-hord-m2` — 199M params, 650M tokens (baseline + treatment)
- `qbz506/p-s-hord-m3` — 353M params, 275M tokens (baseline + treatment + aux)

Each `.pt` self-describes (embeds config/opt/step/tokens_seen) and is resumable
with the prabhasa codebase. m4_preflight was an untrained probe — NOT published.

## Code safety (pushed to GitHub before migration)

All in-progress side branches with unpushed commits were pushed (15 branches
across game-llm, prabhasa-samskrutam, prabodha, PWM, PSALM). Preserved specially:
- ActiveCIrcuitDiscovery: local `talk-site` had diverged from remote → pushed as
  `talk-site-gb10-local` (remote `talk-site` has different work; do NOT
  force-merge without reviewing both).
- PSALM stash → branch `stash-gb10-preserve`.
- prabodha stash → patch at `_MIGRATION/stashes/prabodha-stash0.patch`
  (had a 110MB node_modules blob; apply the patch, don't expect a branch).

## Per-project restart pointers

See each project's own `RESTART.md`. Active projects:

- **game-llm** (kinetic-ai) — the live one. Already running on 5090 as
  `fusion-project/kinetic-twin` (exp39 Phase 1). Research state in
  `research/memory/state.json` (cycle 34, SPEC 0022). Remote: main.
- **prabhasa-samskrutam** — Sanskrit LM / HORD. branch `h-ord/phase1`.
  Weights on HF (above). 22G `data/packed` corpus migrated.
- **pranava** — branch `master`. 32G data migrated.
- **pramana** — branch `main`. Models migrated; HF staging left behind.
