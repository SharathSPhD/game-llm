# CLAUDE.md — Kinetic AI operating guide for AI agents

## What this project is

Kinetic AI (`kinetic_ai`) builds the transition from optimization-based LLM training to
game-theoretic equilibrium, per `res-docs/Game Theoretic LLM Architecture Guide.pdf`:
Magnetic Mirror Descent (MMD) → QRE, Deep Equilibrium Models (DEQ/pcDEQ), token auctions,
SPPO self-play. The discovery target is **EqLM** — a language model whose depth (DEQ
fixed point), training (MMD magnetic anchor), and decoding (QRE temperature + token
auctions) are all equilibrium computations, benchmarked against GPT/BERT-class baselines.

Deliverables: validated science → researcher-facing app (`apps/web` + GB10 gateway) →
paper (`paper/`) + GitHub Pages site (`site/`).

## Pre-registered hypotheses

- **H1 (EqLM parity):** A weight-tied DEQ transformer LM pretrained on BabyLM
  strict-small (10M words) reaches ≥95% of a parameter-matched explicit GPT-2-class
  baseline's BLiMP average, at ≤50% of its peak activation memory for depth.
  Status: PENDING (Phase 2 Tier B). Threshold changes require an ADR.
- **H2 (MMD convergence):** MMD achieves last-iterate convergence to the τ-regularized
  QRE on matrix games and Kuhn poker where simultaneous GDA cycles; empirical linear
  rate (log-linear fit R²≥0.9 over final 50% of trajectory). Status: ACTIVE (Phase 1c/2A).
- **H3 (MPO alignment):** MMD-regularized preference optimization matches or exceeds
  DPO win-rate on a held-out preference set with lower reward-hacking drift
  (KL-to-reference). Status: PENDING (Phase 2 Tier C).
- **H4 (auction decoding):** Truthful token-auction aggregation of 2–3 specialist models
  beats the best single model on mixed-domain eval. Status: PENDING (Phase 2 Tier C).

Hypotheses inform the next iteration/invention toward operator intent; a well-documented
null is a valid closure state, not an endpoint.

## The closure contract is binding

A phase closes only when it satisfies all six layers:

1. **TECHNICAL** — pytest green, ruff+mypy clean, coverage ≥80%.
2. **EMPIRICAL** — all experiment arms run; finding declared with interpretation.
3. **INTEGRITY** — adversarial (Tarka) review of the finding resolved; comparisons
   like-for-like (matched params/tokens/compute); ≥3 md5-distinct seeds.
4. **ARTIFACTS** — code+results pushed; paper/site sections updated from the finding.
5. **MEMORY** — `research/memory/` (state.json, journal.md, findings.md) and ADRs updated.
6. **SIGN-OFF** — operator sign-off on interpretation before merge to main.

Hard rules: never declare "failed" on attempt 1; never a NULL without ≥2 documented
interventions; never report a number you didn't produce; never fabricate a citation;
never two GPU jobs at once (GPU lock in state.json).

## How we work

- **TDD:** failing test first, then code. **Spec-driven:** specs in `research/specs/`
  are the contract; amend the spec before the code. **Config-driven:** everything that
  varies lives in `configs/*.yaml`; no magic numbers in code; resolved config hash
  recorded with every run.
- **EFE autoresearch loop:** `research/cycles/run.md` is the idempotent cycle runner;
  durable state in `research/memory/state.json`. GPU busy → do GPU-free work
  (paper, app, specs). ralph-loop drives closures; triz-engine for inventive steps
  when an RQ stalls.
- **Git:** worktrees under `.claude/worktrees/` for parallel features. Commits as
  `SharathSPhD <qbz506@york.ac.uk>`, no Co-Authored-By trailer. Milestone pushes to
  `https://github.com/SharathSPhD/game-llm.git`. Findings staged SIGN-OFF PENDING
  before merge to main.

## Artifact standard (paper + app) — binding

The paper is benchmarked against `/home/sharaths/projects/ActiveCIrcuitDiscovery/paper`
(MDPI submission: 2183 lines of prose, ZERO bullet lists, 12 figures, 12 tables,
modular `sections/*.tex`, appendix). Every paper edit obeys:

- **Third-person impersonal.** "Retention is measured against..." — never "we", never "our".
- **No `itemize`/`enumerate` in the body.** Lists become prose or tables.
- **Depth, not summary.** Results carry protocol, numbers, interpretation, and the
  ruled-out alternatives, per experiment. Results is the longest section.
- **Rich apparatus.** `booktabs` tables with `\multirow` and CIs; TikZ architecture and
  flow diagrams, not only result plots; captions state the protocol.
- **Modular source.** `paper.tex` = preamble + metadata + `\input`; prose in `sections/*.tex`,
  each opening with a comment recording the run directory and machine its numbers come from.
- **No AI giveaways or process narration.** Banned: "honestly", "we report the full arc",
  "adversarially audited", "commend this practice", "Notably,", "Importantly,",
  rhetorical em-dash asides. Findings are stated; scope limits go in Limitations as
  technical constraints.
- **Appendices** carry per-seed tables, hyperparameters and derivations.

Paper and app are living artifacts: when a finding closes, its paper section, tables,
figures, site card and app surface update in the SAME cycle — this is the ARTIFACTS
layer of the closure contract, not a later cleanup.

## Hardware & environment

- This machine is the GB10 (DGX Spark). RTX 5090 box is reserved (other training).
- Python via `uv`; venv at `.venv` (torch cu130). GPU training jobs run in Docker
  containers; serving must stay portable to RunPod serverless
  (`kinetic_ai/serve/executor.py` abstraction).
- App stack: Vercel (`apps/web`) + Supabase (dedicated project; admin
  sharath.sathish@gmail.com, admin-invited guests) + Cloudflare Worker/KV →
  cloudflared quick tunnel → GB10 FastAPI (`app/server.py`).

## Commands

```bash
.venv/bin/python -m pytest tests/ -q          # tests
.venv/bin/python -m pytest tests/ -q -m "not slow"
.venv/bin/ruff check kinetic_ai/ && .venv/bin/mypy kinetic_ai/
python simulate.py                             # demo simulation
python experiments/run_all.py                  # experiment suite
```
