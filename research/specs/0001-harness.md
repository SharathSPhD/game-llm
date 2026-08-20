# SPEC 0001 — Kinetic AI: Autonomous Research Harness

Status: ACTIVE · Owner: autonomous agent (operator: SharathSPhD) · Spec-driven

## 1. Purpose

Run the Kinetic AI research program — adversarial review → validated game-theoretic
science → EqLM architecture discovery → app → paper+site — as an auditable,
resumable autonomous loop on the GB10, under the six-layer closure contract in
CLAUDE.md.

## 2. Invariants (NEVER violated)

- No mock / synthetic results reported as real.
- Citation integrity: every citation verified at cite time.
- Six-layer closure contract (see CLAUDE.md).
- Statistical honesty: ≥3 md5-distinct seeds, mean ± 95% CI, Holm–Bonferroni for
  multiple comparisons, like-for-like compute/token budgets.
- GPU discipline: one GPU job at a time; lock in `research/memory/state.json`.
- Honest framing: pre-registered thresholds (CLAUDE.md H1–H4) change only via ADR.

## 3. Architecture

```
 ┌────────── orient ──────────┐
 │ state.json · journal · git │
 └──────────┬─────────────────┘
            v
   harvest pending results ──► evaluate + stats gates
            v
   select next action (GPU-aware; EFE: max expected info gain toward H1–H4)
            v
   act (TDD code / experiment launch / paper / app) via subagents & Workflows
            v
   Tarka adversarial review of any finding
            v
   record (journal, findings, state) + commit ──► schedule next cycle
```

## 4. Durable state

- `research/memory/state.json` — cycle no., phase, current RQ, gpu_lock, next_action.
- `research/memory/journal.md` — append-only cycle log.
- `research/memory/findings.md` — validated findings only (Tarka-reviewed).
- `docs/decisions/` — ADRs (Nygard format).

## 5. Research program (RQ backlog)

- RQ-1 (Phase 1): Does the codebase faithfully implement the research doc?
  → adversarial review; adjudicate MMD implementation vs tests vs theory; fix all.
- RQ-2 (H2): MMD last-iterate convergence vs GDA cycling — matrix games, Kuhn poker.
- RQ-3 (H1): EqLM-small vs GPT-2-class baseline on BabyLM strict-small / BLiMP.
- RQ-4 (H3): MPO (magnetic preference optimization) vs DPO.
- RQ-5 (H4): token-auction decoding of specialist ensembles.
- RQ-6: DEQ O(1) depth-memory claim — measured, not cited.

## 6. Phase map

Phase 1 harness+review+fix → Phase 2 science (Tier A CPU / Tier B GB10 pretraining /
Tier C alignment+auctions) → Phase 3 app → Phase 4 paper+site. Milestone push at each
phase closure and each validated finding.
