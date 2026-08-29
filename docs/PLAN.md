# Kinetic AI — Living Plan

Revised 2026-08-29 (post-v1). This document records the objectives in force,
the current phase, machine allocation, and open decisions. It is expected to
change as measurements come in.

## Where the programme stands

v1 shipped 2026-08-29: the empirical record F1–F54 is complete, the paper,
site, app and three Hugging Face artifacts carry it, and the tree is tagged
`v1.0.0-kinetic`. The validated core: at equal compute the weight-tied
fixed-point block reaches 0.958 ± 0.017 of an explicit transformer's quality
with 2.70× fewer resident parameters (F45/F50), the saving is real on-device
(F48), and it exports exactly (safetensors) and with tying preserved (ONNX,
F52). The council line closed as a side result (F41 conditional win; F54
capacity audit). The shortcuts to scale closed honestly: conversion fails at
any gentleness (F51), cheap distillation fails its pilot (F53).

## Current phase — SPEC 0022: the twin at 1B

The prabhasa-samskrutam precedent on this hardware (1.13B params, 5.25B
tokens, ~71h on the RTX 5090) showed the barrier was never parameter count
but token count toward Qwen-equivalence. The operator directed the
architecture claim be taken to 1B scale with utility as a first-class
deliverable. Scope locked by operator answers (2026-08-29):

1. **Twin phase:** tied EqLM vs explicit twin, compute-matched at d=2048/16
   deep, identical FineWeb-Edu data, both arms to 2.5B tokens. Kill gate at
   1B tokens (ppl ratio ≤ 1.20), success at 2.5B (ladder ratio ≥ 0.95 or ppl
   ratio ≤ 1.10).
2. **Extension phase:** tied arm alone continues to 10B tokens for utility.
3. **Utility phase:** SFT → HF (base + instruct) → app serving with the
   anytime depth dial → harness demo.
4. **Independence:** no prabhasa integration; their runbook discipline is
   adopted as generic practice; their agent may later leverage our artifacts.

Gates, geometry, data, eval ladder and budget cap (≤ 24 5090-days, floors
2B/6B) are pre-registered in `research/specs/0022-twin-at-1b.md`. The EFE
ranking that selected this shape over a single-arm utility run and a full 5B
twin is `research/cycles/cycle34_candidates.json`; the TRIZ move that
reconciled the top candidates is segmentation — unequal arm lengths with a
milestone ladder, so science closes in week one and utility keeps growing.

## Objectives in force

| # | Objective | Standing |
|---|---|---|
| O1 | Single-model equilibrium paradigm | F45/F50 exchange rate at 46–121M; SPEC 0022 tests it at 1B |
| O2 | Kinetic core retained (MMD, QRE, implicit depth, auctions) | Implicit depth carries the flagship; other strands recorded F1–F54 |
| O3 | Honest ladder against real baselines | Ladder rungs (Pythia, SmolLM2, TinyLlama) measured on our harness at stated budgets; no frontier-win claim |
| O6 | Ship: HF + API + app | v1 shipped; SPEC 0022 Phase 3 ships the 1B artifacts |
| O7 | EFE autoresearch + TRIZ for inventive steps | Active — cycle 34 selected this run |
| O8 | Both machines used, never contending | 5090 trains; GB10 packs, evals, serves |
| O9 | Paper/site/app updated in the same cycle as each finding | Binding (ARTIFACTS layer) |

## Machine allocation

RTX 5090 (32GB): preflight, twin arms, extension, SFT — sequential, PID-
guarded, checkpoint every 500M tokens. GB10: data packing (CPU), milestone
evals on fetched checkpoints, app serving. Never two training jobs at once.

## Decision points ahead

The 1B-token kill gate is the first: failure records a NULL at scale and
returns the extension decision to the operator. The preflight tok/s pins the
real wall-clock and may shrink token budgets within pre-registered floors.
After the run: whether the anytime dial's retention at 1B justifies the
low-memory serving claim in the card, and whether the SFT model is good
enough to front the app by default or ships clearly labelled as a research
demo.
