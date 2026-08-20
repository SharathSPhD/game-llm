# SPEC 0003 — RQ-2/RQ-6 Tier A experiments (CPU, H2 + supporting claims)

Status: ACTIVE · Phase 2 Tier A · Pre-registered per CLAUDE.md H2

## Design

Config-driven experiments. Layout per experiment:
- `configs/expNN_<slug>.yaml` — every varying quantity (games, lrs, taus, seeds, steps).
- `experiments/expNN_<slug>.py` — loads config, runs, writes
  `results/expNN_<slug>/results.json` (raw per-seed data + resolved config + config
  hash + git commit) and `results/expNN_<slug>/fig_*.pdf` (Okabe-Ito, vector).
- `experiments/run_all.py` — thin orchestrator over the expNN modules.
- `results/` is committed (frozen artifacts). ≥10 seeds where stochastic; mean ± 95%
  bootstrap CI; Holm–Bonferroni across game×method comparisons.

## Experiments

- **exp01_mmd_vs_gda** (H2 core): matching pennies, RPS, biased RPS. Methods:
  simultaneous GDA (no magnet), MMD fixed magnet, MMD + RND resets. Metrics:
  NashConv trajectory (last-iterate), distance-to-QRE(τ) trajectory, empirical
  linear-rate fit (estimate_convergence_rate). Pre-registered: GDA NashConv stays
  bounded away from 0 (cycles); MMD(fixed) → QRE(τ) linearly; MMD(RND) → Nash.
- **exp02_qre_homotopy**: QRE path λ∈[0.01,100] on RPS + matching pennies;
  monotone exploitability; warm-start vs cold-start iteration counts.
- **exp03_deq_solvers** (RQ-6 Tier A part): Anderson vs Broyden vs Picard on
  contraction maps of increasing dim (32→512): iterations-to-tol, wall time,
  residual curves; plus activation-memory proxy: DEQ implicit block vs explicit
  N-layer stack peak memory (torch profiler, CPU) at N=4..32 — pre-register:
  DEQ peak activation memory ~flat in N, explicit stack ~linear.
- **exp04_auction_truthfulness**: second-price + weighted aggregation, 3–5 agents,
  random valuations (10 seeds × 200 auctions): empirical regret of truthful bidding
  vs best misreport grid. Pre-registered: second-price regret ≤ 0 + ε (truthful);
  report weighted-aggregation deviation honestly as measured.

## Closure

Findings → Tarka verification → findings.md (SIGN-OFF PENDING) → paper/site data
feeds. GPU untouched (Tier A is CPU).
