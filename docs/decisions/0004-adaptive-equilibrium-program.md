# 4. H1′: the adaptive-equilibrium program (TRIZ-derived reframing)

Date: 2026-08-21
Status: Accepted (operator directive: TRIZ-driven invention, beyond BabyLM)

## Context

H1 (parity at matched params) formally missed at 0.930 [0.898, 0.949] vs ≥0.95
(F18), at 2.9× wall-clock. TRIZ analysis (.triz/session.jsonl, 2026-08-21) of the
physical contradiction "depth must be many AND few iterations" yields separation
solutions that use the architecture's native properties instead of fighting the
explicit baseline on its home turf.

## Decision

The discovery program pivots to **adaptive equilibrium computation**, pre-registered
as three falsifiable hypotheses (thresholds set before runs):

- **H1′a (warm-start decoding, P10/time):** initializing each decode step's solve
  from the previous token's equilibrium reduces mean iterations-per-token by ≥50%
  at equal output quality (greedy-decode agreement ≥99% with cold-start), bringing
  decode wall-clock to ≤1.5× the explicit baseline (from 2.9×).
- **H1′b (per-token early exit, P3/condition):** per-position residual-gated exit
  reduces average solver iterations ≥40% at ≤0.5pp BLiMP cost; per-token iteration
  counts correlate with linguistic difficulty (reported descriptively).
- **H1′c (think-harder dial, P6):** eval-time solver budget scaling (train@12,
  eval@{12,24,48}) improves BLiMP monotonically at zero training cost; residual
  magnitude predicts per-token error (AUC > 0.6) — equilibrium as built-in
  uncertainty estimate.

Parity (old H1) remains a reported secondary metric. Contraction annealing (P35)
is the standing candidate if H1′ arms stall. Benchmarks: BabyLM/BLiMP retained for
comparability, plus compute-quality Pareto curves and decode-latency measurements
(beyond-benchmark artifacts per operator directive).

## Consequences

exp09 implements warm-start decoding + per-token exit + checkpoint saving (the
missing artifact for HF publication). The app's "equilibrium lens" (per-token
iteration/confidence visualization) becomes a direct rendering of H1′b/c — the
science and the product feature are the same computation.
