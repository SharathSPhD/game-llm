# 2. EqLM discovery target, hypotheses H1–H4, and pre-registered thresholds

Date: 2026-08-20
Status: Accepted

## Context

Operator intent (2026-08-20): not an interactive toy but a discovery — a new LLM
architecture from the game-theoretic research program, benchmarked against
GPT/BERT-class baselines; science first, app reflects real use. Compute: GB10 only
(RTX 5090 reserved). BabyLM-style pretraining + BLiMP-class eval infrastructure is
proven on this machine (PSALM, gptbert_* runs), making the benchmark feasible.

## Decision

Target architecture **EqLM**: weight-tied DEQ transformer block (Anderson/JFB),
optional pcDEQ constraints; MMD training with magnetic reference anchor; QRE
rationality-parameter decoding; token-auction multi-model decoding.

Pre-registered thresholds (as in CLAUDE.md):
- H1: EqLM-small ≥95% of parameter-matched GPT-2-class BLiMP average on BabyLM
  strict-small, at ≤50% peak activation memory for depth.
- H2: MMD last-iterate convergence to τ-regularized QRE where simultaneous GDA
  cycles; empirical linear rate (log-linear R²≥0.9, final 50% of trajectory).
- H3: MPO ≥ DPO win-rate with lower KL-to-reference drift.
- H4: truthful auction ensemble of 2–3 specialists > best single model on
  mixed-domain eval.

Baselines must be parameter/token/compute-matched; ≥3 md5-distinct seeds.

## Consequences

Phase 2 experiment design flows from these thresholds; any change needs a new ADR.
A miss is a documented finding that informs the next architectural iteration (per
operator: hypotheses inform invention; falsification is not the objective).
