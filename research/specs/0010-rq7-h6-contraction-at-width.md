# SPEC 0010 — RQ-7 (H6): contraction that survives width (attacks F20)

Status: ACTIVE · GPU: 5090 only · Pre-registered 2026-08-27 · ADR 0005.

## Hypothesis

H6: at 121M params / 10k matched steps, at least one TRIZ-derived
intervention closes >=50% of the F20 quality gap (A3 0.5377 vs A1 0.6833
BLiMP => gap 0.1456; target >=0.610) while achieving a nonzero certified
convergence rate within the same 12-iteration budget.

## Arms (exp13, seed 42 first; seeds 43/44 for the winning arm)

- B0 control: exp10 A3 config rerun NOT needed — reuse exp10 seed-42 A3
  result (0.537) as the pre-registered control.
- B1 anytime equilibrium: CE loss also on iterates z4 and z8
  (L = CE(z12) + 0.3 CE(z8) + 0.15 CE(z4)), same map as A3.
- B2 trajectory-local contraction: A3 + penalty lambda_c * mean_k
  max(0, L_hat_k - gamma) with L_hat_k a finite-difference Lipschitz
  estimate at visited iterate z_k (one Hutchinson probe per step),
  lambda_c=0.1, gamma=0.9.
- B3 bottleneck-core: solve in d_core=256 (z -> W_down -> core solve ->
  W_up), encoder/decoder width chosen to param-match ~121M.
- Scoring identical to exp10: BLiMP 1000-pair subset, loss curve, solver
  telemetry (convergence rate at rel-tol, mean iters), peak memory, time.

## H6 scoring

MET: any arm reaches BLiMP >= 0.610 (>=50% gap closure) at <=12 iters with
certified convergence rate > 0.5 on 3 seeds. PARTIAL: >=25% gap closure OR
certified convergence without quality loss vs control. MISSED: neither.
A well-documented null on all arms is a valid closure (informs the paper's
open-problem section with mechanisms, not speculation).

## Runtime

~1.6h per arm per seed on the 5090 (batch 32 x 10k, from exp10 timing);
seed-42 screen of B1/B2/B3 (~5h), then 2 more seeds of the best arm (~3.2h).

## Amendment (2026-08-27, pre-registered AFTER seed-42 screen, BEFORE any B4 run)

Seed-42 screen: B1 0.662 BLiMP / conv 0.0 (86% gap closure, no certification);
B2 0.577 / conv 1.00 at 4.0 iters (first certified 121M equilibrium, quality
cost); B3 0.642 / conv 0.20 / fastest. No arm meets both MET halves.

- **B4 (combination arm, prediction pre-registered):** anytime supervision
  (B1) + trajectory-local Lipschitz penalty (B2, lambda_c=0.1, gamma=0.9,
  probe at the final unrolled iterate). Prediction: BLiMP >= 0.610 AND
  eval-time certified convergence rate >= 0.5 — i.e., B4 MEETS the full H6
  gate at seed 42. If met at seed 42, B4 runs seeds 43/44 for the verdict;
  else B1 (best quality arm) runs seeds 43/44 and H6 scores on the
  PARTIAL criteria.
- **B1 budget-sweep rider:** the anytime checkpoint evaluated at solver
  budgets {4, 8, 12}. Prediction: BLiMP degrades gracefully (>= 0.60 at
  budget 8, >= 0.55 at budget 4), demonstrating the anytime property that
  motivated P11. Eval-only.
- Screen observation recorded as a finding regardless: unrolled anytime
  training is FASTER than IFT solver training at this scale (44 vs 92 min)
  — the solve, not backprop, dominates EqLM's training cost.
