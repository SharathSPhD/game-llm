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
