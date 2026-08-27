# ADR 0005 — TRIZ-derived program for H6 (contraction-at-width) and H5 scoping

Date: 2026-08-27 · Status: accepted · Driver: F20 open problem + F22 scoping gap

## TRIZ session (triz-engine)

Problem: certified equilibrium convergence at d=1704 WITHOUT more solver
iterations and WITHOUT less capacity (F20: truncation penalty 7%@d204 ->
21%@d1704 at fixed 12-iteration budget).

- Technical contradiction: Reliability(27) vs Speed(9) -> principles
  21 (Skipping), 35 (Parameter changes), 11 (Beforehand cushioning), 28.
  Accuracy(28) vs Loss of time(25) -> 24 (Intermediary), 34, 28, 32.
- Physical contradiction: the map's Lipschitz constant must be SMALL
  (certify in 12 iters) AND LARGE (capacity). Separation by CONDITION
  scored highest (0.85), separation in SPACE second.

## Ranked solution sketches (IFR-scored)

1. **B1 Anytime equilibrium** (P11, IFR 3/4): auxiliary LM losses on
   intermediate iterates z4/z8 (+final z12) — every truncated iterate is a
   usable representation; the truncation penalty degrades gracefully.
2. **B2 Trajectory-local contraction penalty** (P35 + separation-by-
   condition, IFR 3/4): penalize a finite-difference local Lipschitz
   estimate ||f(z+eps v)-f(z)||/(eps||v||) ONLY at iterates the solver
   actually visits — expressive globally, contractive along the solve path.
3. **B3 Bottleneck-core equilibrium** (P24 + separation-in-space, IFR 2/4):
   solve the fixed point in a learned d_core=256 space between a wide
   encoder/decoder; contraction is cheap to certify in the small space,
   capacity lives outside the recurrent loop; per-iteration cost shrinks.

All three are testable at 121M/10k steps in <=2h each on the RTX 5090 and
are pre-registered as exp13 arms in SPEC 0010.

## H5 metric decision (autoregressive auction, SPEC 0009)

Closed-loop generation cannot use teacher-forced perplexity of its own
output. Judge metric chosen: mean NLL/token of each system's generated
continuation under the FROZEN exp10 seed-42 124M explicit LM (trained on
the full BabyLM mix, independent of the compared 30M systems, applied
identically to all). Secondary: domain-consistency and 3-gram repetition
rate. This is the standard "perplexity under a larger independent LM"
evaluation, pre-registered before any run.
