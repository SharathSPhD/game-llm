# 3. Magnetic AdamW for parameter-space training arms

Date: 2026-08-21
Status: Accepted

## Context

Tier B smoke (exp05, iteration 2) showed arm A3 (EqLM trained with raw
MagneticMirrorDescent, Euclidean geometry, lr=3e-4) not learning (loss 128→125 over
800 steps) while A2 (same model, AdamW) learned (132→28.5). Raw Euclidean mirror
descent is unpreconditioned SGD plus a magnet; from-scratch LM training at small lr
requires adaptive preconditioning. A3-vs-A2 therefore confounded optimizer adaptivity
with magnetism, and could not test H3's actual question (does the magnetic anchor
help?). All arms also started at loss ≈125 (≫ uniform ≈10.8), indicating unscaled
weight-tied logits at init.

## Decision

1. Implement `MagneticAdamW` in kinetic_ai/optim: AdamW update composed with a
   magnetic proximal pull toward a reference (θ ← θ' − lr·τ·(θ' − θ_ref) after the
   AdamW step; reference = EMA or periodic snapshot). This is the standard practical
   form for magnetic/KL-anchored preference optimization; raw MMD stays for
   strategy-space game solving where its theory lives.
2. Fix EqLM/ExplicitLM init: embedding init std 0.02 and scaled tied-head logits so
   initial loss ≈ ln(vocab).
3. Tier B arms become: A1 explicit+AdamW, A2 EqLM+AdamW, A3 EqLM+MagneticAdamW —
   isolating architecture (A1 vs A2) and magnetism (A2 vs A3) cleanly.

## Consequences

H3's optimizer comparisons are now like-for-like in adaptivity. The smoke's A3
non-learning result is recorded as a method finding (F9), not evidence against H3.
