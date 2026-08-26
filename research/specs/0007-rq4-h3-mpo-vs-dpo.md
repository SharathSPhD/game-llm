# SPEC 0007 — RQ-4 (H3): Magnetic Preference Optimization vs DPO

Status: ACTIVE · closure program · pre-registered before any run

## The honest scale-appropriate design

H3 as registered: "MMD-regularized preference optimization matches or exceeds DPO
win-rate on a held-out preference set with lower reward-hacking drift
(KL-to-reference)." At our 110M/BabyLM scale, human-preference chat data is
meaningless — but **BLiMP minimal pairs are bona fide preference pairs**
(sentence_good ≻ sentence_bad, linguist-annotated). This gives a real preference
task our models can genuinely learn, with a held-out split for honest win-rate.

## Arms (all start from the SAME exp10 EqLM-v3 checkpoint = reference π_ref)

- B0: reference model (no tuning) — floor.
- B1: DPO (standard loss, β sweep {0.1, 0.5} in smoke → fix one) on train pairs.
- B2: MPO = identical DPO loss, optimizer swapped to MagneticAdamW with FIXED
  reference = π_ref weights (its theoretical home; τ sweep {1e-3, 1e-2} smoke).
- Matched steps/batch/lr across B1/B2; 3 seeds for the headline pair.

## Data

BLiMP (nyu-mll) split by phenomenon: train = 40 phenomena, held-out = 15 unseen
phenomena (generalization, not memorization), ~100 pairs each. Fixed split file
committed.

## Metrics (pre-registered)

- Win-rate: held-out pairs where log p(good) > log p(bad).
- Drift: KL(π_tuned ‖ π_ref) estimated on a fixed 1M-token corpus sample
  (mean per-token KL), plus BabyLM val loss delta (capability retention).
- **H3 met** iff B2 win-rate ≥ B1 − 1pp AND B2 KL-drift < B1 KL-drift
  (paired over 3 seeds, bootstrap CI).

## Compute

Runs on either box (small: 110M model, ~4k pref pairs, <1h/arm/seed). Smoke
(1 seed, 500 steps) gates the full 3-seed run.
