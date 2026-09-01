# SPEC 0024 — Interventions before the NULL: why the tied arm lost at 1B

Status: REGISTERED 2026-09-01, after the SPEC 0022 kill gate failed (tied
785.4 vs bar 604.2 at 1B tokens, ratio 1.560 and widening) and before any
diagnostic has run. Operator decisions (2026-09-01): Phase 1 completes to
2.5B as registered; two interventions run before C1; the utility path is
decided after the diagnostics.

## Why interventions precede the finding

The closure contract forbids declaring failure on attempt one and forbids a
NULL without at least two documented interventions. The gate's automatic
effect — the extension does not launch — stands. What is not yet earned is
the sentence "compute-matched tying fails at 1B on web data"; these two
arms are the cheapest tests of the two mechanisms most likely to have
produced the gap without the claim being false.

## The interventions

Both are 0.5B-token tied arms, identical to SPEC 0022's Arm T in every
respect except the named change, on the same pack, same stream order, same
schedule. Comparison bars are fixed now: Arm T's own 0.5B perplexity was
1961.6 and Arm E's was 1271.4, so the trajectory ratio at 0.5B was 1.543.

- **I1 — block learning rate scaled by 1/4.** The tied block accumulates
  sixteen gradient contributions per token where an explicit layer takes
  one; the shared 3e-4 may be effectively too hot for the block at d=2048,
  a regime F24 never saw (its record stops at d=1704 with a much smaller
  batch). The block's parameter group trains at 7.5e-5; embedding, head and
  norms stay at 3e-4. The 1/4 factor is 1/sqrt(16), the standard variance
  scaling for summed gradients, chosen over 1/16 to avoid overshooting into
  undertraining in a 0.5B-token window.
- **I2 — final-depth supervision only.** The anytime weights [0.15, 0.3,
  1.0] at depths [6, 11, 16] were tuned at depth 12 and 46–121M scale. If
  intermediate-depth supervision is dragging final-depth quality at this
  width, an arm supervised only at depth 16 reveals it. This deliberately
  risks the F24 lesson (anytime supervision is what made tying trainable)
  — if I2 diverges or collapses, that is itself informative and closes the
  supervision diagnosis in the other direction.

## Pre-registered readings (at 0.5B tokens, held-out ppl)

- **Rescue:** intervention ppl ≤ 1.25 × Arm E's 0.5B value (≤ 1589) — a
  material close from 1.543 — sends that intervention's arm to 1B tokens
  for a formal re-run of the kill gate (≤ 604.2 at 1B). Passing there
  reopens the extension with the fix, per the operator's utility decision.
- **No rescue:** both interventions ≥ 1.40 × Arm E (≥ 1780) records the
  NULL as earned: two mechanisms tested, neither responsible, and the
  finding stands as a capacity result at this scale pending C1's
  vocabulary-axis evidence.
- **Between:** reported as measured; whether to extend an intermediate arm
  to 1B returns to the operator with the numbers.

## Cost and placement

Each arm is ~9h at the measured 15.6k tok/s; both fit in under a day and
run on the 5090 immediately after Phase 1's stage 2 exits, before C1
(operator-fixed order). C1's own registration (SPEC 0023) is unchanged and
its capacity-axis reading is complementary to whatever these arms find.
