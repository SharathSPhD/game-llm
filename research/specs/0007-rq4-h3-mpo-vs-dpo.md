# SPEC 0007 — RQ-4 (H3): Magnetic Preference Optimization vs DPO

Status: ACTIVE (closure program) · GPU: 5090 (after exp10 seed42/43/44 queue drains)
— reassigned off GB10 (2026-08-26): GB10 has a confirmed hardware cooling
defect (12C die-vs-platform sensor gap, BMC throttle to ~60W, hard trip ~91C;
see cycle 16b/16c journal + results/thermal_governor.log,
results/thermal_verify.log) and is excluded from ALL GPU workload per operator
directive until RMA/repair. GB10 remains available for serving, evals, and
CPU-bound work only.

## Insight that makes H3 honest at our scale

BLiMP minimal pairs ARE preference pairs: (sentence_good ≻ sentence_bad). No
reward model, no synthetic judge — linguistic acceptability is the preference.

## Design (pre-registered)

- Base: the trained 110M checkpoints from exp10 (EqLM postln + explicit baseline).
- Train split: 60% of BLiMP phenomena (stratified); held-out: remaining 40%.
- Arms (each on both base models, seeds 42/43/44, each seed fine-tuning its
  own exp10 checkpoints):
  P1 DPO (standard loss, β=0.1, AdamW, reference = frozen base);
  P2 MPO = identical DPO loss but optimizer MagneticAdamW with FIXED reference
  = the frozen base weights (the magnet's theoretical home, per ADR 0003),
  τ ∈ {1e-3, 1e-2} both run (P2a/P2b), both reported.
- **Amendment (2026-08-27, controlled comparison):** P1 is implemented as
  MagneticAdamW with τ=0 — mathematically identical to decoupled AdamW but
  the SAME code path as P2, so the arms differ only in magnet strength. A
  CPU smoke test showed torch.optim.AdamW vs MagneticAdamW implementation
  deltas dominate at small drift, which would confound the KL comparison.
- **Amendment (2026-08-27, split realization):** the shipped pairs file
  (data/blimp_subset.json, 1000 pairs) spans 5 phenomena UIDs; train UIDs =
  {adjunct_island, anaphor_gender_agreement, animate_subject_passive} (600
  pairs), held-out = {anaphor_number_agreement, animate_subject_trans} (400
  pairs) — a phenomenon-level split (unseen phenomena at eval), stricter
  than a within-phenomenon split. KL drift measured on held-out good
  sentences.
- Matched budgets: same pairs, steps, lr per base model.
- Metrics: held-out BLiMP accuracy (win-rate proxy); KL-to-reference drift
  (mean per-token KL on a held-out text sample); catastrophic-drift check
  (train-domain accuracy).
- **H3 scoring:** MPO ≥ DPO on held-out accuracy AND lower KL drift ⇒ MET;
  either half alone ⇒ PARTIAL; neither ⇒ MISSED. Honest either way.

## Runtime

~1-2h per arm on the 5090 (fine-tuning is short); no thermal governor needed
off GB10. Queues behind the exp10 seed42/43/44 H1-at-scale runs on the same GPU
(never two GPU jobs at once).

## Rider (pre-registered 2026-08-27, BEFORE seeds 43/44 or any rider run):
## tau dose-response

Seed-42 interim shows the magnet's total displacement scales ~ lr*tau*T
(~1e-4 relative at tau=1e-2, lr=1e-5, T~675 steps) — the pre-registered tau
range cannot bind at this budget, so any MET at tau<=1e-2 is letter-only.
Rider exp11b: identical protocol, arms tau in {0.1, 1.0, 10.0} (P3a/P3b/P3c)
on both bases, same 3 seeds. Prediction (falsifiable): KL drift decreases
monotonically in tau with visible magnitude by tau=1.0; held-out accuracy is
maintained (magnet preserves generalization) until some tau* where training
is suppressed entirely (train acc stops improving). H3 verdict will be
scored on the pre-registered arms; the rider characterizes the mechanism's
dose-response and informs the paper's honest interpretation.
