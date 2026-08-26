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
- Arms (each on both base models, seed 42 + 2 more if time):
  P1 DPO (standard loss, β=0.1, AdamW, reference = frozen base);
  P2 MPO = identical DPO loss but optimizer MagneticAdamW with FIXED reference
  = the frozen base weights (the magnet's theoretical home, per ADR 0003),
  τ ∈ {1e-3, 1e-2} chosen by short sweep.
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
