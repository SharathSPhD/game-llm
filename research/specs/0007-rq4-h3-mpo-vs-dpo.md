# SPEC 0007 — RQ-4 (H3): Magnetic Preference Optimization vs DPO

Status: ACTIVE (closure program) · GPU: GB10 (thermal-governed) after exp10 seed-42

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

~1-2h per arm on GB10 (fine-tuning is short); thermal governor active.
