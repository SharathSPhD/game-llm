# SPEC 0023 — C1: the byte-level cell of the tying grid

Status: REGISTERED 2026-08-31, before any byte-level training step has run.
Queue position: after SPEC 0022's extension phase on the 5090 (operator
decision, 2026-08-31). Origin: `docs/kineticaiprabhasaarchitectureanalysis.md`
§8, adopted as a kinetic experiment in the kinetic harness; the SPEC 0022
independence lock stands.

## The gap this cell closes

Every ratio in F1–F54 is measured at a BPE vocabulary, where the tied
embedding/head is a fixed term present in both arms. The headline is
therefore (16B + E)/(B + E) — at SPEC 0022's geometry 5.8×, at 121M only
2.70×, and in the limit E → 0 the depth itself, 16×. No run has varied the
vocabulary axis, so a reviewer may attribute part of the measured advantage
to embedding sharing rather than tying. This cell holds architecture, recipe
and content fixed and moves only the tokenizer to bytes, where there is no
embedding to hide behind. It is the cell where the claim is hardest to make,
which is what makes a positive result decisive and a negative one worth
having before the ratio appears in print.

## Arms and geometry

Shared: vocab 256 (raw bytes; document separator byte 0x00, stripped from
text so it is unambiguous), seq_len 2048 bytes, d_model 1536, n_heads 16,
d_ff 6144, embedding/head weight-tied, dropout 0.0, SDPA on, bf16 autocast,
per-block gradient checkpointing, AdamW (0.9/0.95, wd 0.1 on matrices, clip
1.0), WSD schedule in byte space (linear warmup 250M bytes to 3e-4, then
constant; no decay inside this cell), data in fixed 4×2048-byte units from
one shuffled order (seed 42), ~1.05M bytes per optimizer step.

- **Arm E:** ExplicitLM, 16 layers ≈ 453M parameters (16 × 28.3M + 0.4M).
- **Arm T:** EqLM, one block iterated 16 times, anytime supervision at
  [6, 11, 16] with weights [0.15, 0.3, 1.0] ≈ 29M resident parameters.
  Predicted resident ratio ≈ 15.8× — arithmetic, not the experiment. The
  quality ratio is the experiment. The anytime-head overhead that costs
  SPEC 0022's tied arm ~13% is ~0.1% here (the head is 0.4M), which the
  analysis identified as a structural advantage of this regime.

## Data

The byte pack is produced by decoding the SPEC 0022 pack's GPT-2 shards
back to bytes — GPT-2 BPE is byte-reversible, so content is identical to
the running twin's corpus by construction rather than by re-download.
Documents are split at GPT-2 EOS ids before decoding and rejoined with the
0x00 separator; NUL bytes occurring in source text are stripped and counted
in the manifest. Budget: 5.0B train bytes + 20M holdout bytes, packed and
manifested exactly as SPEC 0022's pack (sha256 per shard, pack hash bound
into every checkpoint).

## Schedule

Preflight (30 measured steps per arm, save/resume round trip, GO at
≥ 12k bytes/s — scaled from SPEC 0022's rule by the arms' smaller size;
extrapolated throughput is not accepted). Then Arm E to 5B bytes, then
Arm T, sequentially, milestones at 1B / 2.5B / 5B bytes. At ~40k bytes/s
expected this is ~1.5 days per arm; the cell queues behind the SPEC 0022
extension and never contends with it.

## Pre-registered gates (the analysis's §8.4, unchanged)

- **Kill gate (1B bytes, both arms):** Arm T held-out bits-per-byte ≤ 1.20 ×
  Arm E bpb at identical bytes. Failure records a NULL for byte-level tying
  and closes the cell.
- **Success (5B bytes):** bpb ratio ≤ 1.10, or measured resident-parameter
  ratio ≥ 12× with bpb ratio ≤ 1.15. Either upgrades the exchange-rate claim
  along the vocabulary axis; neither, while passing the kill gate, is
  reported with the measured ratio, not rounded up.
- **Pre-registered risk (the analysis's §4):** if quality tracks parameter
  count rather than depth, this cell may read like a ~29M model rather than
  a 453M-class one. That outcome is a finding about the F45/F50 claim, and
  the record will say so rather than reframe it.

## Readouts

Held-out bits-per-byte (the byte-level analogue of the twin's perplexity
trail); the resident ratio measured from state dicts, not computed; the
anytime retention curve at iterations 6/11/16; and the exp40 ladder scored
through a byte adapter for continuity, reported without prominence — at
these budgets its role is the trajectory, not the verdict.
