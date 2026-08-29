# SPEC 0022 — The twin at 1B: compute-matched tying at prabhasa scale

Status: REGISTERED 2026-08-29, before any 1B training step has run.
Operator scope decisions (AskUserQuestion, 2026-08-29): run shape = twin to
2.5B tokens then tied-arm extension to 10B; tokenizer = GPT-2 50k BPE; utility
= SFT + HF release + app serving + harness demo; this project remains
independent of prabhasa-samskrutam (no integration work here; their agent may
later leverage these artifacts).

## Why this run exists

F45/F50 established the exchange rate at 46–121M parameters on BabyLM data:
at equal compute the weight-tied block reaches 0.958 ± 0.017 of an explicit
transformer's BLiMP with 2.70× fewer parameters. The prabhasa-samskrutam
project demonstrated on this exact hardware that a 1.13B model pretrains on
5.25B tokens in ~71 hours (RTX 5090, 20.9k tok/s measured) — parameter count
was never the barrier; token count toward Qwen-equivalence was (F51, F53).
The claim this run can close is the architecture claim at 1B scale: the same
exchange rate, on general web data, at the scale a practitioner would deploy.

At d=2048 the tying arithmetic turns decisively favourable: the block
dominates the parameter budget, so the resident-memory ratio grows from 2.70×
(121M, embedding-dominated) to ~5.8× (see geometry below). If the quality
ratio holds, the headline is a ~1B-compute-class model resident in ~158M
parameters.

## Arms and geometry

Shared: vocab 50257 (GPT-2 BPE), seq_len 2048, d_model 2048, n_heads 16,
d_ff 8192, learned positions, embedding/head weight-tied (as the EqLM
codebase builds both classes), dropout 0.0 (single-epoch data), model vocab
padded to 50304 (multiple of 64) for matmul throughput with GPT-2 ids never
reaching the pad, fused scaled-dot-product attention enabled identically for
both arms (config-gated; equivalence to the explicit attention asserted by
test_sdpa_equivalence.py), bf16 autocast over fp32 weights, per-block
gradient checkpointing, AdamW (betas 0.9/0.95, weight_decay 0.1 on matrix
parameters only, grad clip 1.0), and a warmup-stable-decay schedule in token
space: linear warmup over 250M tokens to 3e-4, constant thereafter, cosine
decay to 3e-5 only over the extension's final window (8B → 10B tokens) — so
both arms see identical learning rates at identical tokens throughout the
twin phase. Data is consumed in fixed 4×2048-token units from one shuffled
order (seed 42), so arms with different micro-batch sizes read byte-identical
streams; each optimizer step is ~1.05M tokens for either arm.

- **Arm E (explicit twin):** ExplicitLM, 16 layers. ≈0.92B parameters
  (16 × 50.4M block + 107M tied embedding/head + positions).
- **Arm T (tied):** EqLM, one block iterated 16 times (compute-matched:
  1 iteration = 1 layer at identical width). Anytime-unrolled supervision at
  iterations [6, 11, 16] with weights [0.15, 0.3, 1.0] — the F24 regime that
  makes tying trainable, scaled to depth 16. Map form postln, spectral norm
  on, residual damping 0.2, anderson solver at eval. ≈158M resident
  parameters (5.8× fewer than Arm E).

Disclosed asymmetry: anytime supervision computes the 103M-parameter head at
three depths rather than one during training, ≈13% extra training FLOPs
charged to Arm T. Inference is unaffected. Compute-matching refers to block
FLOPs per token, which is what depth costs at serving.

## Data

FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, sample-100BT), tokenized to GPT-2
ids, packed once into uint16 memmap shards: 10.5B tokens train + 20M tokens
holdout (drawn from the shard tail, never trained on). Nemotron-CC-HQ was the
named preference in the deadlock decision; it has no official Hugging Face
distribution (Common Crawl S3 only), and FineWeb-Edu is the accessible
dataset of the same construction (classifier-filtered Common Crawl) already
proven in this stack by the SPEC 0021 cache. Both arms read the identical
byte-identical pack in the identical order. A pack manifest (sha256 per
shard, token count, tokenizer hash) is committed with the code; a checkpoint
must not resume onto a different pack (manifest hash stored in every
checkpoint).

## Schedule and phases

- **Phase P (preflight, hours):** 30 measured steps per arm at final geometry
  on the 5090; median post-warmup tok/s pins the wall-clock estimate.
  Save/resume round-trip exercised. Extrapolated throughput is not accepted —
  the GB10-era lesson (520 tok/s measured vs 11k assumed) is adopted as
  binding practice. GO rule: measured ≥ 5.5k tok/s on Arm E → proceed;
  < 4k tok/s → geometry re-cut (seq 1024 and/or 12 layers) and this spec
  amended before any long run starts.
- **Phase 1 (twin, both arms to 2.5B tokens):** arms run sequentially
  (one GPU), Arm E first (it is the reference the gate divides by).
  Checkpoints every 500M tokens; milestone evals at 0.5B / 1B / 2.5B.
- **Phase 2 (extension, Arm T alone 2.5B → 10B tokens):** continues from the
  Phase 1 checkpoint; milestone evals at 5B / 7.5B / 10B.
- **Phase 3 (utility):** SFT of the final Arm T checkpoint on
  instruction data (SmolTalk-class, ≈1 GPU-day), then HF release (base +
  instruct, safetensors exact + ONNX with tying preserved), app serving swap
  with the anytime depth dial, harness demo.

Budget cap, fixed now: total training wall-clock ≤ 24 days of 5090 time. If
the measured preflight predicts an overrun, token budgets shrink
proportionally with floors of 2B (twin) and 6B (extension), recorded before
launch — not renegotiated mid-run.

## Pre-registered gates

- **Kill gate (at the 1B-token milestone, both arms):** Arm T held-out
  perplexity ≤ 1.20 × Arm E held-out perplexity at identical tokens.
  Failure closes the run as a NULL at 1B scale (the finding is recorded with
  both curves; the F45 result stands at its own scale) and Phase 2 does not
  start without a new operator decision.
- **Success criterion (at 2.5B tokens):** quality ratio ≥ 0.95 on the eval
  ladder aggregate (mean of lm-eval task accuracies, Arm T / Arm E) or
  held-out perplexity ratio ≤ 1.10. Meeting either upgrades F45 to 1B scale;
  meeting neither while passing the kill gate is reported as a partial result
  with the measured ratio, not rounded up.
- **Anytime gate (extension, at 10B tokens):** Arm T at 6 iterations retains
  ≥ 85% of its 16-iteration ladder aggregate, else the anytime claim is
  scoped to small models (F24) and the card says so.
- **Abort rules (runbook):** non-finite loss aborts; loss spike > 2× trailing
  mean warns; 0% GPU utilization for > 10 minutes is a hang, not progress.

## Eval ladder (binding, the ARTIFACTS layer)

At every milestone, for every arm alive at it: held-out perplexity; lm-eval
core suite (ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, SciQ,
LAMBADA-openai; MMLU reported but expected near chance at these budgets);
BLiMP subset via the in-house harness. Context rungs measured on the same
harness, same commit: Pythia-410M and Pythia-1B (300B tokens),
SmolLM2-360M, TinyLlama-1.1B — positioned as published-budget references,
with the token-budget gap stated in every table; no claim of beating a
frontier model is made or implied. Additionally for Arm T: the anytime dial
curve (iterations 6/11/16) and the resident-memory profile (exp35 protocol).
Evals run on the GB10 against fetched checkpoints while the 5090 trains, so
the ladder never blocks the run.

## Machine allocation and safety

The 5090 (32GB, container `rtx5090-train`) carries all training via the
established remote-job scripts; jobs guarded by PID; `PYTORCH_CUDA_ALLOC_CONF
=expandable_segments:True`; ≥ 100GB free disk verified before launch
(checkpoints ≈ 4GB Arm E / 1GB Arm T each, rolling window of 3 + milestone
keeps). The GB10 packs data (CPU), runs milestone evals, and keeps serving
the app. Never two training jobs at once; the pack rsyncs once before launch.

## Relation to the record

This spec extends SPEC 0018 (compute-matched tying) to 1B scale and general
web data. It does not reopen SPEC 0020/0021: conversion (F51) and KD (F53)
stay closed; this is the from-scratch path at a budget the audit showed is
locally affordable. Independence note: prabhasa-samskrutam's measured
precedent (throughput, checkpoint cadence, preflight discipline) is adopted
as practice; no code, data, or claims are shared between the projects.
