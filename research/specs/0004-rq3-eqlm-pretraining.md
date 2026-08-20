# SPEC 0004 — RQ-3 (H1): EqLM-small pretraining on BabyLM strict-small

Status: DRAFT→ACTIVE after Tier A closure · Phase 2 Tier B · GPU: GB10 (single job)

## Assets on this machine (verified 2026-08-20)

- Data: HF cache `datasets--BabyLM-community--BabyLM-2026-Strict-Small` (10M words);
  assembly reference: `PSALM/scripts/assemble_strict_small.py`.
- Eval: `PSALM-integration/vendor/babylm-evaluation-pipeline-2026` (BLiMP + suite);
  operational lessons in `PSALM/research/cycles/run.md` (eval gating, subprocess reaping).
- Baselines: HF cache `models--BabyLM-community--babylm-baseline-100m-gpt-bert-masked-focus`
  and gptbert training infra `gptbert_gb10_run/` (dataset.py, LAMB, GB10-tuned).
- Tokenizer: `PSALM/scripts/build_babylm_joint_tokenizer.py` pattern, or GPT-2 BPE.

## EqLM-small design (pre-registered)

- Architecture: single weight-tied pre-LN transformer block f(z, x) = z +
  Attn(LN(z)) + MLP(LN(z)) with input injection x (token+pos embeddings added each
  iteration); solved to fixed point with Anderson (max 12 iters, tol 1e-3 training /
  1e-4 eval); backward: JFB (Jacobian-free) for training throughput, IFT available
  for ablation. Optional spectral norm for contraction safety.
- Parameter matching: widen the DEQ block (d_model, d_ff) so total params ≈ the
  explicit L-layer baseline (GPT-2-small-class scaled to strict-small budget:
  target ~30–60M params, decided by matching gptbert baseline config).
- Arms (each 1 seed smoke → 3 seeds full):
  A1 explicit GPT-2-class AdamW (baseline);
  A2 EqLM AdamW;
  A3 EqLM MMD (Euclidean Bregman, magnetic anchor = EMA(0.999) of weights, τ swept
  {1e-3, 1e-2} in smoke);
  A4 (stretch) explicit + MMD — isolates optimizer effect from architecture.
- Budget: matched tokens & steps across arms; strict-small 10M words, ~2–4 GB10-hours
  per arm (validate in smoke). One GPU job at a time (state.json gpu_lock).
- Metrics: train/val loss curves; BLiMP average via vendored pipeline; peak
  activation memory vs effective depth (torch profiler, CUDA) — H1's ≤50% memory
  claim measured here; solver iteration stats.

## H1 scoring

EqLM (best of A2/A3) BLiMP ≥ 95% of A1 BLiMP AND peak activation memory ≤50% of A1
⇒ H1 met. Miss ⇒ documented finding + TRIZ iteration on the architecture (e.g.
multi-block DEQ, deeper injection, pcDEQ constraints) per operator directive.

## Ordering

1. Smoke (tiny width, 30min budget): pipeline end-to-end incl. eval, all arms.
2. Full A1+A2 (1 seed) → compare → full A3 with best τ → 3 seeds for headline arms.
3. Tarka verification → findings.md → paper/site feeds → milestone push.
