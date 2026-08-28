# SPEC 0011 — RQ-8 (H7): KineticLM — equilibrium conversion of a real 1.7B base

Status: ACTIVE · GPU: 5090 (uptraining) + GB10 (eval) · Pre-registered 2026-08-28
Depends on: ADR 0006. Carries forward F24/B1 (anytime), F24/B3 (core topology),
F24/B2 (trajectory contraction), F19 (warm start).

## Question

Does the kinetic architecture survive contact with a real, strong base model?
Concretely: can Qwen3-1.7B be converted into the EqLMCore topology — explicit
outer layers around a **weight-tied core iterated to depth** — retaining most
of its benchmark quality at substantially fewer parameters, with a working
inference budget dial?

## Method (pre-registered)

- **Surgery:** keep the first `n_pre` and last `n_post` transformer layers
  explicit (initialized from the base); replace the middle `M` layers with ONE
  shared block, initialized by the published "average" / "stepwise" recipes
  (Relaxed Recursive Transformers, arXiv 2410.20672), iterated `K = M` times.
  Parameter saving = (M − 1) x per-layer params.
- **Uptraining objective (ours):** anytime-unrolled supervision (F24/B1) —
  cross-entropy at recursion depths {K/4, K/2, K} — PLUS token-level KL
  distillation from the frozen base model as teacher. Optional arm: +
  trajectory-local contraction penalty (F24/B2).
- **Corpus:** FineWeb-Edu sample (public, standard for this scale). Budget
  pre-registered at 0.5–2B tokens (5090-feasible); published recipes use
  10–100B, so partial retention is the honest expectation and the gates
  reflect that.
- **Baselines:** (a) the unmodified base under the identical harness; (b) a
  same-parameter-count dense model from the same family where one exists
  (Qwen3-0.6B/1.7B bracket); (c) the surgical model with NO uptraining
  (measures how much the uptraining recovers).

## Scoring (pre-registered gates)

Benchmarks via lm-evaluation-harness: MMLU (5-shot), ARC-Challenge, HellaSwag,
GSM8K (8-shot), average reported as the headline "retention ratio" vs the base
run under the same harness on the same machine.

- **MET:** retention >= 0.90 of base average at <= 60% of base parameters,
  with monotone graceful degradation across budgets {K/4, K/2, K}.
- **PARTIAL:** retention >= 0.75, or MET on the budget dial but not quality.
- **MISSED:** below 0.75 retention. A documented miss with the diagnostic arc
  is a valid closure (per CLAUDE.md) — the honest baseline here is that
  0.5–2B uptraining tokens is 5–50x below published recipes.

## Runtime

Surgery + smoke: hours. Uptraining: ~1–2 days/arm on the 5090 (bf16, grad
checkpointing, seq 2048). Eval: ~1–2 h/model on GB10 under the thermal
governor. Seeds: 1 for the screen, 3 for any headline claim.
