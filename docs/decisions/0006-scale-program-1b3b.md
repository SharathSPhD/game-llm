# ADR 0006 — Scaling the kinetic program to the 1–3B open-weight class

Date: 2026-08-28 · Status: accepted · Supersedes scope of ADR 0004/0005 (11M–121M era)

## Operator intent

Advance beyond BabyLM to the 1–3B instruction-tuned class (Qwen3 / Nemotron /
Llama-3.2), producing real LLM output, **without discarding the architecture and
methods built so far — those remain the core**. Compare against the same
open-weight baselines on public benchmarks. Leverage NVIDIA tooling. Heavy
training on the RTX 5090, parallelized with GB10. Productionize in the app.

## Feasibility ruling (why we adapt rather than pretrain)

Matching Qwen3-1.7B from scratch is out of reach: it saw ~36T tokens; a
Chinchilla-optimal 1B run (20B tokens) is ~23 GPU-days on one 5090 and would
still be far weaker. Published looped/recursive **uptraining** of a pretrained
model costs 10–100B tokens (hours–days on a 5090) and is the established
route (Relaxed Recursive Transformers, arXiv 2410.20672; Huginn-3.5B; Ouro).
Therefore: **adapt strong open-weight bases with our methods, and benchmark
against those same bases** — a like-for-like comparison our closure contract
can actually enforce.

## What carries forward (the kinetic core)

1. **EqLMCore topology (F24/B3)** — explicit encoder/decoder around a shared
   equilibrium core — is exactly the shape a pretrained stack converts into:
   keep the first/last layers explicit, tie and iterate the middle.
2. **Anytime-unrolled training (F24/B1)** — the technique that closed the
   width gap (ratio 0.991) and trains 2.1x faster. It becomes the uptraining
   objective, now combined with teacher distillation.
3. **Trajectory-local contraction (F24/B2)** — the only method that produced a
   certified equilibrium; carried as an arm toward the open problem
   (quality-preserving certification) at real scale.
4. **Warm-start decoding (F19)** and the **budget dial** — serving-time
   properties that matter more, not less, at 1.7B.
5. **Truthful second-price token auctions (F6/F22/F23)** — now over genuinely
   capable specialists, where the F23 closed-loop failure gets a fair retest.

## Integrity: the MPO naming collision (found 2026-08-28)

**"Magnetic Preference Optimization" is already published** (arXiv 2410.16714,
ICLR 2025): policy-space magnetic mirror descent with **self-play**, targeting
the Nash equilibrium of a preference game. Our H3 method is a **different
mechanism** — a parameter-space magnetic pull inside AdamW toward frozen base
weights, applied to the standard DPO loss — but the name and motivation
collide. Actions, binding:
- Our method is renamed **PMA (parameter-space magnetic anchoring)** everywhere.
- The paper cites 2410.16714, states the distinction explicitly, and scopes
  F21's negative dose-response result to PMA — it is NOT evidence against
  policy-space MPO.
- The scale study adds policy-space MPO as a third arm where feasible, so the
  comparison is complete rather than self-serving.

## Program (pre-registered in SPEC 0011–0013)

- **H7 / SPEC 0011 — KineticLM-1.7B**: convert Qwen3-1.7B to the EqLMCore
  topology via anytime+distillation uptraining; retain quality at reduced
  parameters with a working budget dial.
- **H8 / SPEC 0012 — PMA vs DPO/SimPO at 1.7B** on real preference data.
- **H9 / SPEC 0013 — auction decoding over real specialists**, scoring-time
  and closed-loop, with context-aware bids (the F23 follow-up).

## Hardware split

5090 (32GB): all training/uptraining. GB10 (121GB unified, thermal governor
active, operator-cleared): evaluation harness, serving, and inference-heavy
auction work — the two proceed in parallel and never contend for one GPU.

## Evaluation standard

lm-evaluation-harness (EleutherAI) as the public standard: MMLU, GSM8K, ARC-C,
HellaSwag, plus IFEval for instruction following. Every number reported
alongside the same-harness score for the unmodified base — no cross-paper
number lifting.
