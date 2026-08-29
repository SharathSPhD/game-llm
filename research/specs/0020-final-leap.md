# SPEC 0020 — The final leap: a Qwen-class kinetic model, instruction-usable

Registered 2026-08-29 on operator direction: BabyLM was the initial foray, the
council was a digression, and the return to BabyLM was a scale-back. The
programme's intent is a novel kinetic architecture at open-weight scale,
instruction-tuned, benchmarked head-to-head against open-weight models.

## Why the naive leap already failed, and what that teaches

F35 collapsed Qwen3-1.7B's middle layers twelve-to-one and uptrained on 98M
tokens: perplexity nearly recovered while reasoning was destroyed (GSM8K 0.457
to 0.017). Published recursive-uptraining recipes spend 10-100B tokens. The
lesson is not that conversion is impossible but that reuse ratio and budget must
be matched: twelve-fold reuse needs budgets this hardware does not have.

F45 established the exchange rate at gentle scale: tying at equal compute costs
about 4% quality and saves 63% of parameters when trained from scratch with
anytime supervision. F25 located where conversion damage lives (outer layers
carry disproportionate function; average-init dominates stepwise). F47 closed
depth modulation. F24 identified anytime unrolled supervision as the regime that
closes tying gaps.

## The design, assembled from findings

Base: Qwen2.5-1.5B-Instruct (28 layers, d=1536), the ladder's strongest 1.5B.
Surgery: keep the first four and last four layers explicit (F25); tie the middle
twenty layers as TEN cores applied TWICE each — a two-fold reuse ratio, not
twelve-fold, chosen because damage grows steeply with reuse and two-fold is the
gentlest surgery that still saves meaningful parameters. Average-init within
each pair (F25). No depth modulation (F47).

Unique parameters: roughly 1.10B of 1.54B — a 28% reduction — with compute per
token unchanged at 28 block applications.

Training: knowledge distillation from the unmodified base model plus anytime
supervision at intermediate depths (F24/F35 machinery, exp15), targeting 3-6B
tokens over five to seven days split across the RTX 5090 and GB10.

## The honest competitive frame, fixed now

The model is NOT claimed against its own teacher. The head-to-head is
parameter-bracketed: at roughly 1.1B unique parameters it is benchmarked against
Llama-3.2-1B-Instruct, Qwen2.5-0.5B-Instruct and SmolLM2-1.7B-Instruct on the
established ladder (MMLU, ARC-C, HellaSwag, GSM8K with chat templates,
WinoGrande, PIQA), measured on our harness. Its compute per token exceeds a 1B
explicit model's and this is disclosed with every number (F44 discipline): the
claim is quality per PARAMETER, aimed at memory-limited deployment (F48).

Success, pre-registered: mean ladder score at or above Llama-3.2-1B-Instruct's,
at no more than 1.15B unique parameters, while retaining at least 90% of the
teacher's mean. Anything less is reported as the measured exchange rate at scale.

## Gate before the week is committed

A damage probe runs first, in hours rather than days: convert at two-fold reuse
(and one-core-per-four as a harsher comparison), measure initial perplexity
against F25's twelve-fold numbers, then a short uptraining run to check the
recovery slope. Kill criteria, fixed now: if two-fold initial damage exceeds
5x base perplexity, or the slope after 2,000 steps projects held-out perplexity
above 1.5x base at budget end, the leap is reported as gated by hardware and the
programme closes on the F45 exchange rate instead. The probe is the EFE move:
the cheapest observation that decides the largest commitment.

## Closure discipline

This spec bounds the remaining science. In parallel, the application builds its
staged surfaces (browser demo of the 46M artifact, anytime dial, benchmark
page, API); the paper takes its final pass; and if the leap lands, the flagship
becomes the converted model served through the same stack, instruction behavior
inherited from the Instruct base and preserved through distillation.
