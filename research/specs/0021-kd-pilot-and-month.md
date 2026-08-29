# SPEC 0021 — Distilled from-scratch kinetic 1B: pilot gate, then the month

Registered 2026-08-29 on operator decision (all four recommendations accepted).
v1 of the application, paper and artifacts ships now regardless; the run below
has a hard 30-day stop and lands as v1.1 or is reported wherever its curves
reached.

## The construction

A from-scratch tied student (the F45 recipe, which works) trained by knowledge
distillation from an Apache-licensed teacher (Qwen2.5-7B-Instruct for the month;
Qwen2.5-1.5B-Instruct colocated for the pilot). The teacher is the compressed
corpus: it carries the 10^13 tokens the hardware cannot re-absorb, and F51 does
not apply because no pretrained weights are merged — the signal is distilled
into fresh weights.

Data: Nemotron-CC high-quality slice, acquired during the pilot day;
FineWeb-Edu (already cached) is the pilot corpus and the recorded fallback.

## Pilot, which gates everything (~1 day, 5090)

Student: EqLM d=768, tied block, twelve iterations, teacher tokenizer
(vocab 151k), seq 512, anytime supervision. Two arms at identical tokens
(~0.5B): plain CE, and CE plus logit KD. Pre-registered gate: the KD arm must
reduce held-out perplexity by at least 15% against the CE arm at equal tokens.
Pass launches the month; fail closes the leap at F51's verdict with the pilot
recorded.

## The month (both machines)

Tied 1B student and explicit twin, both distilled identically from
Qwen2.5-7B-Instruct (teacher serving logits on GB10, students on the 5090 in
sequence or interleaved), 22-25B tokens. Primary claim: does the exchange rate
(F45: ~0.96 quality at 0.37x parameters) survive 1B scale under KD?

## Pre-registered external anchors

Beat Pythia-1B (300B tokens) and TinyLlama-1.1B (3T tokens) on the ladder.
Report the gap to Llama-3.2-1B-Instruct and Nemotron-Mini-4B as disclosed
context, never as claimed targets. Every number carries token budget, unique
parameters and compute per token (F44 discipline).

## Kill criteria for the month

Day-7 curve check: if held-out loss projects above the CE-only pilot trend
scaled to budget (KD buying nothing at scale), stop and report. Hard stop at
day 30 regardless of state.
