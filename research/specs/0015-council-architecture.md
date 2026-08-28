# SPEC 0015 — The Equilibrium Council: a game-theoretic multi-teacher LLM system

Status: ACTIVE · Machines: 5090 (training/distillation) + GB10 (serving/eval)
Opened 2026-08-28 under operator direction. Supersedes the framing of SPEC 0011
as a standalone architecture study; the conversion work continues as a component
programme (SPEC 0014) rather than as the headline system.

## Objective

Build, benchmark and ship a language-model system whose depth, routing and
aggregation are equilibrium computations, and which **outperforms strong
open-weight single models** on public benchmarks at comparable active
parameters. The system is the deliverable: model set, data, API and application,
published so others can use and reproduce it.

## Architecture

The council is the substrate; **equilibrium decoding (ADR 0008) is the
architecture**. Each token's distribution is the solved quantal response
equilibrium of an influence game among the players, rather than a blend of their
outputs or a selection among them — averaging and routing are its degenerate
cases. The components below are what the players and the solve are made of.

- **Teachers.** Efficient domain specialists sharing one tokenizer, so token-level
  aggregation is well defined. A player earns its seat by being decisive where
  the others are unsure, which is a different objective from being a strong
  stand-alone model and is selected for directly. Sources may be mixed: existing open specialists
  where they are already strong, fine-tuned experts where a gap exists, and
  students distilled down from a larger model.
- **Aggregation.** Equilibrium decoding under the entropy mirror map, with the
  second-price token auction (F6, F27) and uniform logit averaging retained as
  measured incumbents and as the degenerate settings of the same solve. Which
  rule ships is decided by SPEC 0016's measurement, not by preference.
- **Rationality control.** Decoding temperature is treated as the QRE rationality
  parameter, exposed as a system control rather than a hyperparameter.
- **Equilibrium depth.** Weight-tied recursive cores (SPEC 0011/0014) provide the
  inference-budget dial where a teacher benefits from it; F26 constrains where
  this is safe, since conversion preserves likelihood but not multi-step
  capability.
- **Top-down distillation.** A larger model distils into the teachers, and the
  council's own aggregate output distils back into individual teachers, so the
  system improves its own components.

## Baseline ladder (all measured on our harness, never lifted from model cards)

Reported as a ladder rather than a single comparison: Qwen3-1.7B
(active-parameter matched), Qwen3-4B (roughly twice the active size), and a
Nemotron Nano-class sparse model (the closest architectural cousin). The council
is credited only where it wins on the same harness invocation, same sample
limits, same machine.

## Domain selection

Specialist domains are chosen from measured gaps in the ladder rather than a
preset list: Phase 0 evaluates the baselines across knowledge, reasoning,
commonsense, mathematics and truthfulness, and the domains where the
parameter-matched baseline is weakest become the teachers worth building.

## Phases

Phase 0 establishes the ladder and selects domains. Phase 1 assembles the
council from existing specialists and measures it against the ladder, giving a
shippable system early. Phase 2 replaces teachers with distilled students and
measures the lift. Phase 3 ships the OpenAI-compatible API, the Hugging Face
release (models, aggregator configuration, distillation data, model card with
the ladder), and the public benchmark dashboard in the application.

## Machine allocation

The 5090 carries all training and distillation. GB10's 121GB of unified memory
holds the full council resident for serving and evaluation, which no single
32GB card can do; the thermal governor remains active for all GB10 GPU work.
