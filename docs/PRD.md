# Kinetic AI — Product Requirements

Owner: operator (SharathSPhD) · Maintained continuously · Last revised 2026-08-28

## What is being built

A language-model system whose next-token distribution is the solved equilibrium
of a game among model-players, delivered as a usable product rather than a
research artifact. Three things ship together, in this order: an
OpenAI-compatible inference API, a public model and data release, and an
application that exposes the system's equilibrium controls and its measured
standing against open-weight baselines.

## Who it is for

**Practitioners** who want better quality per unit of serving compute than a
single model of comparable active size, without adopting a bespoke serving
stack. They interact through the API and judge it on benchmark numbers, latency
and cost.

**Researchers** who want to reproduce or extend the result. They need the
models, the aggregation configuration, the distillation data, and enough
provenance to re-run any number that appears in the paper or the dashboard.

**The operator**, who needs to see the system's standing against the baseline
ladder at any moment, and to direct effort toward whichever component is
currently weakest.

## What makes it worth using

The system computes something a forward pass cannot. Existing options blend
model outputs (ensembling) or select among them (routing); both are degenerate
cases of solving for the equilibrium, which adapts per token and can take a
token from a non-specialist when the specialist is locally unsure. The
measured precedent is that one-shot aggregation already beat a domain-correct
oracle router at 1.5B (F27).

The cost argument is what makes that practical: after one forward pass per
player, the equilibrium solve is softmax and dot products over the vocabulary,
so the system runs at ensemble cost. Quality has to be bought with better
players and a better solve, not with more forward passes than an ensemble
would need.

## Requirements

### The system

The council must run every player over a single shared tokenizer, since
token-level aggregation is undefined otherwise; this is enforced at load time
rather than assumed. Aggregation is selectable — equilibrium, auction, uniform
averaging, single model — because the comparison between them is itself part of
the product's evidence. Every response carries routing telemetry: per-position
influence weights, solver iterations, and whether the equilibrium converged.

Rationality parameters are user-facing controls with meaning, not hidden
constants: the magnet strength that anchors fluency, the influence rationality
that ranges from averaging to routing, and the iteration budget that trades
compute for refinement.

### The API

OpenAI-compatible `/v1/chat/completions` so existing clients work unmodified,
with the kinetic controls exposed as optional extensions that degrade
gracefully when absent. Authentication and rate limiting are required before
any public exposure. Latency must stay within a small multiple of single-model
serving at the same active parameter count; if it does not, the cost argument
above fails and the design must change.

### The release

Published on Hugging Face: the specialist set, the aggregation configuration
needed to reproduce the council, the distillation data, and a model card
carrying the full baseline ladder with the harness invocation used. Numbers in
the card must be reproducible from the released artifacts on a single GPU.

### The application

A console for the system: chat against the council with the equilibrium
controls live; a leaderboard showing the council against each rung of the
ladder on public benchmarks with per-run provenance; and a view of the
equilibrium itself — influence weights per token, solver iterations, the
disagreements the solve resolved.

## Success criteria

The system beats Qwen3-1.7B, the active-parameter-matched baseline, on the
benchmark suite measured on our own harness. It is credited against Qwen3-4B
and a Nemotron Nano-class sparse model wherever it wins, and the losses are
shown too, since a ladder that only reports wins is not evidence.

Beyond benchmarks, the API serves real requests at acceptable latency, the
released artifacts let a third party reproduce the headline numbers, and the
application makes the system's behaviour legible rather than opaque.

## Non-goals

Matching frontier-scale models. Training a competitive base model from scratch,
which the available compute rules out. Serving at production scale or with
availability guarantees. Beating baselines by mimicking their architecture and
tuning for marginal gains — the system earns its place by computing something
different, or it does not earn it.
