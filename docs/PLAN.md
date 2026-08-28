# Kinetic AI — Living Plan

Revised 2026-08-28. This document is the working tracker: it records the
objectives in force, what phase the build is in, what each machine is doing, and
which decisions are still open. It is expected to change as measurements come
in — a plan that survives contact unchanged was not specific enough.

## Objectives in force

| # | Objective | Standing |
|---|---|---|
| O1 | A new paradigm, not incremental mimicry: the token distribution as a solved game | Paradigm defined (ADR 0008), implemented, unmeasured against baselines |
| O2 | Retain the kinetic core — MMD, QRE, implicit depth, truthful auctions | Every component of O1 is one of these strands; F21's correction is what placed the magnet in policy space |
| O3 | Beat the baseline ladder: Qwen3-1.7B matched, Qwen3-4B larger, Nemotron Nano sparse | Ladder measurement in progress on GB10 |
| O4 | Domain teachers chosen from measured eval gaps, not a preset list | Blocked on the ladder; selection criteria fixed |
| O5 | Top-down distillation from a larger model into the teachers | Not started; depends on O4 |
| O6 | Ship API, then Hugging Face release, then dashboard | Not started; council must beat something first |
| O7 | EFE-driven autoresearch with TRIZ for inventive steps | Active — cheap probes before decisive runs; TRIZ produced the recovery arms now training |
| O8 | Both machines swarmed, never idle, never contending | Active — 5090 trains, GB10 evaluates and serves |
| O9 | Paper and app updated in the same cycle as each result | Enforced by the `academic-paper-style` skill and CLAUDE.md |

## Where the build is

**Phase 0 — measure the ground.** Establishing what the baselines actually
score on our harness, since domain selection (O4) and every later claim (O3)
depend on numbers we produced rather than numbers from model cards. Running now
on GB10 across MMLU, ARC-Challenge, HellaSwag, GSM8K, WinoGrande, PIQA and
TruthfulQA for Qwen3-1.7B and the three candidate players.

**Phase 1 — does the paradigm pay?** The decisive question before anything is
built on top: does solving the equilibrium beat averaging, the auction, the best
single player, and the oracle router on the same prompts? A cheap probe over the
influence rationality and magnet strength narrows the parameter region first;
the full comparison follows. If the equilibrium does not beat averaging here, it
does not deserve a product, and the plan changes rather than the claim softening.

**Phase 2 — better players.** Teachers built for the domains the ladder shows to
be weak, by fine-tuning, by distillation from a larger model, or by adopting an
existing specialist where one is already strong. A player earns its seat by being
decisive where the others are unsure, which is a different objective from being a
good stand-alone model and should be selected for directly.

**Phase 3 — ship.** API first, because serving forces the system to actually
work end to end; then the Hugging Face release with the ladder in the model card;
then the dashboard as public evidence.

## Machine allocation

The RTX 5090's 32GB carries training and distillation: currently the two
TRIZ-derived recovery arms, then teacher construction. GB10's 121GB of unified
memory holds the whole council resident, which no single 32GB card can do, so it
carries evaluation, the equilibrium comparisons and serving. The thermal
governor stays active for all GB10 GPU work and is checked when any new job
starts, after it was found guarding nothing twice.

## Decision points ahead

Whether the equilibrium beats averaging decides whether Phase 2 builds players
for a council or the effort returns to single-model work. The ladder gaps decide
which domains get teachers. Whether a distilled student beats an adopted
open specialist decides where teacher effort goes. Latency under real serving
decides whether the cost argument in the PRD survives; if the solve turns out to
cost more than a small fraction of the forward passes, the iteration budget
becomes the first thing to cut.

## How the research loop runs

Cheap probes precede expensive runs, and the probe is chosen to reduce the
uncertainty that most affects the next decision — the (β, τ) probe before the
decisive comparison is the current instance. TRIZ is used when a measurement
stalls a line rather than as routine practice; it produced the depth-curriculum
and rank-annealed LoRA arms after the damage curve showed conversion could not
be recovered by more of the same. Adversarial review runs against any result
before it enters the paper, and its corrections are recorded rather than
silently applied.

## What would change this plan

A finding that the equilibrium is indistinguishable from averaging at every
setting, which would move effort to player quality instead of aggregation. A
latency measurement that breaks the cost argument. A ladder result showing the
parameter-matched baseline is strong everywhere, which would push the comparison
toward the larger baselines and change what counts as a win. Any of these is a
reason to revise this document, not to reinterpret the result.
