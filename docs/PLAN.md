# Kinetic AI — Living Plan

Revised 2026-08-28. This document is the working tracker: it records the
objectives in force, what phase the build is in, what each machine is doing, and
which decisions are still open. It is expected to change as measurements come
in — a plan that survives contact unchanged was not specific enough.

## Objectives in force

| # | Objective | Standing |
|---|---|---|
| O1 | A new paradigm, not incremental mimicry: the token distribution as a solved game | Refuted at answer level (F29): indistinguishable from averaging, and higher influence rationality is harmful. Open in the sequential arena, where influence can follow verification of a jointly-authored prefix rather than confidence |
| O2 | Retain the kinetic core — MMD, QRE, implicit depth, truthful auctions | Every component of O1 is one of these strands; F21's correction is what placed the magnet in policy space |
| O3 | Beat the baseline ladder | Ladder measured (F28). The bar is Qwen2.5-1.5B-Instruct at 0.626 MMLU, not the nominally matched Qwen3-1.7B at 0.583; no aggregation yet exceeds 0.642 |
| O4 | Domain teachers chosen from measured eval gaps, not a preset list | Deferred by F29: with 20 points of oracle headroom unclaimed, better players cannot be the bottleneck until selection works |
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

**Phase 1 — does the paradigm pay?** *Answered, in part, and the answer moved
this plan.* Aggregating over answer options let one GPU pass fund an unlimited
offline sweep, so the comparison ran over 8,301 questions instead of the tens a
generation sweep affords. Solving the equilibrium scored 0.6311 against uniform
averaging's 0.6304 with a standard error of 0.0053, and raising the influence
rationality cost up to eight points (F29). The mechanism did not merely fail to
help; the thing it does is the thing that hurts.

The diagnosis is specific and is what the plan now turns on. Influence follows
each player's agreement with the emerging consensus, which rewards confidence,
and confidence is not competence — the weakest player is no less emphatic for
being wrong two questions in three. Weighting players by measured reliability is
the only intervention that moved anything (+1.14 points), and a solved game
added nothing on top of it. Meanwhile some player answers correctly on 83% of
the questions against the best aggregate's 64%, so the council's complementarity
is real and almost entirely unclaimed.

**Phase 1b — influence from verification rather than confidence.** What answer
level cannot test is the only part of the construction an ensemble cannot
imitate: in generation the consensus is jointly authored, so a player conditions
on a prefix it might not have chosen and can be asked to *score what the council
proposed*. That is a verification signal rather than a confidence signal, and it
is the one channel F29 leaves open. The council becomes a market in which a
proposed continuation must clear the members' collective valuation, with
truthful bidding (F6) keeping the valuations honest. This is the current
decisive test, and it requires first repairing the generative harness, which F28
found to be measuring formatting rather than capability.

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

The first of these has now happened. F29 found the equilibrium indistinguishable
from averaging at every setting tested, which under the previous version of this
plan would have moved effort to player quality. It has not, and the reason is
worth recording: the same measurement showed a 20-point per-example oracle gap,
so the council's players are demonstrably good enough and selection is what
fails. Moving to better players would have been the pre-committed response and
the wrong one. What remains open is a
latency measurement that breaks the cost argument. A ladder result showing the
parameter-matched baseline is strong everywhere, which would push the comparison
toward the larger baselines and change what counts as a win. Any of these is a
reason to revise this document, not to reinterpret the result.
