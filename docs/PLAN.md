# Kinetic AI — Living Plan

Revised 2026-08-28. This document is the working tracker: it records the
objectives in force, what phase the build is in, what each machine is doing, and
which decisions are still open. It is expected to change as measurements come
in — a plan that survives contact unchanged was not specific enough.

## Course correction, 2026-08-29

The council line drifted from the programme's subject and is demoted to a side
result. Kinetic AI's claim is and always was a **single-model** claim: one model
whose depth, training and decoding are equilibrium computations, matched in
parameters and compute against a conventional model of the same size. A council
of four separate off-the-shelf models routed by a lookup table satisfies none of
that, and comparing four models against one is not an architecture comparison at
all — it is a systems comparison, and its eight-point margin is reported as such.
What the council work legitimately contributed is method: the ladder, the fair
bars, the pre-registration discipline and the adversarial review that caught two
inflated claims before they were published.

The thrust returns to EqLM. F24 established parity — the anytime-trained tied
block reaches a ratio of 0.991 against a param-matched twelve-layer explicit
transformer, one seed exceeding it. Parity is not the objective. The untested
property that could carry it past parity is the one an explicit stack cannot
express: a fixed-point model's depth is a stopping criterion rather than an
architecture, so it can spend five iterations on an easy token and twenty on a
hard one. At matched *mean* depth the equilibrium model spends the same compute
unevenly, and whether uneven spending wins is exp31's question.

## Objectives in force

| # | Objective | Standing |
|---|---|---|
| O1 | A new paradigm, not incremental mimicry: a SINGLE model whose depth, training and decoding are equilibrium computations | Parity reached at 121M (F24, ratio 0.991 at matched params and budget). **Live: adaptive per-token depth (exp31) — the property an explicit stack cannot express, tested at matched mean compute.** All council mechanisms refuted or reduced to their degenerate limits (F29/F36/F41) |
| O2 | Retain the kinetic core — MMD, QRE, implicit depth, truthful auctions | Every component of O1 is one of these strands; F21's correction is what placed the magnet in policy space |
| O3 | Beat the baseline ladder | **MET against the baseline single model, pre-registered (F41): 0.6194 vs 0.5361, +8.33 points, z=4.42 on 360 fresh questions, every seed positive, 1.25 expected generations per request.** Not met against a fallback router, which the same system equals by construction — the honest claim is the first, stated with the second |
| O4 | Domain teachers chosen from measured eval gaps, not a preset list | Reopened by F32/F33: the answer-level headroom was ~3 points once the oracle is gated on identifiable confidence, not 20, so better players are back in contention — and F33 shows the council already holds one genuine specialist whose strength only appears on generative work |
| O5 | Top-down distillation from a larger model into the teachers | Not started; depends on O4 |
| O6 | Ship API, then Hugging Face release, then dashboard | Not started; council must beat something first |
| O7 | EFE-driven autoresearch with TRIZ for inventive steps | Active — cheap probes before decisive runs; TRIZ produced the recovery arms now training |
| O8 | Both machines swarmed, never idle, never contending | Active — 5090 trains, GB10 evaluates and serves |
| O9 | Paper and app updated in the same cycle as each result | Enforced by the `academic-paper-style` skill and CLAUDE.md |

## Where the build is

**Phase 0 — measure the ground.** Establishing what the baselines actually
score on our harness, since domain selection (O4) and every later claim (O3)
depend on numbers we produced rather than numbers from model cards. Complete
(F28) across MMLU, ARC-Challenge, HellaSwag, WinoGrande, PIQA and TruthfulQA for
all four candidate players. GSM8K is excluded rather than reported: strict-match
scored zero for every model, which measures the answer format the task expects
and not the capability, so the generative harness needs chat templates and a
per-domain generation budget before any generative claim rests on it.

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

**Correction that reframes Phase 1.** Two audits landed after the results above
and both cut against how they were first reported. The twenty-point per-example
oracle is mostly not extractable: gating it on the correct player reaching even
0.5 confidence — twice chance on four options — drops the ceiling to 0.658, so
the best rule at 0.6415 was operating within 1.6 points of what those
distributions support rather than 20 short of it (F32). And the arena itself was
near-homogeneous, because the one task on which these players differ in kind was
excluded from it by a harness fault: with chat templates applied the mathematics
variant scores 0.795 on GSM8K against the generalist's 0.595, while placing last
on MMLU. A mixed arena carries ten points of routable headroom where the tested
set carried one (F33). The aggregation results stand for what they measured; what
they cannot support is a claim about councils whose members genuinely differ.

**Phase 1c — the anchored answer vote (current).** With every competing-with-
the-router route closed (F34–F38), the TRIZ engine inverted the dependency:
the router became the reference policy, and the council overrides it only when
its net vote margin exceeds the magnet strength. Offline, every parameter cell
beats the router and all three held-out folds are positive at a mean of +0.06.
SPEC 0017 pre-registers uniform weighting and tau = 1.0 on fresh seeds 45–47;
the claim stands or falls on that run. If confirmed, the mechanism ships as the
council's serving mode (it needs only one generation per player and answer
extraction — no cross-scoring passes, so it costs N single-model decodes and
nothing quadratic); if refuted, the finding is recorded and the programme
returns to candidate generation.

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

Whether influence driven by verification of a jointly-authored prefix behaves
differently from influence driven by confidence is the open question the whole
paradigm now rests on; F29 settled the confidence half of it in the negative.
F34 raised the bar that question has to clear: a domain router with no machinery
in it takes 7.5 points over the best single player and beats every council rule
measured, so a council that merely beats a single model has demonstrated nothing.
Whether a distilled student beats an adopted
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
