# SPEC 0016 — Equilibrium decoding: measurement programme

Status: ACTIVE · Machines: GB10 (all measurement; council resident in unified
memory) · Opened 2026-08-28 · Implements ADR 0008 · Serves PLAN Phase 1

## What is being measured

Whether solving for the equilibrium beats the aggregation rules it generalises.
The decoder is already implemented and unit-tested; what is unknown is whether
the extra expressiveness converts into task accuracy, and at which settings of
the influence rationality and the magnet.

## Systems compared

All systems decode the same prompts, closed-loop, from an identical shared chat
template, so the only difference is how the next token is chosen. Each player is
a Qwen2.5-1.5B-Instruct variant; vocabulary identity is enforced at load.

Single players establish the floor. Uniform logit averaging and the second-price
token auction are the incumbent aggregation rules, both already measured at this
scale (F27). A domain-correct routing baseline shows what perfect prompt-level
routing achieves. Equilibrium decoding enters as a family indexed by influence
rationality and magnet strength, since those are the parameters that interpolate
between averaging and routing.

## Procedure

A cheap probe over the parameter grid on twenty prompts narrows the region worth
spending on; the decisive comparison then runs the selected settings over the
full eighty-prompt mixed stream on three seeds, which is what the closure
contract requires for any number that leaves the repository.

Scoring is objective task accuracy — numeric match on GSM8K, letter match on
MMLU — because the judge-based metric used earlier proved dominated by the
judge's style prior rather than by system quality.

## Targets

The equilibrium is worth building on if it beats uniform logit averaging by more
than seed-to-seed noise on the mixed stream, since averaging is the strongest
incumbent measured so far and the auction was statistically indistinguishable
from it. Beating the best single player is necessary but not sufficient —
aggregation already achieves that.

Two secondary quantities decide whether the cost argument in the PRD holds:
solver iterations to convergence per token with warm starting, and wall-clock
overhead against the uniform-averaging system on identical prompts. The
iteration budget is a control, so degradation under truncation is measured
rather than assumed.

## What a negative result changes

If the equilibrium cannot be separated from averaging at any setting, the
conclusion is that aggregation quality is bounded by player quality at this
scale, and effort moves to Phase 2 teacher construction with averaging as the
aggregation rule. That is a redirection of build effort, not a softened claim.

## Amendment, 2026-08-28: the answer-level arena is closed

The measurement ran and separated nothing. Over 8,301 questions the best of 45
settings scored 0.6311 against uniform averaging's 0.6304 at a standard error of
0.0053, and accuracy fell monotonically as influence concentrated, by eight
points at the largest rationality tested (F29). Eleven aggregation rules, five
experiments and two families of repair — mechanism design and per-player
calibration — all landed at or below the mean (F30).

The pre-committed response above was to move effort to player quality. That
response is **not** taken, and the reason is a second measurement in the same
data: some player answers correctly on 82.6% of the questions while the best
single player manages 62.5%. Player quality is demonstrably not the binding
constraint; selection is. Redirecting to teacher construction would have been
the specified move and the wrong one, which is recorded here because a spec that
silently drops its own pre-commitment is worth less than one that says why.

What the arena could not test is what the amendment turns on. Answer-level
aggregation fixes the evidence: four distributions over the same options, from
players that never see what the others produced, so every rule is a reweighting
and reweighting cannot recover what it discards. The sequential setting supplies
evidence that is not fixed — each player writes a chain of reasoning the others
never had, and pricing a peer's chain is new information rather than a re-reading
of one's own beliefs.

## Phase 1b: cross-examination over generated solutions

Each council member answers the prompt in full, then every member scores every
member's solution, and selection rules operate on that matrix of valuations:
each candidate priced by the whole council, priced by everyone except its author
(the second-price intuition of F6 carried from tokens to solutions), chosen by
the influence game of ADR 0008 with candidates as options, chosen by
self-consistency over extracted answers, and the per-candidate oracle as the
ceiling. Best-single from the ladder is the bar.

Scoring solutions as text rather than tokens removes the shared-tokenizer
constraint, so the arena also widens which models may sit on a council. The
target is separation from best-single and from self-consistency by more than
seed noise across three seeds; a rule that merely matches self-consistency is
reproducing a known technique and does not support the paradigm.

Two prerequisites are stated because F28 found them violated. Generation must
apply each model's chat template and allow a per-domain token budget, since the
ladder's GSM8K column measured answer formatting rather than arithmetic; and
candidate valuations must be per-token rather than summed, or the market prices
brevity.

## Provenance

Every run records its resolved configuration hash and commit, writes per-seed
results, and stores per-position telemetry — influence weights, iteration counts,
convergence — for the application's equilibrium view and for the paper's figures.
