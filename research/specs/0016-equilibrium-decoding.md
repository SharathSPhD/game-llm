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

## Provenance

Every run records its resolved configuration hash and commit, writes per-seed
results, and stores per-position telemetry — influence weights, iteration counts,
convergence — for the application's equilibrium view and for the paper's figures.
