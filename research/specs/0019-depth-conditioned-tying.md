# SPEC 0019 — Depth-conditioned weight tying

Registered 2026-08-29 at 09:15, while seed 42 was in its first minutes of
training and before any evaluation existed.

## What F45 leaves open

At equal compute the tied block reaches 0.958 of the explicit baseline with 2.70
times fewer parameters. The residual four percent is attributable to a specific
deficiency rather than to capacity in general: twelve applications of one static
map cannot express what twelve distinct maps can, because every application is
the same function of its input.

## The mechanism

The TRIZ session for this gap (matrix cell 36/26, principles Local Quality and
Dynamics) differentiates the map in time rather than in parameters. A per-
iteration scale and shift modulates the shared block, so the same weights
implement a different function at depth three than at depth nine and the fixed
point becomes the equilibrium of a periodically time-varying system.

Cost is two vectors of width d_model per iteration: 18,432 parameters at
d_model 768 and twelve iterations, against the roughly 78M that untying eleven
blocks would add. Compute is unchanged, since modulation is elementwise. The
modulation initialises as the exact identity, so the conditioned and plain
models share an architecture and an initialisation and differ only by the
conditioning; iterations past the trained depth reuse the final modulation, so
the anytime budget dial of F24 survives.

## Prediction, stated before any result

Depth conditioning will recover part of the gap but not all of it. The reasoning
is that a scale and shift can differentiate the map's *output distribution* at
each depth without changing what function of the input it computes, so it should
help where the deficiency is calibration across depths and not where the
deficiency is genuinely distinct computation. A recovery of one to two of the
four points is the expected outcome.

## Success criteria, fixed now

Beating the plain tied block by more than the seed standard deviation of 0.0169
counts as the mechanism working. Reaching a ratio of 1.0 or above against the
explicit baseline counts as an outright win at equal compute with 2.7 times fewer
parameters, and would be the programme's first. A ratio indistinguishable from
the plain tied block means the loss is capacity rather than expressiveness, which
closes modulation as a direction and is reported as such.

## Protocol

Three seeds, 42 through 44, on the RTX 5090, identical token cache, steps, batch,
optimiser and anytime supervision schedule as exp32, evaluated on the same BLiMP
subset against the same exp10 A1 explicit baselines. The only difference from
exp32 is the modulation.
