# ADR 0009 — Influence follows verification, not confidence

Date: 2026-08-28 · Status: accepted · Amends: [ADR 0008](0008-equilibrium-decoding-paradigm.md)

## Context

ADR 0008 defined the paradigm: the next-token distribution is the solved
τ-regularized quantal response equilibrium of an influence game among
model-players, with a player's influence following its agreement with the
emerging consensus. Uniform averaging and routing are the degenerate cases at
zero and large influence rationality, so the construction was expected to beat
both.

It was measured over 8,301 questions from 61 tasks with four instruction-tuned
players, aggregating over answer options so that one evaluation pass funded the
entire parameter sweep offline. The best of 45 settings scored 0.6311 against
uniform averaging's 0.6304 at a standard error of 0.0053, and accuracy fell
monotonically as influence concentrated — by eight points at the largest
rationality tested (F29). Eleven rules across five experiments, including
leave-one-out proposal pricing, trimmed and median aggregation, Borda count,
competence weighting, a learned per-question competence gate, and per-player
temperature calibration, all landed at or below the mean (F30). Calibration
worked as calibration, cutting expected calibration error from 0.151 to 0.037,
and still did not move the aggregate.

Against that, some player answers correctly on 82.6% of the same questions while
the best single player manages 62.5%.

## Decision

Two changes, both narrower than abandoning ADR 0008.

The influence payoff is no longer agreement with the consensus. That quantity
measures confidence, and the measurements above establish that confidence — in
every form tried, calibrated or raw, agreement or sharpness or dissent — does not
track competence well enough to allocate influence by. Influence must instead
follow a player's assessment of *what another player produced*, which is the only
signal in the system that a player did not generate itself.

The arena moves from answer-level aggregation to generation. This is not a
convenience: it is forced by what the measurements showed. Every rule at answer
level reweights one fixed body of evidence — four distributions over the same
options, from players that never see what the others produced — and concentrating
weight on a subset discards evidence a council with a twenty-point per-example
ceiling cannot afford to lose. New information, not new weights, is the only
route to that ceiling, and generation is where new information enters: a player
writing out a solution produces a chain of reasoning the other players never had,
so scoring a peer's chain is evidence rather than introspection.

## Consequences

Phase 2 teacher construction is deferred rather than accelerated. The
pre-registered response to a negative aggregation result was to move effort to
player quality; that response is not taken, because the same data shows player
quality is not the binding constraint. Reversing a pre-commitment requires a
recorded reason, and this is it.

The council may now include models that do not share a tokenizer, since solutions
are scored as text rather than aggregated as token distributions. This widens
the candidate pool the PRD's release can draw on.

The cost profile changes and the PRD's argument must be re-checked against it.
Answer-level aggregation cost one forward pass per player; cross-examination
costs one generation per player plus one scoring pass per pair, which is
quadratic in council size. A council of four is affordable; the claim that the
system runs at ensemble cost does not survive unmodified and will be restated
from measurement rather than from the design.

What ADR 0008 got right is retained. The solver is sound, converges in a handful
of iterations, contains averaging exactly at zero rationality, and remains the
selection rule over candidates. What changes is what it is given to weigh.

## Alternatives rejected

Abandoning the equilibrium for weighted ensembling was considered, since constant
competence weights were the only intervention that moved the answer-level
aggregate. It is rejected because a weighted ensemble is a set of better
constants rather than a different computation, and a gain of about one point,
selected as the best of eleven rules and roughly 1.5 standard errors, is not a
result to build a product on.

Improving the players first was considered and is the pre-registered response.
It is rejected for now on the evidence stated above, and would become correct
again if cross-examination also fails to separate from self-consistency, since
that would leave extraction genuinely exhausted.
