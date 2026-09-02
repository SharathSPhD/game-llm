# ADR 0011 — The programme halts at the scale boundary; closure without a NULL

Date: 2026-09-02 · Status: accepted · Amends: [ADR 0010](0010-closure-after-interventions.md)

## Context

ADR 0010, accepted earlier the same day, closed the scale programme *after*
SPEC 0024's two interventions, which were then running (I1 since 09:17, I2
queued). At 13:00 the operator reviewed the utility case once more: both
arms of the 1B twin score at chance on every public benchmark at 2.5B
tokens; the tied arm's projected perplexity at 10B tokens (≈ 70–100) would
still leave it there; Pythia-410m's token budget is roughly 220 GPU-days on
the one card available; and the interventions, whatever they read, could
only refine the *mechanism* of a result whose product consequence was
already fixed. The programme's objective was utility.

## Decision

1. SPEC 0024 is halted immediately. I1 was stopped at 13:06 with 209.7M of
   500M tokens seen; I2 does not run; SPEC 0023 C1 does not run. The partial
   I1 trajectory is archived (`results/scale/exp39/i1_blocklr_halted/`) and
   described in F55 as an observation, not a reading.
2. No NULL is declared. The closure contract requires two completed
   interventions before a NULL, and they were not completed. The 1B result
   is recorded as a **scale boundary** (F55) with the mechanism unresolved.
   This is a deliberate, documented departure from ADR 0010's closure point,
   made by the operator, and the record says so.
3. Everything else in ADR 0010 stands: the equilibrium decoding layer is the
   product; the app ships replay-only until the GB10 returns, with the F41
   council as the default live council; serving is profile-driven; the
   paper goes to arXiv as v2 with the boundary stated; the Tarka layer for
   F45–F54 closes before the tag.
4. The programme is closed with the tag `v2.0.0-closure`. Resumption, if
   ever funded, follows `docs/RESUMPTION.md`.

## Consequences

Roughly four GPU-hours of I1 are discarded and the mechanism question
(learning-rate scaling versus supervision schedule versus capacity) stays
open. In exchange the card is free for nothing — the point is that no
further training was going to change what ships. The paper's twin section
reports the halt rather than a mechanism, and its limitations section
states the unresolved question plainly.
