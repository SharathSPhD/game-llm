# SPEC 0018 — Compute-matched weight tying

Registered 2026-08-29, before the training run's first evaluation.

## What F44 leaves open

F44 established that the parity result held parameters and iterations equal
while the tied block cost 4.92 times an explicit layer, so parity was bought with
roughly five times the arithmetic, and equal arithmetic costs roughly a quarter
of the quality. That measures the exchange rate at one point on the curve — a
tied block made wide in order to match parameters — and says nothing about the
configuration a practitioner would actually choose if arithmetic were the
constraint.

## The configuration under test

Set the tied block to the explicit baseline's own width, d_model 768 with
d_ff 3072. One iteration is then exactly one explicit layer, and twelve
iterations are exactly twelve layers: compute is equal by construction rather
than by calibration, which removes the confound that made the earlier comparison
misreadable.

The tied model then holds roughly 7.1M block parameters against the baseline's
85.1M, a twelvefold reduction, with embeddings identical because the width is
identical. Total parameters fall from 123.8M to roughly 46M.

## Prediction, stated before the result

Quality will fall below the explicit baseline, because twelve applications of one
block cannot express what twelve distinct blocks can, and F20 already showed the
tied map's effective contraction degrades with iteration count. The question is
how far. A ratio above 0.95 at a twelfth of the block parameters would make
weight tying a strong parameter-compression technique at zero compute cost, which
is a more useful claim than the parity it replaces. A ratio below 0.85 would mean
the technique earns its place only where parameters are scarce enough to justify
losing that much quality, and the exchange rate should then be reported as the
finding rather than the architecture recommended.

## Protocol

Two seeds, 42 and 43, on the RTX 5090, using the identical token cache, step
count, batch size, optimiser settings and anytime supervision schedule as the
exp13 B1 arm, so the only difference from the 4.92x-compute model is the width
that makes compute equal. Evaluated on the same BLiMP subset against the same
exp10 A1 explicit baseline.

## What each outcome changes

A high ratio moves the architecture line toward parameter-constrained deployment
and makes the compute-matched configuration the one to release. A low ratio
closes the tying line as a quality proposition and leaves the exchange rate as
the contribution. Either way the number is reported with the parameter and
compute accounting beside it, since F44 showed that quoting quality alone is what
made the earlier claim misleading.
