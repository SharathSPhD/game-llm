# Design note — ordering objectives translated to English (EFE candidate, no GPU committed)

Source: prabhasa-samskrutam H-ORD thread (R0–R5), digested 2026-08-31 from
their record including every null. Operator direction: leverage the thread,
which "mostly reported null" — the nulls are the design constraints here.

## What their record fixes about any adoption

The quotient loss is a length-normalised squared difference of paired
sentence NLLs across licit reorderings; the contrastive hinge requires licit
permutations to price at least a margin below matched word-salad scrambles.
What worked, worked only as a **finishing pass on a pretrained spine**
(~2.5h, last blocks + head trainable): from-scratch application collapsed
the language model outright (their O3a, P0 2.671), passive augmentation
moved magnitudes but not pricing structure (r1a), and an architectural
memory module was inert without objective gradient (r1b). Their calibration
rung (R5) hit a hard ceiling — the objective enforces invariance, it cannot
calibrate deltas to rarity. Adoption here therefore means: one loss term,
applied late, judged by a pre-registered pricing battery, with no
architectural additions and no calibration ambitions.

## The translation problem

Sanskrit's kāraka licensing gives licit permutations for free; English does
not, and English word order carries more meaning, so the *invariance*
(quotient) claim is weaker here. The safer English objective is the
**contrastive** one: licit alternations must price below matched salads by
a margin — grammar-shaped pricing rather than order-blindness.

Licit-alternation sources for English, all generable from a dependency
parse with rule transforms and a parser round-trip filter: dative
alternation (gave the dog a bone / gave a bone to the dog), particle shift
(picked up the book / picked the book up), adjunct ordering (yesterday at
the park / at the park yesterday), coordination swap (apples and oranges),
and adverbial preposing (Suddenly, she left / She left suddenly). Illicit
controls are within-sentence word salads, byte-multiset-preserved, exactly
their P3. Their perturbation and battery machinery (P0–P4, paired
bootstrap, seeded draws) is language-agnostic and can be reimplemented
against this harness's eval conventions without touching their repository.

## The pilot, if EFE selects it

Finishing pass on the tied 10B checkpoint (post-extension): contrastive
hinge (margin swept over their values 0.05/0.10/0.15 bpb-equivalent) with a
small quotient term, last blocks + head trainable, hours-scale on either
machine. Pre-registered readouts before any run: canonical held-out ppl
must not degrade beyond 1% (their trade was ppl-improving; English may not
be); the licit-alternation tax must fall; salads must stay expensive
(licit/salad pricing ratio below a bar fixed at registration). Nulls are
recorded as nulls.

## Why this is spine, not dilution

A pricing constraint enforced through a margin objective is a mechanism-
design condition on the model's implicit valuations — the same family as
the truthful-bidding results (F6) — and it composes with the equilibrium
decoding stack rather than replacing any part of it. It adds no module, in
keeping with what their F7/r1b nulls punished.
