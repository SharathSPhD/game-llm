# SPEC 0017 — Pre-registered confirmation of the magnetically anchored answer vote

Registered 2026-08-28, before any confirmation data exists. The offline result
(exp27 over exp23's stored candidates) showed every cell of a 14-point
(weighting, tau) grid beating the domain router in-sample, and all three
held-out folds positive with a mean margin of +0.0597. Because that data also
produced F34 and F36, an independent draw is required before any claim is made.

## The mechanism under test

The magnetically anchored answer vote (TRIZ session 2026-08-28, principles 13,
8, 24, 10): each of the four players generates a full solution; answers are
extracted and collapsed into equivalence classes; a class's score is its number
of votes plus tau if it is the class chosen by the domain router of F34. The
router is therefore the reference policy of the game and the council overrides
it only when its net vote margin exceeds tau. At large tau the mechanism is
exactly the router.

## Pre-registered configuration

Uniform weighting, tau = 1.0. Chosen as the middle of a grid the offline result
showed to be flat, before any confirmation prompt is drawn. No other cell may be
substituted when reporting the outcome; the full grid is reported for
transparency but the claim rides on this cell.

## Data

exp23 re-run at seeds 45, 46 and 47. Each seed draws fresh GSM8K and MMLU
questions through the seeded shuffles, so no confirmation question was seen by
the offline analysis. Same council, same prompts template, same budgets, same
extraction.

## Success criterion

Mean margin over the domain router across the three seeds greater than zero AND
pooled paired z at least 2.0, computed on questions where mechanism and router
disagree. Failure on either criterion is reported as the finding.
