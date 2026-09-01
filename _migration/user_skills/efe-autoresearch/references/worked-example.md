# A worked cycle, and a retrospective validation

Read this when setting up an EFE loop for the first time in a project, or when a
ranking looks wrong and the arithmetic needs checking by hand.

## Contents

- One candidate scored by hand, so the script's output can be verified
- Choosing diagnosticity values without pretending they are measurements
- Setting the cost weight
- Retrospective validation against a real research record
- What a cycle entry looks like when it closes

## One candidate, by hand

Take the hypothesis *better players raise the ceiling* at prior `q = 0.4`, and an
offline reanalysis with diagnosticity `(0.75, 0.25)` and payoff `0.5`.

The chance of a positive outcome, marginalising over whether the hypothesis
holds:

```
p_positive = 0.4 × 0.75 + 0.6 × 0.25 = 0.30 + 0.15 = 0.45
```

The two posteriors:

```
positive: 0.4 × 0.75 / 0.45          = 0.6667
negative: 0.4 × 0.25 / 0.55          = 0.1818
```

Each posterior's divergence from the prior, in nats:

```
KL(0.6667 ‖ 0.4) = 0.6667·ln(1.6667) + 0.3333·ln(0.5556) = 0.1444
KL(0.1818 ‖ 0.4) = 0.1818·ln(0.4545) + 0.8182·ln(1.3636) = 0.1105
```

Weighted by how likely each outcome is:

```
epistemic = 0.45 × 0.1444 + 0.55 × 0.1105 = 0.1258 nats
pragmatic = 0.5 × 0.45                    = 0.2250
cost      = 0                             = 0
G         = −(0.1258 + 0.2250) + 0        = −0.3508
```

which is what `efe_rank.py --example` prints for `offline_reanalysis`. If a
hand-check and the script disagree, the input file is usually wrong rather than
the arithmetic — most often a diagnosticity pair written in the wrong order.

Notice the shape of the epistemic term. It is largest when the prior sits near a
half and the diagnosticity is sharp, and it collapses toward zero as either the
belief hardens or the experiment blunts. That is the behaviour wanted: there is
nothing to learn from a sharp experiment about a settled question, and nothing to
learn from a blunt experiment about anything.

## Choosing diagnosticity honestly

These are judgements. The question to answer for each pair is: if this hypothesis
were true, how often would this experiment come out positive, and if it were
false, how often would it come out positive anyway?

Some anchors that keep the numbers from drifting into fantasy. A direct
measurement of the quantity in question, adequately powered, sits around
`(0.9, 0.1)`. A proxy measurement — right construct, wrong setting — sits nearer
`(0.75, 0.25)`. A simulation whose assumptions are themselves uncertain rarely
beats `(0.7, 0.3)`. An experiment that would look much the same either way is
`(0.5, 0.5)` and should be written as such rather than nudged to `(0.55, 0.45)`
to make it look worth running.

Being pessimistic here costs little. An experiment whose diagnosticity is
understated simply waits a cycle; one that is overstated consumes a budget and
returns a result nobody can interpret.

## Setting the cost weight

Ask what quantity of the resource is worth one nat, and write the answer into the
file as `cost_units`. If a day of GPU time is worth about one nat, the weight is
`1/24` per GPU-hour. If a reviewer's afternoon is worth a nat, the weight is
about `0.25` per hour of review.

Check the weight is doing work rather than deciding everything. Run the ranking
with `--cost-weight 0` as well: if the ordering is unchanged, cost is not
influencing the decision and can be left out of the story; if the ordering
inverts completely, the weight has replaced the analysis with a budget rule and
should be argued for explicitly.

## Retrospective validation

A model that cannot reproduce decisions already known to be good is not encoding
the programme's practice. The validation is cheap: seed the belief state as it
stood before a past decision, list the actions available then, and check the
ranking.

Worked case, from the programme this skill came from. Eleven aggregation rules
had failed to beat a plain average, and the open question was whether better
players would raise the ceiling. Two actions bore on it: an offline simulation
over already-stored scores at no GPU cost, and a distillation training run at
eight GPU-hours with slightly sharper diagnosticity and twice the payoff.

The ranking put the free simulation first, and the simulation went on to close
the branch — it showed the intervention would have moved the system in the wrong
direction, for nothing. The training run would have consumed a day to reach the
same conclusion.

What makes this a real test rather than a flattering one is that the training run
has both the sharper diagnosticity and the larger payoff. It loses only because
cost enters the score. An implementation that ignored cost would rank it first,
so the test discriminates.

Write these as tests in the project's suite. Ranking checks that live only in a
notebook stop being run.

## A closed cycle entry

Keep the record short enough that there is no excuse for skipping it:

```
cycle 27 · belief entropy in 2.383 nats
  chose  latency_measurement   G=-0.814  expected gain 0.339 nats  cost 0.5 GPU-h
  saw    solve costs 1.4x single-model serving — positive
  gained 0.287 nats realised (expected 0.339)
  belief P(solve_is_affordable_at_serving) 0.65 → 0.94
  entropy out 2.096 nats
```

The realised gain is worth recording next to the expected one. A programme whose
realised gains persistently undershoot has diagnosticity values that are too
optimistic, and the fix is to lower them rather than to keep being surprised.
