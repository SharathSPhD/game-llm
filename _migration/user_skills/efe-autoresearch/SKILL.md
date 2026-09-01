---
name: efe-autoresearch
description: Expected Free Energy driven autoresearch — choose the next experiment by minimising EFE over expected information gain, expected progress toward the goal, and what the run costs, then fold the outcome back into an explicit belief state. Use this whenever deciding what to run next in an empirical programme, prioritising experiments or ablations, planning a research cycle or sprint, deciding whether a result justifies an expensive follow-up, or when a line of work has stalled and the next step is unclear — and also whenever a project claims an "autoresearch loop", "active inference", "EFE", "epistemic value" or "information gain" in its docs, since the failure mode this skill exists to prevent is that claim being made without any computation behind it. Applies equally when reviewing whether experiment selection in an existing codebase is real or decorative.
---

# EFE-driven autoresearch

Empirical programmes drift toward two failure modes. They run the experiment that
is easiest to start, or they run the one whose result would be most satisfying.
Neither is the one that most reduces uncertainty about what to build. Expected
Free Energy makes the choice a computation, so it can be inspected, disagreed
with, and checked afterwards against what was actually run.

The construction is from active inference. An agent holds beliefs over hidden
state — the things it does not know but whose truth decides what to do next — and
scores each available action by the free energy it expects to incur. That score
decomposes into an epistemic term rewarding information gain and a pragmatic term
rewarding progress toward what is wanted. Subtracting cost gives a ranking. The
minimum is what to run.

## The failure this exists to prevent

The most common way an EFE loop goes wrong is that it does not condition on the
action. If the likelihood of an outcome does not depend on which experiment is
run, then every experiment yields the same expected information, every action
scores identically, and the agent is decoration on a decision made some other
way. This is not a hypothetical: it is what the first implementation in the
project this skill came from actually did, scoring all five candidates at exactly
−5.607, and its tests passed because they only checked that a number came back.

So the first thing to verify in any EFE implementation — including one you have
just written — is that different actions receive different scores, and that the
difference tracks something real about the experiments. `scripts/efe_rank.py`
refuses to produce a ranking when all candidates score alike, for this reason.

## Defining the hidden state

Hidden state is the set of questions whose answers change what gets built. A good
hypothesis is one where believing it and disbelieving it lead to different work.
"The model could be better" is not a hypothesis; "better players raise the
council's ceiling, rather than the aggregation rule being the constraint" is,
because confirming it sends effort to training and refuting it sends effort to
selection.

Keep hypotheses as independent binary marginals unless correlations have actually
been estimated. A joint distribution over five hypotheses has thirty-two cells
and no measurement to fill them; independent marginals are legible, updateable,
and honest about what is not known. State the simplification rather than hiding
it.

Three or five hypotheses is usually right. Fewer and the agent has nothing to
discriminate; more and the belief state becomes a fiction nobody maintains.

## Defining the actions

Each candidate experiment carries three things.

**Diagnosticity** is the probability of a positive outcome when a hypothesis is
true, paired with that probability when it is false. Equal values mean the
experiment cannot discriminate the hypothesis and contributes exactly zero
information about it — that zero is what makes the ranking depend on the action.
Sharper separation means more information. These numbers are judgements, not
measurements, and they are the part of the model most worth arguing about, so
write them where they can be seen and edited rather than burying them.

**Payoff** is how much confirming the hypothesis advances the goal. This is what
stops the agent from resolving questions that are merely open in preference to
questions that matter.

**Cost** is what running it consumes — GPU-hours, wall-clock, money, a reviewer's
attention. Cost belongs in the score rather than in a side constraint, because
the interesting decisions are exactly the ones where a cheaper experiment that
resolves less beats an expensive one that resolves more.

Include the null action of analysing data already collected. Offline reanalysis
costs nothing and is the class of action programmes most consistently underrate;
in the project this skill came from, an offline sweep over already-stored scores
closed an expensive training branch for free.

## The computation

For each hypothesis a candidate bears on, with prior `q` and diagnosticity
`(p_true, p_false)`:

```
p_positive = q · p_true + (1 − q) · p_false
posterior_if_positive = q · p_true / p_positive
posterior_if_negative = q · (1 − p_true) / (1 − p_positive)

epistemic += p_positive · KL(posterior_if_positive ‖ q)
           + (1 − p_positive) · KL(posterior_if_negative ‖ q)

pragmatic += payoff · p_positive
```

Then `G = −(epistemic + pragmatic) + cost_weight · cost`, and the action
minimising `G` is next. Both terms stay separately reportable: they answer
different questions and routinely disagree, and an experiment that is highly
informative about something inconsequential should be visibly that rather than
averaged into one opaque number.

Choose `cost_weight` by asking what quantity of resource is worth one nat of
information, and say so. A weight that makes every expensive experiment
unreachable has replaced the decision with a budget rule.

## Updating

Fold each outcome back in by Bayes, and record the information actually gained
alongside what was expected. A programme whose realised gains persistently fall
short of expectation has a miscalibrated model of its own experiments, and that
is worth knowing early.

Refuse to update a hypothesis on an experiment that carries no evidence about it.
Permitting that silently is how a belief state becomes fiction.

Convergence means confident either way. A hypothesis driven to near-certainly
false is as much a result as one driven to near-certainly true, and in most real
programmes there are more of the former.

## Validating against the record

An EFE model that cannot reproduce good decisions the programme already made is
not encoding its practice. Take two or three past choices whose outcomes are
known, seed the belief state as it stood at the time, and check the ranking puts
the chosen action first. Where it disagrees, either the model is wrong or the
past choice was — both are worth finding out, and the disagreement is more
informative than agreement.

Write that check as a test, not a one-off. It is the only thing standing between
a working selector and a plausible-looking one.

## Running it

`scripts/efe_rank.py` is a dependency-light implementation. Describe the state
and candidates in a JSON file and run:

```
python3 scripts/efe_rank.py candidates.json
```

It prints the ranking with both terms and the cost separated, refuses to rank
when all scores are identical, and exits non-zero in that case so it can gate a
cycle runner. `--json` emits machine-readable output for a dashboard or journal
entry. See `references/worked-example.md` for a complete input file, the maths
worked through by hand for one candidate, and a retrospective validation against
a real research record.

## Reporting a cycle

When a cycle closes, record what was believed going in, which action was chosen
and its expected gain, what was observed, the realised gain, and the updated
belief. That record is what makes the loop auditable later, and it is short
enough that there is no excuse for not keeping it.

Resist reporting the ranking as though it were an oracle. It is a model with
hand-set diagnosticities, and its value is that it makes those judgements
explicit and arguable, not that it removes judgement.
