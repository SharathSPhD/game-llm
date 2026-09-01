"""Council aggregation rules, and the label-convention recovery they depend on.

These are the rules measured in F29 and F30. They are collected here rather than
left in experiment scripts for two reasons: the paper's numbers should rest on
tested code, and the application exposes aggregation as a user-facing choice, so
the rule the researcher compares must be the same object the server runs.

The rules divide into three families. Averaging keeps every player's evidence and
is the bar the others have to clear. The voting and market rules discount a
player's own enthusiasm in different ways, on the theory that a confidently wrong
member is what damages an aggregate. The influence game solves for the
equilibrium of ADR 0008, of which averaging is the zero-rationality case.

Measured on 8,301 questions across four instruction-tuned models, none of the
rules beat averaging and the game degraded monotonically as influence
concentrated. The interpretation that survived every check is in F30: each rule
reweights one fixed body of evidence, and concentrating weight discards part of
it. That is worth knowing when choosing a rule here — the alternatives are
offered because the comparison is evidence, not because any of them is better.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kinetic_ai.decode.equilibrium import EquilibriumConfig, solve_equilibrium


def mean_rule(ell: torch.Tensor) -> torch.Tensor:
    """Uniform averaging in log space, the geometric mean of the players."""
    return ell.mean(dim=1).argmax(dim=-1)


def median_rule(ell: torch.Tensor) -> torch.Tensor:
    """Element-wise median, which a single emphatic player cannot move."""
    return ell.median(dim=1).values.argmax(dim=-1)


def drop_keenest_rule(ell: torch.Tensor) -> torch.Tensor:
    """Average each option after excluding whichever player likes it most."""
    if ell.shape[1] < 2:
        return mean_rule(ell)
    keenest = ell.max(dim=1).values
    return ((ell.sum(dim=1) - keenest) / (ell.shape[1] - 1)).argmax(dim=-1)


def leave_one_out_proposal_rule(ell: torch.Tensor) -> torch.Tensor:
    """Each player proposes its best option and the others price it.

    The proposer's own score is excluded from the valuation of its own proposal,
    so a player prevails only when the rest of the council independently finds
    its answer plausible. This carries the second-price intuition validated for
    token auctions (F6) across to whole answers.
    """
    if ell.shape[1] < 2:
        return mean_rule(ell)
    proposals = ell.argmax(dim=-1)
    totals = ell.sum(dim=1)
    own = ell.gather(2, proposals.unsqueeze(-1)).squeeze(-1)
    value = totals.gather(1, proposals) - own
    best = value.argmax(dim=1)
    return proposals.gather(1, best.unsqueeze(1)).squeeze(1)


def borda_rule(ell: torch.Tensor) -> torch.Tensor:
    """Rank-sum voting, which keeps preference and discards magnitude."""
    ranks = ell.argsort(dim=-1).argsort(dim=-1).float()
    return ranks.sum(dim=1).argmax(dim=-1)


def game_distribution(
    ell: torch.Tensor,
    beta: float = 0.25,
    tau: float = 0.0,
    eta: float = 0.5,
    max_iter: int = 32,
) -> torch.Tensor:
    """The equilibrium consensus of the influence game of ADR 0008, over options.

    At ``beta = 0`` the weights stay uniform and the fixed point is the softmax
    of the mean log-probability, which is the sense in which the game
    generalises averaging rather than competing with it. The solve reaches that
    point only to fp32 rounding, so on an exact tie between options the
    rounding residue decides the argmax — and it decides it differently on
    aarch64 and x86_64. Anything that must be platform-stable compares this
    distribution, not the index :func:`game_rule` selects from it.
    """
    y = solve_equilibrium(
        ell,
        EquilibriumConfig(beta=beta, tau=tau, eta=eta, max_iter=max_iter),
        reference=torch.softmax(ell.mean(dim=1), dim=-1),
    )
    assert isinstance(y, torch.Tensor)
    return y


def game_rule(
    ell: torch.Tensor,
    beta: float = 0.25,
    tau: float = 0.0,
    eta: float = 0.5,
    max_iter: int = 32,
) -> torch.Tensor:
    """The influence game of ADR 0008: the option the equilibrium consensus favours."""
    return game_distribution(ell, beta=beta, tau=tau, eta=eta, max_iter=max_iter).argmax(dim=-1)


RULES = {
    "mean": mean_rule,
    "median": median_rule,
    "drop_keenest": drop_keenest_rule,
    "leave_one_out_proposal": leave_one_out_proposal_rule,
    "borda": borda_rule,
    "game": game_rule,
}


def aggregate(ell: torch.Tensor, rule: str, **kwargs: float) -> torch.Tensor:
    """Apply a named rule to ``[B, N, K]`` player log-probabilities.

    Args:
        ell: Log-probabilities over K options for each of N players, batched.
        rule: A key of :data:`RULES`.
        **kwargs: Rule parameters; only the game takes any.

    Returns:
        The selected option index per batch element.
    """
    if rule not in RULES:
        raise KeyError(f"unknown rule {rule!r}; available: {sorted(RULES)}")
    fn = RULES[rule]
    return fn(ell, **kwargs) if rule == "game" else fn(ell)  # type: ignore[operator]


def competence_weights(accuracies: torch.Tensor, gamma: float = 20.0) -> torch.Tensor:
    """Fixed influence weights from each player's measured reliability.

    The only intervention in F29 and F30 that moved the aggregate, worth about a
    point. It requires held-out accuracies, so it is a property of a deployed
    council rather than of a single question, and it should be fitted on data the
    council is not then scored against.
    """
    return F.softmax(gamma * accuracies, dim=0)


def weighted_rule(ell: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Averaging with per-player weights, e.g. from :func:`competence_weights`."""
    return (weights.view(1, -1, 1) * ell).sum(dim=1).argmax(dim=-1)


def oracle_any_correct(ell: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Whether any single player selects the correct option.

    The ceiling on what selection could achieve. Reporting it alongside a rule's
    accuracy is what distinguishes "this rule is close to the best possible" from
    "this rule leaves most of the council's knowledge on the table"; on the
    measured council the gap was twenty points.
    """
    return (ell.argmax(dim=-1) == gold.unsqueeze(1)).any(dim=1)


def recover_label_offset(
    scores: torch.Tensor,
    labels: torch.Tensor,
    graded: torch.Tensor,
    threshold: float = 0.99,
) -> int | None:
    """Recover a task's answer-label convention from an existing grading.

    Benchmarks disagree about what an answer label means: some number their
    options from zero and some from one, and a single dataset can carry both
    conventions in different fields of the same record. Assuming one produces no
    error — just a wrong index, a plausible-looking accuracy, and conclusions
    drawn from noise. On WinoGrande the assumed convention scored players at 0.13
    against a true 0.62.

    The convention is instead recovered from evidence: whichever offset makes
    "the top-scoring option is the labelled one" agree with the grading that was
    actually recorded is the convention in force.

    Args:
        scores: ``[B, K]`` per-option scores from one grader.
        labels: ``[B]`` answer labels as the dataset states them.
        graded: ``[B]`` the correctness the harness recorded for those scores.
        threshold: Required agreement before an offset is accepted.

    Returns:
        The offset to add to a label to obtain an option position, or ``None``
        when neither candidate reproduces the grading — in which case the labels
        cannot be trusted and the caller should drop the task rather than guess.
    """
    best: int | None = None
    best_agreement = -1.0
    predicted = scores.argmax(dim=-1)
    for offset in (0, -1):
        gold = labels + offset
        valid = (gold >= 0) & (gold < scores.shape[-1])
        agreement = torch.where(
            valid, ((predicted == gold) == (graded == 1.0)).float(), torch.zeros(())
        ).mean()
        if float(agreement) > best_agreement:
            best, best_agreement = offset, float(agreement)
    return best if best_agreement >= threshold else None
