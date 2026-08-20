"""Self-Play Preference Optimization (SPPO / S-SPPO).

Implements the SPPO framework from Wu et al. (2024) and the semantic
extension S-SPPO from subsequent work.

Core Idea:
    Instead of training against a fixed reward model (as in RLHF/DPO),
    SPPO uses self-play: the model plays against copies of itself,
    generating preference pairs, and updates via multiplicative weights.

    The key insight is that this converges to a Nash equilibrium of the
    preference game, which corresponds to the optimal policy under the
    Bradley-Terry preference model.

Algorithm (SPPO):
    1. Sample pairs (x, y_win, y_lose) from the current policy
    2. Compute win rates via a preference model
    3. Update policy weights using multiplicative update:
       w_{t+1}(y|x) ∝ w_t(y|x) · exp(η · (win_rate(y) - 0.5))

S-SPPO Extension:
    Adds semantic calibration:
    - Supervision gate: Only update when the model's prediction is
      semantically incorrect (not just stylistically different)
    - Representation diversification: Add a repulsive force in latent
      space to prevent mode collapse

References:
    [1] Wu et al. "Self-Play Preference Optimization for Language Model
        Alignment" (arXiv:2405.00675)
    [2] Wu et al. "S-SPPO: Semantic Self-Play Preference Optimization"
        (arXiv:2606.01561)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from kinetic_ai.config import SelfPlayConfig


class PreferenceModel(Protocol):
    """Protocol for preference models.

    A preference model takes two responses and returns the probability
    that the first response is preferred over the second.
    """

    def preference_prob(
        self, prompt: Tensor, response_a: Tensor, response_b: Tensor
    ) -> Tensor:
        """Compute P(a > b | prompt).

        Args:
            prompt: The prompt/context. Shape depends on implementation.
            response_a: First response.
            response_b: Second response.

        Returns:
            Scalar probability that a is preferred over b.
        """
        ...


class BradleyTerryPreference:
    """Bradley-Terry preference model based on reward scores.

    P(a > b) = sigmoid(r(a) - r(b))

    where r(·) is a reward function.

    Args:
        reward_fn: Function mapping (prompt, response) → scalar reward.
    """

    def __init__(self, reward_fn: Callable[[Tensor, Tensor], Tensor]) -> None:
        self.reward_fn = reward_fn

    def preference_prob(
        self, prompt: Tensor, response_a: Tensor, response_b: Tensor
    ) -> Tensor:
        r_a = self.reward_fn(prompt, response_a)
        r_b = self.reward_fn(prompt, response_b)
        return torch.sigmoid(r_a - r_b)


@dataclass
class SelfPlayResult:
    """Results from one round of self-play.

    Attributes:
        win_rates: Average win rate of new policy vs. old. Should be > 0.5.
        policy_entropy: Entropy of the updated policy (monitors mode collapse).
        kl_from_reference: KL divergence from the reference policy.
        num_samples: Number of preference pairs evaluated.
    """

    win_rates: list[float]
    policy_entropy: float
    kl_from_reference: float
    num_samples: int


def multiplicative_weights_update(
    log_weights: Tensor,
    advantages: Tensor,
    eta: float,
) -> Tensor:
    """Multiplicative weights update (in log-space for numerical stability).

    log_w_{t+1} = log_w_t + η · advantage
    w_{t+1} ∝ w_t · exp(η · advantage)

    This is the core of SPPO: actions with positive advantage (win_rate > 0.5)
    get upweighted, actions with negative advantage get downweighted.

    Args:
        log_weights: Current log-weights. Shape: (num_responses,)
        advantages: Win-rate advantages (win_rate - 0.5). Shape: (num_responses,)
        eta: Step size for the multiplicative update.

    Returns:
        Updated log-weights (unnormalized).
    """
    return log_weights + eta * advantages


def compute_win_rates(
    preference_model: PreferenceModel,
    prompts: Tensor,
    responses: Tensor,
) -> Tensor:
    """Compute pairwise win rates for a set of responses.

    For each response i, the win rate is the average probability of
    being preferred over all other responses j ≠ i.

    Args:
        preference_model: The preference model to use.
        prompts: Prompt tensor. Shape: (batch, ...)
        responses: Response tensor. Shape: (batch, num_responses, ...)

    Returns:
        Win rates. Shape: (batch, num_responses)
    """
    batch_size = responses.shape[0]
    num_responses = responses.shape[1]
    win_rates = torch.zeros(batch_size, num_responses)

    for i in range(num_responses):
        total_wins = torch.zeros(batch_size)
        for j in range(num_responses):
            if i == j:
                continue
            p_win = preference_model.preference_prob(
                prompts, responses[:, i], responses[:, j]
            )
            total_wins += p_win.squeeze()
        win_rates[:, i] = total_wins / max(num_responses - 1, 1)

    return win_rates


def sppo_round(
    log_weights: Tensor,
    win_rates: Tensor,
    config: SelfPlayConfig,
) -> tuple[Tensor, float]:
    """Execute one round of SPPO.

    Args:
        log_weights: Current log-policy weights. Shape: (num_responses,)
        win_rates: Win rates from pairwise comparisons. Shape: (num_responses,)
        config: Self-play configuration.

    Returns:
        Tuple of (updated_log_weights, mean_advantage).
    """
    # Compute advantages: how much better than random (0.5)
    advantages = win_rates - 0.5

    # Multiplicative weights update
    updated_log_weights = multiplicative_weights_update(
        log_weights, advantages, config.eta
    )

    return updated_log_weights, advantages.mean().item()


def semantic_gate(
    predictions: Tensor,
    references: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """S-SPPO semantic supervision gate.

    Only allows updates when the model's prediction is semantically
    incorrect (not just stylistically different). This prevents the
    model from being penalized for valid but different responses.

    Args:
        predictions: Model's predicted log-probabilities. Shape: (batch, vocab)
        references: Reference/ground-truth token ids. Shape: (batch,)
        threshold: Confidence threshold for gating.

    Returns:
        Binary mask: 1 where update should be applied. Shape: (batch,)
    """
    # Check if the model assigns high probability to the reference
    probs = torch.softmax(predictions, dim=-1)
    ref_probs = probs.gather(1, references.unsqueeze(1)).squeeze(1)

    # Gate: only update when the model is wrong (low probability on reference)
    return (ref_probs < threshold).float()


def latent_repulsion(
    embeddings: Tensor,
    strength: float = 0.1,
) -> Tensor:
    """S-SPPO latent-space repulsive force for diversity.

    Adds a repulsive force between response embeddings in latent space
    to prevent mode collapse. Responses that are too similar in embedding
    space are pushed apart.

    Uses a soft repulsion kernel: F = -strength · Σ_{i≠j} exp(-||e_i - e_j||²)

    Args:
        embeddings: Response embeddings. Shape: (num_responses, embed_dim)
        strength: Repulsion strength coefficient.

    Returns:
        Repulsion loss (scalar, to be added to the training objective).
    """
    n = embeddings.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Pairwise distances
    dists = torch.cdist(embeddings, embeddings, p=2)  # (n, n)

    # Soft repulsion kernel (exclude diagonal)
    mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    repulsion = torch.exp(-dists[mask]).mean()

    return strength * repulsion


def run_self_play(
    preference_model: PreferenceModel,
    prompts: Tensor,
    responses: Tensor,
    config: SelfPlayConfig | None = None,
) -> tuple[Tensor, list[SelfPlayResult]]:
    """Run the full SPPO self-play loop.

    Args:
        preference_model: Model for computing preferences.
        prompts: Prompt tensor. Shape: (batch, ...)
        responses: Initial response candidates. Shape: (batch, num_responses, ...)
        config: Self-play configuration.

    Returns:
        Tuple of (final_log_weights, round_results).
    """
    config = config or SelfPlayConfig()
    num_responses = responses.shape[1]

    # Initialize uniform log-weights
    log_weights = torch.zeros(num_responses)
    results: list[SelfPlayResult] = []

    for round_idx in range(config.num_rounds):
        # Compute win rates
        win_rates = compute_win_rates(preference_model, prompts, responses)
        mean_win_rates = win_rates.mean(dim=0)  # Average over batch

        # SPPO update
        log_weights, mean_adv = sppo_round(log_weights, mean_win_rates, config)

        # Compute diagnostics
        policy = F.softmax(log_weights, dim=-1)
        entropy = -(policy * torch.log(policy + 1e-10)).sum().item()
        uniform = torch.ones_like(policy) / num_responses
        kl = F.kl_div(
            torch.log(policy + 1e-10), uniform, reduction="sum"
        ).item()

        results.append(
            SelfPlayResult(
                win_rates=mean_win_rates.tolist(),
                policy_entropy=entropy,
                kl_from_reference=kl,
                num_samples=prompts.shape[0] * num_responses,
            )
        )

    return log_weights, results
