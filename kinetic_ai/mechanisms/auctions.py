"""Token-by-token auction mechanism for multi-agent LLM orchestration.

Implements the mechanism design framework from Duetting et al. (WWW 2024),
where multiple LLM agents compete via auctions at each autoregressive
generation step.

Two modes:
    1. Winner-Take-All (Second Price): Classic Vickrey auction where the
       highest bidder's distribution is used. Truthful by Vickrey's theorem:
       dominant strategy is to bid your true valuation regardless of others' bids.

    2. Weighted Aggregation: Agents' distributions are mixed proportionally
       to their bids via a monotone aggregation function (softmax weights).
       This preserves contributions from all agents but DOES NOT satisfy
       incentive compatibility with the current VCG payment rule. Agents may
       find it profitable to overbid or underbid their true valuations.
       (See auctions-001 regression test for evidence.)

CAUTION: Only second-price is proven truthful. Weighted aggregation should
not be used when truthfulness is required (e.g., in competitive settings).
The monotonicity of aggregation (∂p_out/∂b_i ≥ 0) is necessary but NOT
sufficient for truthfulness; correct externality-based payments are also
required and not yet derived for this setting.

References:
    [1] Duetting et al. "Mechanism Design for Large Language Models"
        (WWW 2024, Best Paper, arXiv:2310.10826)
    [2] Myerson, R.B. "Optimal Auction Design" (1981)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from kinetic_ai.config import AuctionConfig, AuctionType


@dataclass
class AuctionResult:
    """Results from a single token auction round.

    Attributes:
        output_distribution: The final token probability distribution
            after auction resolution.
        sampled_token: Token sampled from the output distribution.
        winner_id: Index of the winning agent (for second-price mode).
        payments: Per-agent payments (what each agent pays).
        bids: Original bids submitted.
        agent_distributions: Original distributions submitted.
    """

    output_distribution: Tensor
    sampled_token: int
    winner_id: int
    payments: Tensor
    bids: Tensor
    agent_distributions: Tensor


class TokenAuction:
    """Token-level auction mechanism for multi-agent content generation.

    At each autoregressive step:
        1. Each agent i computes its preferred next-token distribution p_i
        2. Each agent i submits a scalar bid b_i representing its valuation
        3. The auctioneer aggregates distributions and determines payments

    MECHANISM PROPERTIES:
    - Second-price (Vickrey): Truthful by dominant strategy. Winner pays
      max(second-highest bid, reserve price).
    - Weighted aggregation: Monotone but NOT truthful with current VCG
      payments. Agents may benefit from strategic bidding (overbidding).

    Args:
        config: Auction configuration.
    """

    def __init__(self, config: AuctionConfig | None = None) -> None:
        self.config = config or AuctionConfig()

    def run_auction(
        self,
        agent_bids: Tensor,
        agent_distributions: Tensor,
    ) -> AuctionResult:
        """Run a single token auction round.

        Args:
            agent_bids: Scalar bids from N agents. Shape: (N,).
            agent_distributions: Probability distributions from N agents
                over the vocabulary. Shape: (N, vocab_size).

        Returns:
            AuctionResult with the resolved output distribution, sampled
            token, winner, and payment information.
        """
        if self.config.auction_type == AuctionType.SECOND_PRICE:
            return self._second_price_auction(agent_bids, agent_distributions)
        elif self.config.auction_type == AuctionType.WEIGHTED_AGGREGATION:
            return self._weighted_aggregation_auction(agent_bids, agent_distributions)
        else:
            raise ValueError(f"Unknown auction type: {self.config.auction_type}")

    def _second_price_auction(
        self,
        agent_bids: Tensor,
        agent_distributions: Tensor,
    ) -> AuctionResult:
        """Classic Vickrey (second-price) winner-take-all auction.

        Winner = highest bidder. Payment = second-highest bid.
        Output distribution = winner's distribution.

        Truthful by Vickrey's theorem: bidding your true valuation is
        a dominant strategy.
        """
        n_agents = len(agent_bids)
        assert n_agents == len(agent_distributions)

        # Apply reserve price
        eligible = agent_bids >= self.config.reserve_price
        if not eligible.any():
            # No eligible bidders: uniform distribution, zero payments
            output_dist = torch.ones(
                agent_distributions.shape[-1],
                device=agent_distributions.device,
            ) / agent_distributions.shape[-1]
            return AuctionResult(
                output_distribution=output_dist,
                sampled_token=int(torch.multinomial(output_dist, 1).item()),
                winner_id=-1,
                payments=torch.zeros(n_agents, device=agent_bids.device),
                bids=agent_bids,
                agent_distributions=agent_distributions,
            )

        sorted_bids, sorted_indices = torch.sort(agent_bids, descending=True)
        winner_idx = int(sorted_indices[0].item())

        # Second price: winner pays max(second-highest bid, reserve price)
        # This ensures truthfulness: winner pays at least the auctioneer's reserve
        payments = torch.zeros(n_agents, device=agent_bids.device)
        if n_agents > 1:
            payments[winner_idx] = max(sorted_bids[1].item(), self.config.reserve_price)
        else:
            payments[winner_idx] = self.config.reserve_price

        output_dist = agent_distributions[winner_idx]
        sampled_token = int(torch.multinomial(output_dist, 1).item())

        return AuctionResult(
            output_distribution=output_dist,
            sampled_token=sampled_token,
            winner_id=winner_idx,
            payments=payments,
            bids=agent_bids,
            agent_distributions=agent_distributions,
        )

    def _weighted_aggregation_auction(
        self,
        agent_bids: Tensor,
        agent_distributions: Tensor,
    ) -> AuctionResult:
        """Weighted aggregation auction with monotone output function.

        Instead of winner-take-all, all agents' distributions are mixed
        with weights proportional to their bids (via softmax for smoothness).

        The output distribution is:
            p_out(v) = Σ_i w_i · p_i(v)
        where:
            w_i = softmax(b / temperature)_i

        This satisfies the monotonicity condition:
            ∂p_out/∂b_i ≥ 0 (in a distributional sense)

        because increasing b_i increases w_i, which increases the influence
        of agent i's distribution on the output.

        TRUTHFULNESS WARNING: This mechanism is NOT truthful. The current VCG
        payment rule uses bids as if they were valuations, violating the
        separation of concerns required for incentive compatibility. Agents
        may find it profitable to bid strategically (overbidding or underbidding).

        Payment rule (attempted generalized second-price):
            Each agent pays their "externality" — the difference between
            the welfare of others with and without agent i. However, since
            welfare depends on bids (not true valuations), this does not
            guarantee truthfulness.
        """
        n_agents = len(agent_bids)
        assert n_agents == len(agent_distributions)

        # Apply reserve price
        eligible = agent_bids >= self.config.reserve_price
        if not eligible.any():
            output_dist = torch.ones(
                agent_distributions.shape[-1],
                device=agent_distributions.device,
            ) / agent_distributions.shape[-1]
            return AuctionResult(
                output_distribution=output_dist,
                sampled_token=int(torch.multinomial(output_dist, 1).item()),
                winner_id=-1,
                payments=torch.zeros(n_agents, device=agent_bids.device),
                bids=agent_bids,
                agent_distributions=agent_distributions,
            )

        # Compute mixing weights via softmax over bids
        weights = F.softmax(
            agent_bids / max(self.config.aggregation_temp, 1e-8), dim=0
        )  # (N,)

        # Aggregate distributions: weighted mixture
        output_dist = torch.einsum("i,iv->v", weights, agent_distributions)

        # Normalize (should already sum to ~1, but ensure numerical stability)
        output_dist = output_dist / output_dist.sum().clamp(min=1e-8)

        # Determine "winner" (highest weight agent)
        winner_idx = int(torch.argmax(weights).item())

        # Compute VCG-style payments (externality pricing)
        payments = self._compute_vcg_payments(
            agent_bids, agent_distributions, weights, output_dist
        )

        sampled_token = int(torch.multinomial(output_dist, 1).item())

        return AuctionResult(
            output_distribution=output_dist,
            sampled_token=sampled_token,
            winner_id=winner_idx,
            payments=payments,
            bids=agent_bids,
            agent_distributions=agent_distributions,
        )

    def _compute_vcg_payments(
        self,
        bids: Tensor,
        distributions: Tensor,
        weights: Tensor,
        output_dist: Tensor,
    ) -> Tensor:
        """Compute VCG (Vickrey-Clarke-Groves) externality payments.

        Each agent pays the harm they cause to others by participating.

        Payment_i = (welfare of others without i) - (welfare of others with i)

        Where "welfare of agent j" = b_j · similarity(p_j, p_out)
        measured by dot product (cosine-like).
        """
        n_agents = len(bids)
        payments = torch.zeros(n_agents, device=bids.device)

        for i in range(n_agents):
            # Compute output distribution without agent i
            mask = torch.ones(n_agents, dtype=torch.bool, device=bids.device)
            mask[i] = False

            if mask.sum() == 0:
                continue

            bids_without_i = bids[mask]
            dists_without_i = distributions[mask]
            weights_without_i = F.softmax(
                bids_without_i / max(self.config.aggregation_temp, 1e-8), dim=0
            )
            output_without_i = torch.einsum(
                "j,jv->v", weights_without_i, dists_without_i
            )
            output_without_i = output_without_i / output_without_i.sum().clamp(min=1e-8)

            # Welfare of others with agent i
            welfare_with_list = [
                bids[j] * torch.dot(distributions[j], output_dist)
                for j in range(n_agents)
                if j != i
            ]
            welfare_with = sum(welfare_with_list) if welfare_with_list else torch.tensor(0.0, device=bids.device)

            # Welfare of others without agent i
            mask_sum_result = mask.sum()
            n_without = int(mask_sum_result.item()) if isinstance(mask_sum_result, Tensor) else int(mask_sum_result)
            welfare_without_list = [
                bids[mask][j] * torch.dot(dists_without_i[j], output_without_i)
                for j in range(n_without)
            ]
            welfare_without = sum(welfare_without_list) if welfare_without_list else torch.tensor(0.0, device=bids.device)

            welfare_diff = welfare_without - welfare_with
            payments[i] = max(0.0, welfare_diff.item() if isinstance(welfare_diff, Tensor) else welfare_diff)

        return payments


class SequentialAuction:
    """Manages a sequence of token-by-token auctions for autoregressive generation.

    This wraps TokenAuction to handle the sequential nature of text generation,
    maintaining history and allowing agents to condition their bids on
    previously generated tokens.

    Args:
        config: Auction configuration.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(self, config: AuctionConfig | None = None, max_tokens: int = 100) -> None:
        self.auction = TokenAuction(config)
        self.max_tokens = max_tokens
        self.history: list[AuctionResult] = []

    def step(
        self,
        agent_bids: Tensor,
        agent_distributions: Tensor,
    ) -> AuctionResult:
        """Run one auction step and record the result.

        Args:
            agent_bids: Bids from N agents for this token position.
            agent_distributions: Preferred distributions from N agents.

        Returns:
            AuctionResult for this step.
        """
        result = self.auction.run_auction(agent_bids, agent_distributions)
        self.history.append(result)
        return result

    def get_generated_tokens(self) -> list[int]:
        """Return the sequence of tokens generated so far."""
        return [r.sampled_token for r in self.history]

    def get_total_payments(self) -> Tensor:
        """Return cumulative payments per agent across all steps."""
        if not self.history:
            return torch.tensor([])
        return torch.stack([r.payments for r in self.history]).sum(dim=0)

    def reset(self) -> None:
        """Clear the generation history."""
        self.history = []
