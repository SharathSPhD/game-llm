"""Tests for token auction mechanisms.

Validates:
    1. Second-price auction truthfulness properties
    2. Weighted aggregation monotonicity
    3. VCG payment correctness
    4. Sequential auction state management
    5. Distribution validity (sums to 1, non-negative)
"""

import pytest
import torch

from kinetic_ai.config import AuctionConfig, AuctionType
from kinetic_ai.mechanisms.auctions import SequentialAuction, TokenAuction


class TestSecondPriceAuction:
    """Tests for the Vickrey second-price auction."""

    def setup_method(self) -> None:
        config = AuctionConfig(auction_type=AuctionType.SECOND_PRICE, vocab_size=5)
        self.auction = TokenAuction(config)

    def test_highest_bidder_wins(self) -> None:
        """The agent with the highest bid should win."""
        bids = torch.tensor([1.0, 3.0, 2.0])
        dists = torch.softmax(torch.randn(3, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)
        assert result.winner_id == 1, "Highest bidder should win"

    def test_winner_pays_second_price(self) -> None:
        """Winner should pay the second-highest bid."""
        bids = torch.tensor([1.0, 5.0, 3.0])
        dists = torch.softmax(torch.randn(3, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)
        assert result.payments[1].item() == pytest.approx(3.0), "Payment should equal 2nd price"

    def test_output_is_winners_distribution(self) -> None:
        """Output distribution should be the winner's distribution."""
        bids = torch.tensor([1.0, 5.0])
        dists = torch.softmax(torch.randn(2, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)
        assert torch.allclose(result.output_distribution, dists[1])

    def test_single_agent(self) -> None:
        """Should handle single-agent case gracefully."""
        bids = torch.tensor([5.0])
        dists = torch.softmax(torch.randn(1, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)
        assert result.winner_id == 0
        assert result.output_distribution.sum() > 0

    def test_output_distribution_valid(self) -> None:
        """Output distribution should be a valid probability distribution."""
        bids = torch.tensor([1.0, 2.0, 3.0])
        dists = torch.softmax(torch.randn(3, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)
        assert torch.all(result.output_distribution >= 0)
        assert torch.isclose(result.output_distribution.sum(), torch.tensor(1.0), atol=1e-5)

    def test_second_price_respects_reserve_price(self) -> None:
        """Winner must pay at least the reserve price in second-price auction.

        Regression test for auctions-002: when second-highest bid falls below
        reserve price (only one eligible bidder), winner still pays reserve.

        Vickrey auction theory: payment = max(second_highest_bid, reserve_price)
        """
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE, vocab_size=5, reserve_price=2.0
        )
        auction = TokenAuction(config)

        # Scenario: Only agent 1 qualifies (bid 2.5 >= reserve 2.0)
        # Second-highest bid (1.5) is below reserve
        bids = torch.tensor([1.0, 2.5, 1.5])
        dists = torch.softmax(torch.randn(3, 5), dim=-1)

        result = auction.run_auction(bids, dists)

        # Agent 1 should win
        assert result.winner_id == 1, "Highest bidder should win"

        # Agent 1 must pay AT LEAST the reserve price (not the ineligible second-highest)
        assert result.payments[1].item() >= config.reserve_price, (
            f"Payment {result.payments[1].item()} must be >= reserve "
            f"{config.reserve_price}"
        )
        assert result.payments[1].item() == pytest.approx(config.reserve_price), (
            f"With only 1 eligible bidder, payment should equal reserve price "
            f"{config.reserve_price}, got {result.payments[1].item()}"
        )

    def test_second_price_payment_equals_second_highest(self) -> None:
        """Second-price auction payment equals second-highest eligible bid.

        This is the key property that makes Vickrey truthful. With payment
        independent of winner's own bid (only depends on others' bids),
        truthful bidding becomes optimal.
        """
        bids = torch.tensor([1.0, 5.0, 3.0])
        dists = torch.softmax(torch.randn(3, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)

        # Winner should pay second-highest bid
        assert result.payments[1].item() == pytest.approx(3.0)

        # Non-winners should pay 0
        assert result.payments[0].item() == pytest.approx(0.0)
        assert result.payments[2].item() == pytest.approx(0.0)


class TestWeightedAggregation:
    """Tests for the weighted distribution aggregation auction."""

    def setup_method(self) -> None:
        config = AuctionConfig(
            auction_type=AuctionType.WEIGHTED_AGGREGATION, vocab_size=5
        )
        self.auction = TokenAuction(config)

    def test_output_is_mixture(self) -> None:
        """Output should be a mixture of all agents' distributions."""
        bids = torch.tensor([1.0, 1.0])  # Equal bids
        dists = torch.zeros(2, 5)
        dists[0, 0] = 1.0  # Agent 0 wants token 0
        dists[1, 4] = 1.0  # Agent 1 wants token 4

        result = self.auction.run_auction(bids, dists)

        # With equal bids, output should mix equally
        assert result.output_distribution[0] > 0.3, "Should have weight on token 0"
        assert result.output_distribution[4] > 0.3, "Should have weight on token 4"

    def test_monotonicity(self) -> None:
        """Higher bid should give more influence on output.

        This is the key incentive-compatibility property.
        """
        dists = torch.zeros(2, 5)
        dists[0, 0] = 1.0
        dists[1, 4] = 1.0

        # Low bid for agent 0
        bids_low = torch.tensor([0.5, 2.0])
        result_low = self.auction.run_auction(bids_low, dists)
        weight_on_0_low = result_low.output_distribution[0].item()

        # High bid for agent 0
        bids_high = torch.tensor([5.0, 2.0])
        result_high = self.auction.run_auction(bids_high, dists)
        weight_on_0_high = result_high.output_distribution[0].item()

        assert weight_on_0_high > weight_on_0_low, "Higher bid should increase influence"

    def test_output_distribution_valid(self) -> None:
        """Output distribution should be valid."""
        bids = torch.tensor([1.0, 2.0, 3.0])
        dists = torch.softmax(torch.randn(3, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)
        assert torch.all(result.output_distribution >= 0)
        assert torch.isclose(result.output_distribution.sum(), torch.tensor(1.0), atol=1e-5)

    def test_vcg_payments_non_negative(self) -> None:
        """VCG payments should be non-negative."""
        bids = torch.tensor([1.0, 2.0, 3.0])
        dists = torch.softmax(torch.randn(3, 5), dim=-1)

        result = self.auction.run_auction(bids, dists)
        assert torch.all(result.payments >= 0), "VCG payments should be non-negative"

    def test_reserve_price(self) -> None:
        """Bids below reserve price should be excluded."""
        config = AuctionConfig(
            auction_type=AuctionType.WEIGHTED_AGGREGATION,
            vocab_size=5,
            reserve_price=2.0,
        )
        auction = TokenAuction(config)

        bids = torch.tensor([0.5, 1.0])  # Both below reserve
        dists = torch.softmax(torch.randn(2, 5), dim=-1)

        result = auction.run_auction(bids, dists)
        assert result.winner_id == -1, "No winner when all bids below reserve"

    def test_weighted_aggregation_not_truthful_with_vcg(self) -> None:
        """Weighted aggregation with current VCG payments is NOT truthful.

        Regression test for auctions-001: the current VCG payment formula uses
        bids (not true valuations) in welfare calculations, violating IC.

        This test documents that weighted aggregation with VCG is non-truthful.
        Agents may find it profitable to overbid (or underbid) their true valuation.

        For truthful weighted aggregation, a different payment rule is needed
        (Duetting et al. 2024 discusses this carefully for monotone aggregation).
        """
        # Setup: orthogonal preferences
        dists = torch.zeros(2, 5)
        dists[0, 0] = 1.0  # Agent 0 prefers token 0
        dists[1, 4] = 1.0  # Agent 1 prefers token 4

        # Agent 0 has true valuation v0=2.0
        v0 = 2.0
        opponent_bid = 1.0

        # Compute utility for different bids by agent 0
        test_bids = torch.tensor([1.0, 2.0, 3.0, 4.0])
        utilities = []

        for b0 in test_bids:
            bids = torch.tensor([b0.item(), opponent_bid])
            result = self.auction.run_auction(bids, dists)
            value = torch.dot(dists[0], result.output_distribution).item()
            payment = result.payments[0].item()
            utility = value - payment
            utilities.append(utility)

        # Find optimal bid
        optimal_idx = torch.tensor(utilities).argmax().item()
        optimal_bid = test_bids[optimal_idx].item()

        # Under CURRENT (non-truthful) mechanism, optimal_bid != v0=2.0
        # This documents the non-truthfulness; optimal bid is likely higher (overbidding)
        is_truthful = abs(optimal_bid - v0) < 0.4
        assert (
            not is_truthful
        ), (
            f"Weighted aggregation with VCG is expected to be NON-TRUTHFUL. "
            f"For v0={v0}, optimal bid is {optimal_bid} (should differ if non-truthful). "
            f"This documents that the mechanism does not satisfy dominant-strategy IC."
        )


class TestSequentialAuction:
    """Tests for sequential (token-by-token) auction management."""

    def test_generates_sequence(self) -> None:
        """Should produce a sequence of tokens."""
        config = AuctionConfig(
            auction_type=AuctionType.WEIGHTED_AGGREGATION, vocab_size=5
        )
        seq_auction = SequentialAuction(config, max_tokens=10)

        for _ in range(5):
            bids = torch.tensor([1.0, 2.0])
            dists = torch.softmax(torch.randn(2, 5), dim=-1)
            seq_auction.step(bids, dists)

        tokens = seq_auction.get_generated_tokens()
        assert len(tokens) == 5
        assert all(0 <= t < 5 for t in tokens)

    def test_total_payments_accumulate(self) -> None:
        """Payments should accumulate over steps."""
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE, vocab_size=5
        )
        seq_auction = SequentialAuction(config, max_tokens=10)

        for _ in range(3):
            bids = torch.tensor([1.0, 3.0])
            dists = torch.softmax(torch.randn(2, 5), dim=-1)
            seq_auction.step(bids, dists)

        total = seq_auction.get_total_payments()
        assert total.shape == (2,)
        # Agent 1 (winning) should have positive cumulative payment
        assert total[1] > 0

    def test_reset_clears_history(self) -> None:
        """Reset should clear all history."""
        config = AuctionConfig(
            auction_type=AuctionType.SECOND_PRICE, vocab_size=5
        )
        seq_auction = SequentialAuction(config)

        bids = torch.tensor([1.0, 2.0])
        dists = torch.softmax(torch.randn(2, 5), dim=-1)
        seq_auction.step(bids, dists)

        seq_auction.reset()
        assert len(seq_auction.get_generated_tokens()) == 0
