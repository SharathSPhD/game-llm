"""Payoff matrices and game definitions.

Provides standard game definitions (normal-form and extensive-form)
used for testing and benchmarking game-theoretic algorithms.

Normal-Form Games:
    - Rock-Paper-Scissors (canonical zero-sum, 3x3)
    - Matching Pennies (simplest zero-sum, 2x2)
    - Prisoner's Dilemma (canonical cooperative/defect)
    - Custom matrix games

Extensive-Form Games:
    - Kuhn Poker (canonical EFG benchmark, 3 cards, 2 players)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class NormalFormGame:
    """A two-player normal-form (matrix) game.

    Attributes:
        payoff_1: Payoff matrix for player 1. Shape: (A1, A2)
            where A1 = number of actions for player 1,
                  A2 = number of actions for player 2.
            Entry (i, j) = payoff to player 1 when P1 plays action i
            and P2 plays action j.
        payoff_2: Payoff matrix for player 2. Shape: (A1, A2).
        name: Human-readable name.
    """

    payoff_1: Tensor
    payoff_2: Tensor
    name: str = "custom"

    @property
    def num_actions_1(self) -> int:
        return self.payoff_1.shape[0]

    @property
    def num_actions_2(self) -> int:
        return self.payoff_1.shape[1]

    @property
    def is_zero_sum(self) -> bool:
        return torch.allclose(self.payoff_1 + self.payoff_2, torch.zeros_like(self.payoff_1))

    def expected_payoff(self, strategy_1: Tensor, strategy_2: Tensor) -> tuple[float, float]:
        """Compute expected payoffs for both players.

        Args:
            strategy_1: Mixed strategy for player 1. Shape: (A1,)
            strategy_2: Mixed strategy for player 2. Shape: (A2,)

        Returns:
            Tuple of (expected_payoff_1, expected_payoff_2).
        """
        u1 = strategy_1 @ self.payoff_1 @ strategy_2
        u2 = strategy_1 @ self.payoff_2 @ strategy_2
        return u1.item(), u2.item()

    def best_response_payoff(self, player: int, opponent_strategy: Tensor) -> float:
        """Compute best response payoff for a player against opponent strategy.

        Args:
            player: 1 or 2.
            opponent_strategy: The opponent's mixed strategy.

        Returns:
            The payoff of the best response.
        """
        if player == 1:
            payoffs = self.payoff_1 @ opponent_strategy  # (A1,)
        else:
            payoffs = self.payoff_2.T @ opponent_strategy  # (A2,)  (but opponent is P1 here)
            # Actually: if player 2 is computing BR, opponent = P1's strategy
            payoffs = opponent_strategy @ self.payoff_2  # (A2,)
        return payoffs.max().item()

    def utility_gradient(
        self, player: int, own_strategy: Tensor, opponent_strategy: Tensor
    ) -> Tensor:
        """Compute gradient of expected utility w.r.t. own strategy.

        For player 1: ∂u1/∂σ1 = A · σ2
        For player 2: ∂u2/∂σ2 = A^T · σ1  (where A = payoff_2)

        Args:
            player: 1 or 2.
            own_strategy: This player's current strategy (unused for linear payoffs).
            opponent_strategy: The opponent's strategy.

        Returns:
            Gradient vector. Shape: (num_actions,)
        """
        if player == 1:
            return self.payoff_1 @ opponent_strategy
        else:
            return self.payoff_2.T @ opponent_strategy


def rock_paper_scissors() -> NormalFormGame:
    """Rock-Paper-Scissors: canonical 3x3 zero-sum game.

    Nash equilibrium: (1/3, 1/3, 1/3) for both players.
    """
    A = torch.tensor([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ])
    return NormalFormGame(payoff_1=A, payoff_2=-A, name="rock_paper_scissors")


def matching_pennies() -> NormalFormGame:
    """Matching Pennies: simplest 2x2 zero-sum game.

    Nash equilibrium: (0.5, 0.5) for both players.
    """
    A = torch.tensor([
        [1.0, -1.0],
        [-1.0, 1.0],
    ])
    return NormalFormGame(payoff_1=A, payoff_2=-A, name="matching_pennies")


def prisoners_dilemma() -> NormalFormGame:
    """Prisoner's Dilemma: canonical cooperative game.

    Actions: 0 = Cooperate, 1 = Defect
    Nash equilibrium: (Defect, Defect) — both players defect.
    """
    A1 = torch.tensor([
        [-1.0, -3.0],
        [0.0, -2.0],
    ])
    A2 = torch.tensor([
        [-1.0, 0.0],
        [-3.0, -2.0],
    ])
    return NormalFormGame(payoff_1=A1, payoff_2=A2, name="prisoners_dilemma")


def coordination_game() -> NormalFormGame:
    """Pure coordination game: both players want to match actions.

    Nash equilibria: (0,0) and (1,1) are both pure-strategy NE.
    """
    A = torch.tensor([
        [2.0, 0.0],
        [0.0, 1.0],
    ])
    return NormalFormGame(payoff_1=A, payoff_2=A, name="coordination_game")


@dataclass
class KuhnPokerGame:
    """Kuhn Poker: canonical 2-player extensive-form game.

    Cards: {J, Q, K} (values 0, 1, 2)
    Each player gets 1 card, ante 1 chip.

    Actions:
        - Check (passive)
        - Bet 1 chip

    Sequence of play:
        1. P1 acts: Check or Bet
        2a. If P1 checks: P2 acts: Check or Bet
            - If P2 checks: Showdown
            - If P2 bets: P1 acts: Fold or Call
        2b. If P1 bets: P2 acts: Fold or Call

    This game has a known Nash equilibrium (Kuhn 1950).
    NashConv at equilibrium = 0.

    Information sets for P1: {J, Q, K} × {initial, facing_bet} = 6
    Information sets for P2: {J, Q, K} × {check, bet} = 6
    """

    name: str = "kuhn_poker"

    # Card values
    JACK: int = 0
    QUEEN: int = 1
    KING: int = 2
    CARDS: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.CARDS = [self.JACK, self.QUEEN, self.KING]

    @property
    def num_info_sets_per_player(self) -> int:
        return 6  # 3 cards × 2 decision points

    @property
    def num_actions(self) -> int:
        return 2  # Check/Fold or Bet/Call

    def get_strategy_size(self) -> int:
        """Total number of action probabilities in a behavioral strategy."""
        return self.num_info_sets_per_player * self.num_actions

    def evaluate(
        self, strategy_1: Tensor, strategy_2: Tensor
    ) -> float:
        """Compute expected payoff for player 1 under given strategies.

        Strategies are behavioral strategies indexed by information set.
        strategy[info_set_idx * 2] = prob of action 0 (check/fold)
        strategy[info_set_idx * 2 + 1] = prob of action 1 (bet/call)

        Info set indexing:
            P1: card * 2 + {0: initial, 1: facing_bet}
            P2: card * 2 + {0: facing_check, 1: facing_bet}

        Returns expected payoff for P1 (zero-sum, so P2 = -P1).
        """
        total_payoff = 0.0

        for c1 in self.CARDS:
            for c2 in self.CARDS:
                if c1 == c2:
                    continue  # Can't have same card

                deal_prob = 1.0 / 6.0  # Uniform over 6 possible deals
                winner = 1 if c1 > c2 else -1

                # P1 info sets: card * 2 + context
                p1_initial = c1 * 2  # Initial decision
                p1_facing_bet = c1 * 2 + 1  # Facing P2's bet after P1 checked

                # P2 info sets: card * 2 + context
                p2_facing_check = c2 * 2  # P1 checked
                p2_facing_bet = c2 * 2 + 1  # P1 bet

                # Extract probabilities
                p1_bet_initial = strategy_1[p1_initial * 2 + 1]
                p1_check_initial = 1.0 - p1_bet_initial
                p1_call = strategy_1[p1_facing_bet * 2 + 1]
                p1_fold = 1.0 - p1_call

                p2_bet_after_check = strategy_2[p2_facing_check * 2 + 1]
                p2_check_after_check = 1.0 - p2_bet_after_check
                p2_call = strategy_2[p2_facing_bet * 2 + 1]
                p2_fold = 1.0 - p2_call

                # Path 1: P1 checks → P2 checks → Showdown (pot = 2)
                payoff_1 = deal_prob * p1_check_initial * p2_check_after_check * winner * 1

                # Path 2: P1 checks → P2 bets → P1 folds (P1 loses ante)
                payoff_2 = deal_prob * p1_check_initial * p2_bet_after_check * p1_fold * (-1)

                # Path 3: P1 checks → P2 bets → P1 calls → Showdown (pot = 4)
                payoff_3 = (
                    deal_prob * p1_check_initial * p2_bet_after_check * p1_call * winner * 2
                )

                # Path 4: P1 bets → P2 folds (P1 wins ante)
                payoff_4 = deal_prob * p1_bet_initial * p2_fold * 1

                # Path 5: P1 bets → P2 calls → Showdown (pot = 4)
                payoff_5 = deal_prob * p1_bet_initial * p2_call * winner * 2

                payoff_sum = payoff_1 + payoff_2 + payoff_3 + payoff_4 + payoff_5
                total_payoff += float(payoff_sum.item()) if isinstance(payoff_sum, Tensor) else payoff_sum

        return total_payoff

    def nash_equilibrium_p1(self) -> Tensor:
        """Return the known Nash equilibrium strategy for Player 1.

        Kuhn (1950): P1's optimal strategy (symmetric Nash):
            - J: bet with prob 1/3 (bluff), fold always after P2 bets
            - Q: check always, call with prob 1/3 after P2 bets
            - K: bet with prob 1 (value bet), call always

        The symmetric Nash has both players using identical strategies.
        This produces game value of -1/18 for P1 (zero-sum).

        Note: There's a family of NE parameterized by α ∈ [0, 1/3].
        This returns the one with α = 1/3.
        """
        strategy = torch.zeros(self.get_strategy_size())
        alpha = 1.0 / 3.0

        # J initial: bet with prob alpha (bluff)
        strategy[0 * 2 + 1] = alpha  # bet
        # J facing bet: always fold
        strategy[1 * 2 + 1] = 0.0  # call prob = 0

        # Q initial: always check
        strategy[2 * 2 + 1] = 0.0
        # Q facing bet: call with prob alpha
        strategy[3 * 2 + 1] = alpha

        # K initial: bet with prob 1 (always bet)
        strategy[4 * 2 + 1] = 1.0
        # K facing bet: always call
        strategy[5 * 2 + 1] = 1.0

        return strategy

    def nash_equilibrium_p2(self) -> Tensor:
        """Return the known Nash equilibrium strategy for Player 2.

        In the symmetric Kuhn poker Nash equilibrium, both players use
        the same behavioral strategy. This method returns the same strategy
        as nash_equilibrium_p1() for use by Player 2.

        Returns:
            Strategy tensor for Player 2 (identical to Player 1's).
        """
        return self.nash_equilibrium_p1().clone()
