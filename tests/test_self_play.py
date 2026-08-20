"""Tests for SPPO self-play module.

Validates:
    1. Multiplicative weights update correctly adjusts log-weights
    2. Win-rate computation is symmetric and bounded
    3. Semantic gate and latent repulsion behave correctly
    4. Full self-play loop converges
"""

import torch

from kinetic_ai.config import SelfPlayConfig
from kinetic_ai.games.self_play import (
    BradleyTerryPreference,
    latent_repulsion,
    multiplicative_weights_update,
    run_self_play,
    semantic_gate,
    sppo_round,
)


class TestMultiplicativeWeights:
    """Tests for the multiplicative weights update."""

    def test_positive_advantage_increases_weight(self) -> None:
        """Positive advantage should increase the log-weight."""
        log_w = torch.zeros(3)
        advantages = torch.tensor([0.3, -0.1, 0.0])
        updated = multiplicative_weights_update(log_w, advantages, eta=1.0)
        assert updated[0] > log_w[0], "Positive advantage should increase weight"
        assert updated[1] < log_w[1], "Negative advantage should decrease weight"
        assert updated[2] == log_w[2], "Zero advantage should not change weight"

    def test_eta_scaling(self) -> None:
        """Larger eta should produce larger weight changes."""
        log_w = torch.zeros(3)
        advantages = torch.tensor([0.2, 0.2, 0.2])

        updated_small = multiplicative_weights_update(log_w, advantages, eta=0.1)
        updated_large = multiplicative_weights_update(log_w, advantages, eta=10.0)

        assert (updated_large > updated_small).all()

    def test_zero_eta_no_change(self) -> None:
        """eta=0 should produce no change."""
        log_w = torch.randn(5)
        advantages = torch.randn(5)
        updated = multiplicative_weights_update(log_w, advantages, eta=0.0)
        assert torch.allclose(updated, log_w)


class TestSPPORound:
    """Tests for a single SPPO round."""

    def test_round_returns_valid_output(self) -> None:
        """SPPO round should return updated weights and mean advantage."""
        config = SelfPlayConfig(eta=1.0)
        log_w = torch.zeros(4)
        win_rates = torch.tensor([0.6, 0.5, 0.4, 0.5])

        updated, mean_adv = sppo_round(log_w, win_rates, config)

        assert updated.shape == log_w.shape
        assert isinstance(mean_adv, float)

    def test_winning_response_upweighted(self) -> None:
        """Responses with win_rate > 0.5 should be upweighted."""
        config = SelfPlayConfig(eta=1.0)
        log_w = torch.zeros(3)
        win_rates = torch.tensor([0.8, 0.2, 0.5])

        updated, _ = sppo_round(log_w, win_rates, config)

        # Convert to probabilities to check
        probs = torch.softmax(updated, dim=-1)
        assert probs[0] > probs[1], "Higher win rate should have higher weight"


class TestSemanticGate:
    """Tests for the S-SPPO semantic supervision gate."""

    def test_low_confidence_passes(self) -> None:
        """When model is wrong, gate should allow update (mask=1)."""
        predictions = torch.zeros(2, 5)  # Uniform-ish logits
        references = torch.tensor([0, 1])

        mask = semantic_gate(predictions, references, threshold=0.5)
        # Softmax of zeros = uniform = 0.2 per class, which is < 0.5
        assert mask.sum() == 2.0, "All should pass when confidence < threshold"

    def test_high_confidence_blocks(self) -> None:
        """When model is confident and correct, gate should block (mask=0)."""
        predictions = torch.zeros(2, 5)
        predictions[0, 0] = 100.0  # Very confident on class 0
        predictions[1, 1] = 100.0  # Very confident on class 1
        references = torch.tensor([0, 1])

        mask = semantic_gate(predictions, references, threshold=0.5)
        assert mask.sum() == 0.0, "Should block when model is correct and confident"


class TestLatentRepulsion:
    """Tests for the S-SPPO latent repulsion force."""

    def test_identical_embeddings_max_repulsion(self) -> None:
        """Identical embeddings should produce maximum repulsion."""
        embeddings = torch.ones(3, 4)  # All identical
        loss = latent_repulsion(embeddings, strength=1.0)
        assert loss > 0, "Identical embeddings should have positive repulsion"

    def test_distant_embeddings_low_repulsion(self) -> None:
        """Well-separated embeddings should have low repulsion."""
        embeddings = torch.eye(4) * 100  # Very far apart
        loss = latent_repulsion(embeddings, strength=1.0)
        assert loss < 0.01, "Distant embeddings should have near-zero repulsion"

    def test_single_embedding_zero_repulsion(self) -> None:
        """Single embedding should have zero repulsion."""
        embeddings = torch.randn(1, 4)
        loss = latent_repulsion(embeddings, strength=1.0)
        assert loss == 0.0


class TestBradleyTerryPreference:
    """Tests for the Bradley-Terry preference model."""

    def test_higher_reward_preferred(self) -> None:
        """Response with higher reward should be preferred."""

        def reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
            return response.sum(dim=-1, keepdim=True)

        pref = BradleyTerryPreference(reward_fn)
        prompt = torch.zeros(1, 3)
        response_a = torch.ones(1, 3) * 2  # Higher reward
        response_b = torch.ones(1, 3) * 1  # Lower reward

        prob = pref.preference_prob(prompt, response_a, response_b)
        assert prob > 0.5, "Higher reward should be preferred"

    def test_equal_rewards_fifty_fifty(self) -> None:
        """Equal rewards should give 0.5 probability."""

        def reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0]])

        pref = BradleyTerryPreference(reward_fn)
        prompt = torch.zeros(1, 3)
        response_a = torch.ones(1, 3)
        response_b = torch.ones(1, 3)

        prob = pref.preference_prob(prompt, response_a, response_b)
        assert torch.isclose(prob, torch.tensor(0.5), atol=1e-5)


class TestFullSelfPlay:
    """Integration test for the full self-play loop."""

    def test_self_play_runs(self) -> None:
        """Full self-play loop should run without errors."""

        def reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
            return response.mean(dim=-1, keepdim=True)

        pref = BradleyTerryPreference(reward_fn)
        config = SelfPlayConfig(num_rounds=3, eta=0.5)

        prompts = torch.randn(2, 4)
        responses = torch.randn(2, 3, 4)  # 2 batches, 3 candidates

        log_weights, results = run_self_play(pref, prompts, responses, config)

        assert len(results) == 3
        assert log_weights.shape == (3,)
        for r in results:
            assert len(r.win_rates) == 3
            assert r.policy_entropy >= 0

    def test_self_play_converges_to_best_response(self) -> None:
        """SPPO should converge to Nash: best response gets highest weight, entropy decreases.

        With deterministic reward ranking, policy should concentrate on best response
        and entropy should strictly decrease over rounds.
        """

        def reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
            """Deterministic reward: first coordinate differentiates responses."""
            return response[:, 0:1]

        pref = BradleyTerryPreference(reward_fn)
        config = SelfPlayConfig(num_rounds=5, eta=0.5)

        # Responses with fixed rewards: [2.0, 1.0, 0.0]
        prompts = torch.zeros(2, 4)
        responses = torch.zeros(2, 3, 4)
        responses[:, 0, 0] = 2.0  # Best
        responses[:, 1, 0] = 1.0  # Middle
        responses[:, 2, 0] = 0.0  # Worst

        log_weights, results = run_self_play(pref, prompts, responses, config)

        entropies = [r.policy_entropy for r in results]

        # Convergence Property 1: Entropy strictly decreases (mode collapse)
        assert entropies[0] > entropies[-1], (
            f"Entropy should decrease (mode collapse). Got {entropies[0]:.4f} → {entropies[-1]:.4f}"
        )

        # Convergence Property 2: Monotonic decrease
        for i in range(len(entropies) - 1):
            assert entropies[i] >= entropies[i + 1] - 1e-6, (
                f"Entropy must monotonically decrease: {entropies[i]:.4f} at round {i} "
                f"then {entropies[i + 1]:.4f} at round {i + 1}"
            )

        # Convergence Property 3: Best response concentrated
        final_policy = torch.softmax(log_weights, dim=-1)
        assert final_policy[0] > final_policy[1], (
            f"Best response should have highest weight: {final_policy[0]:.4f} > {final_policy[1]:.4f}"
        )
        assert final_policy[1] > final_policy[2], (
            f"Ranking should match rewards: {final_policy[1]:.4f} > {final_policy[2]:.4f}"
        )

    def test_self_play_converges_to_uniform_with_equal_rewards(self) -> None:
        """With equal rewards, Nash = uniform; entropy and policy should stay constant.

        When all responses have equal reward, advantages should be zero and policy
        should remain uniform with constant entropy.
        """

        def reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
            """All responses equally good."""
            return torch.ones(prompt.shape[0], 1)

        pref = BradleyTerryPreference(reward_fn)
        config = SelfPlayConfig(num_rounds=5, eta=0.5)

        prompts = torch.zeros(2, 4)
        responses = torch.randn(2, 3, 4)

        log_weights, results = run_self_play(pref, prompts, responses, config)

        # Expected entropy for uniform distribution over 3 responses
        uniform_policy = torch.ones(3) / 3
        uniform_entropy = -(uniform_policy * torch.log(uniform_policy)).sum().item()

        # Entropy should stay constant at uniform value
        for r in results:
            assert abs(r.policy_entropy - uniform_entropy) < 0.01, (
                f"With equal rewards, entropy should be {uniform_entropy:.4f}, "
                f"got {r.policy_entropy:.4f}"
            )

        # Policy should be uniform
        final_policy = torch.softmax(log_weights, dim=-1)
        expected_uniform = torch.ones(3) / 3
        assert torch.allclose(final_policy, expected_uniform, atol=0.01), (
            f"With equal rewards, policy should be uniform {expected_uniform}, got {final_policy}"
        )

    def test_win_rates_vary_with_policy(self) -> None:
        """Win rates must change as policy evolves (demonstrates fixed-response bug).

        This test verifies that win rates vary across rounds, indicating that
        the algorithm is sampling from the current policy distribution rather than
        using fixed responses.
        """

        def reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
            return response.sum(dim=-1)

        pref = BradleyTerryPreference(reward_fn)

        # Two responses with very different rewards to show clear preference
        base_responses = torch.tensor([
            [10.0, 10.0, 10.0, 10.0],  # Dominant
            [1.0, 1.0, 1.0, 1.0],       # Poor
        ])

        prompts = torch.zeros(1, 4)
        # Reshape responses to (batch=1, num_responses=2, dims=4)
        responses = base_responses.unsqueeze(0)
        config = SelfPlayConfig(num_rounds=5, eta=0.5)

        _, results = run_self_play(pref, prompts, responses, config)

        # Extract win rates for response 0 across rounds
        rates_r0 = [r.win_rates[0] for r in results]
        variance = max(rates_r0) - min(rates_r0)

        # With proper resampling, variance should be significant
        # With fixed responses, variance ≈ 0 (bug)
        assert variance > 0.1, (
            f"Win rates should vary with policy evolution. "
            f"Variance {variance:.3f} indicates fixed responses (bug not fixed). "
            f"Rates: {rates_r0}"
        )

    def test_semantic_calibration_parameter_affects_convergence(self) -> None:
        """S-SPPO semantic_calibration=True should differ from False.

        Enabling semantic calibration with latent repulsion should change how
        the algorithm updates, resulting in different final log weights compared
        to standard SPPO when embeddings are provided.
        """

        def reward_fn(prompt: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
            return response.sum(dim=-1, keepdim=True)

        pref = BradleyTerryPreference(reward_fn)

        # Without semantic calibration (standard SPPO)
        config_no_cal = SelfPlayConfig(
            num_rounds=5, eta=0.5, semantic_calibration=False, repulsion_strength=0.0
        )
        torch.manual_seed(42)
        prompts = torch.randn(4, 3)
        responses = torch.randn(4, 3, 3)
        # Create embeddings for candidates
        embeddings = torch.randn(3, 8)
        log_w_no_cal, _ = run_self_play(
            pref, prompts, responses, config_no_cal, embeddings=embeddings
        )

        # With semantic calibration and latent repulsion
        config_with_cal = SelfPlayConfig(
            num_rounds=5, eta=0.5, semantic_calibration=True, repulsion_strength=0.1
        )
        torch.manual_seed(42)
        prompts = torch.randn(4, 3)
        responses = torch.randn(4, 3, 3)
        embeddings = torch.randn(3, 8)
        log_w_with_cal, _ = run_self_play(
            pref, prompts, responses, config_with_cal, embeddings=embeddings
        )

        # Must differ due to latent repulsion being applied
        assert not torch.allclose(log_w_no_cal, log_w_with_cal, atol=1e-6), (
            "repulsion_strength parameter has NO EFFECT"
        )
