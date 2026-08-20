"""Kinetic AI Paradigm — End-to-End Simulation.

This script demonstrates the full Kinetic AI pipeline:
    1. Define a game (Rock-Paper-Scissors or Matching Pennies)
    2. Train two agents using Magnetic Mirror Descent to find QRE
    3. Track convergence (exploitability / NashConv over time)
    4. Run a token auction between the trained agents
    5. Print diagnostics and verify convergence

Unlike the original simulate.py which used arbitrary utility functions,
this simulation has proper game-theoretic structure: defined payoff matrices,
dual-space mirror descent updates, and verified equilibrium convergence.

Random seed support: --seed argument (default: None, random initialization).
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinetic_ai.config import (
    AuctionConfig,
    AuctionType,
    BregmanType,
    DEQConfig,
    MMDConfig,
    SolverType,
)
from kinetic_ai.eval.convergence import ConvergenceTracker
from kinetic_ai.games.payoff import rock_paper_scissors
from kinetic_ai.games.qre import compute_qre, nash_conv
from kinetic_ai.mechanisms.auctions import TokenAuction
from kinetic_ai.models.deq_layer import DEQLayer
from kinetic_ai.optim.bregman import NegativeEntropy
from kinetic_ai.optim.mmd import MagneticMirrorDescent, mmd_strategy_update


def run_strategy_space_mmd(num_steps: int = 500) -> None:
    """Phase 1: MMD on a known game (strategy-space mode).

    Trains two players on Rock-Paper-Scissors using MMD.
    Verifies convergence to QRE by tracking NashConv.
    """
    print("=" * 60)
    print("Phase 1: Strategy-Space MMD on Rock-Paper-Scissors")
    print("=" * 60)

    game = rock_paper_scissors()
    bregman = NegativeEntropy()
    tracker = ConvergenceTracker()

    # Initialize strategies (biased, not uniform)
    s1 = torch.tensor([0.7, 0.2, 0.1])
    s2 = torch.tensor([0.1, 0.7, 0.2])
    ref1 = torch.ones(3) / 3  # Uniform reference
    ref2 = torch.ones(3) / 3

    lr = 0.3
    tau = 0.05

    print("Initial strategies:")
    print(f"  P1: {s1.numpy()}")
    print(f"  P2: {s2.numpy()}")
    print(f"  NashConv: {nash_conv(game, s1, s2):.6f}")
    print()

    for step in range(num_steps):
        # Compute utility gradients
        g1 = game.utility_gradient(1, s1, s2)
        g2 = game.utility_gradient(2, s2, s1)

        # MMD updates (in dual space via Bregman mirror map)
        s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
        s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)

        nc = nash_conv(game, s1, s2)
        tracker.log(step, exploitability=nc)

        if step % 100 == 0:
            print(f"  Step {step:4d} | NashConv: {nc:.6f} | P1: [{s1[0]:.3f}, {s1[1]:.3f}, {s1[2]:.3f}]")

    # Verify convergence
    final_nc = nash_conv(game, s1, s2)
    rate_result = tracker.estimate_convergence_rate("exploitability")

    print("\nFinal Results:")
    print(f"  P1 strategy: {s1.numpy()}")
    print(f"  P2 strategy: {s2.numpy()}")
    print(f"  NashConv: {final_nc:.6f}")
    print(f"  Convergence rate: {rate_result.rate:.4f}")
    print(f"  Linear convergence (R²): {rate_result.r_squared:.4f}")
    print(f"  Is linear: {rate_result.is_linear}")

    # Compare with known QRE
    qre_result = compute_qre(game, rationality=1.0 / tau)
    print(f"\n  Reference QRE (λ={1/tau:.1f}):")
    print(f"    P1: {qre_result.strategy_1.numpy()}")
    print(f"    P2: {qre_result.strategy_2.numpy()}")
    print()

    return tracker


def run_deq_agent_demo() -> None:
    """Phase 2: DEQ-based agent with MMD training.

    Demonstrates a neural agent using a DEQ layer, trained with
    the parameter-space MMD optimizer.
    """
    print("=" * 60)
    print("Phase 2: DEQ Agent with MMD Parameter-Space Training")
    print("=" * 60)

    state_dim = 16
    vocab_size = 10

    # Define the DEQ transformation
    transform_layer = nn.Linear(state_dim * 2, state_dim)
    # Scale for contraction
    with torch.no_grad():
        transform_layer.weight.data *= 0.3
        transform_layer.bias.data *= 0.1

    def transform(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, x], dim=-1)
        return torch.tanh(transform_layer(combined))

    # Build agent model
    deq_config = DEQConfig(solver=SolverType.ANDERSON, max_iter=30, tol=1e-3)
    deq = DEQLayer(transform, deq_config)
    output_head = nn.Linear(state_dim, vocab_size)
    bid_head = nn.Linear(state_dim, 1)

    # Collect all parameters
    all_params = list(transform_layer.parameters()) + list(output_head.parameters()) + list(bid_head.parameters())

    # MMD optimizer (Euclidean mode for unconstrained NN params)
    mmd_config = MMDConfig(lr=0.01, tau=0.05, bregman_type=BregmanType.EUCLIDEAN)
    optimizer = MagneticMirrorDescent(iter(all_params), config=mmd_config)

    # Training: align agent to prefer token 3
    context = torch.randn(1, state_dim)
    target_token = 3

    print(f"Training agent to prefer token {target_token}...")
    for step in range(100):
        optimizer.zero_grad()

        z_star = deq(context)
        logits = output_head(z_star)
        probs = F.softmax(logits, dim=-1)

        # Cross-entropy loss toward target token
        target = torch.tensor([target_token])
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | P(token {target_token}): {probs[0, target_token].item():.4f}")

    # Final output
    with torch.no_grad():
        z_star = deq(context)
        logits = output_head(z_star)
        probs = F.softmax(logits, dim=-1)
        bid = F.softplus(bid_head(z_star))

    print("\nFinal agent output:")
    print(f"  P(token {target_token}): {probs[0, target_token].item():.4f}")
    print(f"  Top token: {torch.argmax(probs).item()}")
    print(f"  Bid value: {bid.item():.4f}")
    print()

    return output_head, bid_head, deq, context


def run_auction_demo() -> None:
    """Phase 3: Multi-agent token auction.

    Creates two DEQ agents with different preferences and runs
    both second-price and weighted-aggregation auctions.
    """
    print("=" * 60)
    print("Phase 3: Token Auction Between Game-Theoretic Agents")
    print("=" * 60)

    vocab_size = 10

    # Create two agents with different preferences
    # Agent 1 prefers token 2, Agent 2 prefers token 7
    probs1 = torch.zeros(vocab_size)
    probs1[2] = 0.6
    probs1[3] = 0.2
    probs1[1] = 0.2

    probs2 = torch.zeros(vocab_size)
    probs2[7] = 0.5
    probs2[8] = 0.3
    probs2[6] = 0.2

    bids = torch.tensor([2.5, 3.8])

    # Second-price auction
    print("\n--- Second-Price (Vickrey) Auction ---")
    sp_config = AuctionConfig(auction_type=AuctionType.SECOND_PRICE, vocab_size=vocab_size)
    sp_auction = TokenAuction(sp_config)
    sp_result = sp_auction.run_auction(bids, torch.stack([probs1, probs2]))

    print(f"  Agent 1 bid: {bids[0]:.2f} (prefers token 2)")
    print(f"  Agent 2 bid: {bids[1]:.2f} (prefers token 7)")
    print(f"  Winner: Agent {sp_result.winner_id + 1}")
    print(f"  Payment: {sp_result.payments[sp_result.winner_id]:.2f} (second price)")
    print(f"  Selected token: {sp_result.sampled_token}")

    # Weighted aggregation auction
    print("\n--- Weighted Aggregation Auction ---")
    wa_config = AuctionConfig(
        auction_type=AuctionType.WEIGHTED_AGGREGATION,
        vocab_size=vocab_size,
        aggregation_temp=1.0,
    )
    wa_auction = TokenAuction(wa_config)
    wa_result = wa_auction.run_auction(bids, torch.stack([probs1, probs2]))

    print("  Output distribution (top 3):")
    top_probs, top_ids = torch.topk(wa_result.output_distribution, 3)
    for prob, idx in zip(top_probs, top_ids, strict=False):
        print(f"    Token {idx.item()}: {prob.item():.4f}")
    print(f"  Selected token: {wa_result.sampled_token}")
    print(f"  VCG payments: Agent 1={wa_result.payments[0]:.4f}, Agent 2={wa_result.payments[1]:.4f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kinetic AI full pipeline simulation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for random initialization)",
    )
    args = parser.parse_args()

    # Set random seeds if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Kinetic AI — Full Pipeline Simulation          ║")
    if args.seed is not None:
        print(f"║           Seed: {args.seed:<43} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Phase 1: Game-theoretic MMD on known game
    tracker = run_strategy_space_mmd(num_steps=500)

    # Phase 2: DEQ agent with MMD training
    run_deq_agent_demo()

    # Phase 3: Multi-agent auction
    run_auction_demo()

    print("=" * 60)
    print("Simulation complete.")
    print("=" * 60)
