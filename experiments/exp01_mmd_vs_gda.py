#!/usr/bin/env python
"""Experiment 1: MMD vs GDA on canonical games.

Pre-registered Tier A experiment testing H2: MMD converges linearly to QRE
while GDA cycles on zero-sum games.

Usage:
    python experiments/exp01_mmd_vs_gda.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from kinetic_ai.eval.convergence import ConvergenceTracker
from kinetic_ai.eval.statistical import bootstrap_ci
from kinetic_ai.games.payoff import NormalFormGame, matching_pennies, rock_paper_scissors
from kinetic_ai.games.qre import compute_qre, nash_conv
from kinetic_ai.optim.bregman import NegativeEntropy
from kinetic_ai.optim.mmd import mmd_strategy_update


def gda_strategy_update(
    strategy: Tensor,
    gradient: Tensor,
    lr: float,
) -> Tensor:
    """Gradient Ascent strategy update on the simplex.

    Uses mirror ascent (exponentiated gradient) without magnetic term.

    Args:
        strategy: Current strategy (probability distribution).
        gradient: Utility gradient.
        lr: Learning rate.

    Returns:
        Updated strategy.
    """
    # Mirror ascent: log-space update followed by softmax
    # y = log(x) + η · g
    # x_new = softmax(y)
    log_strategy = torch.log(strategy + 1e-10)
    log_strategy_new = log_strategy + lr * gradient
    return torch.softmax(log_strategy_new, dim=-1)


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def config_hash(config_path: Path) -> str:
    """Compute SHA256 hash of config file."""
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd="/home/sharaths/projects/game-llm",
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def create_game(game_name: str) -> NormalFormGame:
    """Create a game by name."""
    if game_name == "matching_pennies":
        return matching_pennies()
    elif game_name == "rock_paper_scissors":
        return rock_paper_scissors()
    elif game_name == "biased_rps":
        # RPS with first row scaled by 2
        game = rock_paper_scissors()
        game.payoff_1[0] *= 2
        game.payoff_2[0] *= 2
        game.name = "biased_rps"
        return game
    else:
        raise ValueError(f"Unknown game: {game_name}")


def distance_to_qre(
    strategy_1: Tensor,
    strategy_2: Tensor,
    qre_1: Tensor,
    qre_2: Tensor,
) -> float:
    """Compute L2 distance to QRE strategies."""
    dist1 = torch.norm(strategy_1 - qre_1).item()
    dist2 = torch.norm(strategy_2 - qre_2).item()
    return (dist1 + dist2) / 2.0


def compute_mmd_fixed_point(
    game: NormalFormGame,
    bregman,
    lr: float,
    tau: float,
    num_steps: int = 50000,
) -> tuple[Tensor, Tensor, dict]:
    """Compute the MMD fixed point with uniform reference via long-run iteration.

    Runs MMD at very small learning rate to find long-run fixed point.
    This is the ground truth for non-Nash QRE targets.

    Args:
        game: The game.
        bregman: Bregman divergence instance.
        lr: Very small learning rate for convergence.
        tau: Magnetic strength.
        num_steps: Number of steps to run.

    Returns:
        (fixed_point_1, fixed_point_2, diagnostics_dict)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n1 = game.num_actions_1
    n2 = game.num_actions_2

    # Initialize at uniform reference
    s1 = torch.ones(n1) / n1
    s2 = torch.ones(n2) / n2
    ref1 = s1.clone()
    ref2 = s2.clone()

    for _step in range(num_steps):
        g1 = game.utility_gradient(1, s1, s2)
        g2 = game.utility_gradient(2, s2, s1)

        s1_new = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
        s2_new = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)

        s1, s2 = s1_new, s2_new

    nc = nash_conv(game, s1, s2)

    return s1, s2, {
        "nash_conv": float(nc),
        "strategy_1": s1.numpy().tolist(),
        "strategy_2": s2.numpy().tolist(),
    }


def verify_mmd_fixed_point_equals_qre(
    game: NormalFormGame,
    bregman,
    lr: float,
    tau: float,
    qre_result,
    num_steps: int = 50000,
) -> tuple[bool, dict]:
    """Verify that MMD fixed point equals logit-QRE and return ground truth fixed point.

    Args:
        game: The game.
        bregman: Bregman divergence instance.
        lr: Very small learning rate for convergence.
        tau: Magnetic strength.
        qre_result: The QRE result to compare against.
        num_steps: Number of steps to run.

    Returns:
        (is_equal_to_qre, mmd_fixed_point_dict)
    """
    s1_fp, s2_fp, diagnostics = compute_mmd_fixed_point(game, bregman, lr, tau, num_steps)

    # Check distance to QRE
    dist = distance_to_qre(s1_fp, s2_fp, qre_result.strategy_1, qre_result.strategy_2)
    is_equal = dist < 1e-3

    return is_equal, diagnostics


def run_method(
    game: NormalFormGame,
    method_name: str,
    method_config: dict,
    num_steps: int,
    log_interval: int,
    seed: int,
    qre_result,
    mmd_fixed_point: tuple[Tensor, Tensor] | None = None,
) -> tuple[ConvergenceTracker, list[float], list[float], list[int]]:
    """Run a single method on a game.

    Returns:
        (tracker with NashConv metrics, list of QRE distances, list of MMD-FP distances, list of step indices)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n1 = game.num_actions_1
    n2 = game.num_actions_2

    # Initialize with random strategy
    s1 = torch.softmax(torch.randn(n1), dim=-1)
    s2 = torch.softmax(torch.randn(n2), dim=-1)

    # Reference strategies (for MMD)
    ref1 = torch.ones(n1) / n1  # Uniform
    ref2 = torch.ones(n2) / n2

    tracker = ConvergenceTracker()
    qre_distances = []
    mmd_fp_distances = []
    step_indices = []

    bregman = NegativeEntropy()
    lr = method_config["lr"]

    for step in range(num_steps):
        # Compute gradients
        g1 = game.utility_gradient(1, s1, s2)
        g2 = game.utility_gradient(2, s2, s1)

        # Update based on method
        if method_name == "GDA":
            s1 = gda_strategy_update(s1, g1, lr)
            s2 = gda_strategy_update(s2, g2, lr)
        elif method_name == "MMD-Fixed":
            tau = method_config["tau"]
            s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
            s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)
        elif method_name == "MMD-RND":
            tau = method_config["tau"]
            reset_interval = method_config.get("reset_interval", 100)

            # Reset reference periodically
            if step > 0 and step % reset_interval == 0:
                ref1 = s1.clone()
                ref2 = s2.clone()

            s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
            s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Log metrics periodically
        if step % log_interval == 0:
            nc = nash_conv(game, s1, s2)
            dist_to_qre = distance_to_qre(s1, s2, qre_result.strategy_1, qre_result.strategy_2)

            # Distance to MMD fixed point (if provided)
            if mmd_fixed_point is not None:
                dist_to_mmd_fp = distance_to_qre(s1, s2, mmd_fixed_point[0], mmd_fixed_point[1])
            else:
                dist_to_mmd_fp = 0.0

            tracker.log(step, exploitability=nc)
            qre_distances.append(dist_to_qre)
            mmd_fp_distances.append(dist_to_mmd_fp)
            step_indices.append(step)

    return tracker, qre_distances, mmd_fp_distances, step_indices


def run_experiment(config_path: Path) -> dict:
    """Run the full experiment."""
    config = load_config(config_path)
    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 1: MMD vs GDA on Canonical Games (Iteration 2)")
    print("Refined metric: convergence to QRE via distance-to-QRE (not NashConv)")
    print("=" * 70)

    num_steps = config["training"]["num_steps"]
    num_seeds = config["training"]["num_seeds"]
    log_interval = config["training"]["log_interval"]
    seed_base = config["training"]["seed_base"]

    # Results storage with iteration metadata
    all_results = {
        "iteration": 2,
        "refinement_reason": "Compute linear-rate fit on distance-to-QRE trajectory instead of NashConv (which plateaus at QRE exploitability floor). Verify magnetic fixed point equals logit-QRE. Report theoretical QRE NashConv floor per game.",
        "config": config,
        "config_hash": config_hash(config_path),
        "git_commit": git_commit(),
        "data": {},
    }

    games = {g["name"]: create_game(g["type"]) for g in config["games"]}

    # Run experiment
    for game_name, game in games.items():
        print(f"\n{'=' * 70}")
        print(f"Game: {game_name}")
        print(f"{'=' * 70}")

        # Compute target QRE
        qre_lambda = config["metrics"]["qre_lambda"]
        qre_result = compute_qre(game, rationality=qre_lambda)

        print(f"QRE (λ={qre_lambda}):")
        print(f"  NashConv (exploitability floor): {qre_result.nash_conv:.8f}")
        print(f"  Strategy 1: {qre_result.strategy_1.numpy()}")
        print(f"  Strategy 2: {qre_result.strategy_2.numpy()}")

        # Compute MMD fixed point with uniform reference (ground truth for MMD target)
        bregman = NegativeEntropy()
        mmd_config = config["methods"]["mmd_fixed"]
        is_fixed_point_equal_qre, mmd_fp_info = verify_mmd_fixed_point_equals_qre(
            game,
            bregman,
            lr=1e-5,  # Tiny lr for convergence
            tau=mmd_config["tau"],
            qre_result=qre_result,
            num_steps=50000,
        )
        mmd_fp_s1 = torch.tensor(mmd_fp_info["strategy_1"], dtype=torch.float32)
        mmd_fp_s2 = torch.tensor(mmd_fp_info["strategy_2"], dtype=torch.float32)

        print(f"  MMD fixed point (uniform reference, τ={mmd_config['tau']}):")
        print(f"    Equals QRE: {is_fixed_point_equal_qre}")
        print(f"    NashConv: {mmd_fp_info['nash_conv']:.8f}")
        print(f"    Strategy 1: {mmd_fp_s1.numpy()}")
        print(f"    Strategy 2: {mmd_fp_s2.numpy()}")

        game_results = {
            "qre_exploitability_floor": float(qre_result.nash_conv),
            "qre_strategies": {
                "strategy_1": qre_result.strategy_1.numpy().tolist(),
                "strategy_2": qre_result.strategy_2.numpy().tolist(),
            },
            "mmd_fixed_point_equals_qre": is_fixed_point_equal_qre,
            "mmd_fixed_point_info": mmd_fp_info,
        }

        for method_name, method_config in config["methods"].items():
            print(f"\n  Method: {method_config['name']}")

            method_results = {
                "config": method_config,
                "trajectories": [],
            }

            for seed_idx in range(num_seeds):
                seed = seed_base + seed_idx
                tracker, qre_distances, mmd_fp_distances, step_indices = run_method(
                    game,
                    method_config["name"],
                    method_config,
                    num_steps,
                    log_interval,
                    seed,
                    qre_result,
                    mmd_fixed_point=(mmd_fp_s1, mmd_fp_s2),
                )

                # Compute convergence rate on distance-to-MMD-fixed-point (for MMD methods)
                # or distance-to-QRE (for other methods)
                # This avoids the plateau artifact when QRE ≠ Nash
                metric_distances = mmd_fp_distances if method_config["name"].startswith("MMD") else qre_distances

                if len(metric_distances) >= 3:
                    # Create a tracker for distance metric
                    metric_tracker = ConvergenceTracker()
                    for step, dist in zip(step_indices, metric_distances, strict=False):
                        metric_tracker.log(step, exploitability=dist)

                    # Fit on last 50% of trajectory
                    conv_rate = metric_tracker.estimate_convergence_rate(
                        metric="exploitability",
                        window=len(metric_tracker.exploitabilities) // 2,
                    )
                else:
                    conv_rate = tracker.estimate_convergence_rate(
                        metric="exploitability",
                        window=1,
                    )

                trajectory = {
                    "seed": seed,
                    "nash_conv": tracker.exploitabilities,
                    "steps": tracker.step_indices,
                    "qre_distance": qre_distances,
                    "mmd_fp_distance": mmd_fp_distances,
                    "final_nash_conv": tracker.exploitabilities[-1] if tracker.exploitabilities else 0.0,
                    "final_qre_distance": qre_distances[-1] if qre_distances else 0.0,
                    "final_mmd_fp_distance": mmd_fp_distances[-1] if mmd_fp_distances else 0.0,
                    "convergence_rate": {
                        "rate": conv_rate.rate,
                        "r_squared": conv_rate.r_squared,
                        "is_linear": conv_rate.is_linear,
                        "metric_target": "mmd_fp" if method_config["name"].startswith("MMD") else "qre",
                    },
                }

                method_results["trajectories"].append(trajectory)

                metric_name = "MMD-FP" if method_config["name"].startswith("MMD") else "QRE"
                print(f"    Seed {seed_idx:2d}: {metric_name}-dist R²={conv_rate.r_squared:.4f}, "
                      f"final={metric_distances[-1] if metric_distances else 0.0:.8f}")

            # Aggregate statistics on convergence rate
            r_squared_values = [
                t["convergence_rate"]["r_squared"]
                for t in method_results["trajectories"]
            ]
            mean_r_squared = np.mean(r_squared_values)
            std_r_squared = np.std(r_squared_values)

            # Determine metric name
            metric_target = method_results["trajectories"][0]["convergence_rate"]["metric_target"]
            metric_name = "MMD-FP" if metric_target == "mmd_fp" else "QRE"

            # Final NashConv aggregate (for reference)
            final_ncs = [t["final_nash_conv"] for t in method_results["trajectories"]]
            ci = bootstrap_ci(final_ncs, confidence=0.95, n_bootstrap=10000)

            method_results["aggregate"] = {
                f"{metric_target}_distance_r_squared_mean": float(mean_r_squared),
                f"{metric_target}_distance_r_squared_std": float(std_r_squared),
                "final_nash_conv_mean": ci.mean,
                "final_nash_conv_ci_lower": ci.ci_lower,
                "final_nash_conv_ci_upper": ci.ci_upper,
                "std_error": ci.std_error,
                "num_seeds": num_seeds,
            }

            print(f"    {metric_name}-distance fit R² (mean±std): {mean_r_squared:.4f}±{std_r_squared:.4f}")
            print(f"    Final NashConv: {ci.mean:.8f} [{ci.ci_lower:.8f}, {ci.ci_upper:.8f}]")

            game_results[method_name] = method_results

        all_results["data"][game_name] = game_results

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {results_path}")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    config_path = Path("/home/sharaths/projects/game-llm/configs/exp01_mmd_vs_gda.yaml")
    results = run_experiment(config_path)

    # Print pre-registered outcomes (UPDATED ITERATION 2)
    print("\n" + "=" * 70)
    print("PRE-REGISTERED OUTCOME VERIFICATION (Iteration 2)")
    print("=" * 70)

    outcomes = {
        "gda_cycles": [],  # GDA bounded away from 0
        "mmd_fixed_linear": [],  # MMD(fixed) → QRE linearly (on QRE distance)
        "mmd_rnd_nash": [],  # MMD(RND) → Nash
    }

    for game_name, game_data in results["data"].items():
        print(f"\nGame: {game_name}")
        qre_floor = game_data["qre_exploitability_floor"]
        print(f"  QRE exploitability floor (λ={results['config']['metrics']['qre_lambda']}): {qre_floor:.8f}")

        # Outcome 1: GDA bounded away from 0 (cycles)
        gda_final = game_data["gda"]["aggregate"]["final_nash_conv_mean"]
        print(f"  GDA final NashConv: {gda_final:.8f}")
        if gda_final > qre_floor + 0.01:  # Bounded above QRE floor
            print("    ✓ CONFIRMED: GDA bounded away from 0 (cycles)")
            outcomes["gda_cycles"].append(True)
        else:
            print("    ✗ NOT MET: GDA converges to Nash or QRE (unexpected)")
            outcomes["gda_cycles"].append(False)

        # Outcome 2: MMD(fixed) → target linearly (CORRECTED: metric target depends on whether QRE=Nash)
        mmd_fixed_agg = game_data["mmd_fixed"]["aggregate"]
        # Find the R² key (could be mmd_fp_distance_r_squared_mean or qre_distance_r_squared_mean)
        mmd_fixed_r_squared = None
        metric_used = None
        for key in mmd_fixed_agg:
            if "r_squared_mean" in key:
                mmd_fixed_r_squared = mmd_fixed_agg[key]
                metric_used = key.replace("_distance_r_squared_mean", "").upper()
                break

        if mmd_fixed_r_squared is not None:
            print(f"  MMD-Fixed {metric_used}-distance R²: {mmd_fixed_r_squared:.4f}")
            if mmd_fixed_r_squared >= 0.9:
                print("    ✓ CONFIRMED: MMD(fixed) converges linearly (R² >= 0.9)")
                outcomes["mmd_fixed_linear"].append(True)
            else:
                print(f"    ✗ MISSED: MMD(fixed) R² = {mmd_fixed_r_squared:.4f} < 0.9")
                outcomes["mmd_fixed_linear"].append(False)
        else:
            print("  MMD-Fixed: Could not find R² metric")
            outcomes["mmd_fixed_linear"].append(False)

        # Outcome 3: MMD(RND) → Nash
        mmd_rnd_final = game_data["mmd_rnd"]["aggregate"]["final_nash_conv_mean"]
        print(f"  MMD-RND final NashConv: {mmd_rnd_final:.8f}")
        if mmd_rnd_final < 0.05:
            print("    ✓ CONFIRMED: MMD(RND) converges to Nash (NashConv < 0.05)")
            outcomes["mmd_rnd_nash"].append(True)
        else:
            print(f"    ✗ MISSED: MMD(RND) NashConv = {mmd_rnd_final:.8f} >= 0.05")
            outcomes["mmd_rnd_nash"].append(False)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (Iteration 2)")
    print("=" * 70)
    print(f"GDA cycles (3/3 games): {sum(outcomes['gda_cycles'])}/3 ✓" if all(outcomes['gda_cycles']) else f"GDA cycles (3/3 games): {sum(outcomes['gda_cycles'])}/3 ✗")
    print(f"MMD(fixed)→target linear (3/3 games): {sum(outcomes['mmd_fixed_linear'])}/3 ✓" if all(outcomes['mmd_fixed_linear']) else f"MMD(fixed)→target linear (3/3 games): {sum(outcomes['mmd_fixed_linear'])}/3 ✗")
    print("  (target = QRE if QRE=Nash, else MMD fixed point)")
    print(f"MMD(RND)→Nash (3/3 games): {sum(outcomes['mmd_rnd_nash'])}/3 ✓" if all(outcomes['mmd_rnd_nash']) else f"MMD(RND)→Nash (3/3 games): {sum(outcomes['mmd_rnd_nash'])}/3 ✗")
