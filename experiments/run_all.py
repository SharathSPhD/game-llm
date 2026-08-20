"""Experiment runner: generates all results for the paper.

Runs the following experiments:
    1. MMD convergence on canonical games (RPS, Matching Pennies)
    2. QRE path tracing (λ sweep)
    3. DEQ convergence: Anderson vs Broyden vs Picard
    4. Auction mechanism comparison: second-price vs weighted aggregation
    5. pcDEQ property verification
    6. Statistical comparison of convergence rates

Each experiment produces:
    - Raw data (JSON) for reproducibility
    - Convergence plots (PNG/SVG)
    - Statistical test results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from kinetic_ai.config import (
    DEQConfig,
    SolverType,
)
from kinetic_ai.eval.convergence import ConvergenceTracker
from kinetic_ai.eval.statistical import bootstrap_ci, paired_bootstrap_test, wilcoxon_signed_rank
from kinetic_ai.games.payoff import (
    coordination_game,
    matching_pennies,
    rock_paper_scissors,
)
from kinetic_ai.games.qre import nash_conv, qre_path
from kinetic_ai.optim.bregman import NegativeEntropy
from kinetic_ai.optim.mmd import mmd_strategy_update


def experiment_1_mmd_convergence(
    output_dir: Path,
    num_seeds: int = 10,
    num_steps: int = 1000,
) -> dict:
    """Experiment 1: MMD convergence to QRE on canonical games.

    Tests Theorem 1 from the paper: MMD with negative entropy mirror map
    converges linearly to the QRE.

    Args:
        output_dir: Where to save results.
        num_seeds: Number of random seeds for confidence intervals.
        num_steps: Training steps per seed.

    Returns:
        Results dictionary with convergence data.
    """
    print("=" * 60)
    print("Experiment 1: MMD Convergence on Canonical Games")
    print("=" * 60)

    games = {
        "matching_pennies": matching_pennies(),
        "rock_paper_scissors": rock_paper_scissors(),
        "coordination": coordination_game(),
    }

    bregman = NegativeEntropy()
    lr = 0.3
    tau_values = [0.01, 0.05, 0.1, 0.5]

    results: dict = {}

    for game_name, game in games.items():
        print(f"\n--- {game_name} ---")
        results[game_name] = {}

        for tau in tau_values:
            seed_results: list[dict] = []

            for seed in range(num_seeds):
                torch.manual_seed(seed)
                tracker = ConvergenceTracker()

                n1 = game.num_actions_1
                n2 = game.num_actions_2
                s1 = torch.softmax(torch.randn(n1), dim=-1)
                s2 = torch.softmax(torch.randn(n2), dim=-1)
                ref1 = torch.ones(n1) / n1
                ref2 = torch.ones(n2) / n2

                t0 = time.time()
                for step in range(num_steps):
                    g1 = game.utility_gradient(1, s1, s2)
                    g2 = game.utility_gradient(2, s2, s1)
                    s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
                    s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)
                    nc = nash_conv(game, s1, s2)
                    tracker.log(step, exploitability=nc, wall_time=time.time() - t0)

                rate = tracker.estimate_convergence_rate("exploitability")
                final_nc = tracker.exploitabilities[-1]

                seed_results.append({
                    "seed": seed,
                    "final_exploitability": final_nc,
                    "convergence_rate": rate.rate,
                    "r_squared": rate.r_squared,
                    "is_linear": rate.is_linear,
                    "wall_time": time.time() - t0,
                    "exploitabilities": tracker.exploitabilities,
                })

            # Statistical analysis across seeds
            final_exps = [r["final_exploitability"] for r in seed_results]
            rates = [r["convergence_rate"] for r in seed_results]

            ci_exp = bootstrap_ci(final_exps)
            ci_rate = bootstrap_ci(rates)

            results[game_name][f"tau_{tau}"] = {
                "seed_results": seed_results,
                "exploitability_ci": {
                    "mean": ci_exp.mean,
                    "lower": ci_exp.ci_lower,
                    "upper": ci_exp.ci_upper,
                },
                "rate_ci": {
                    "mean": ci_rate.mean,
                    "lower": ci_rate.ci_lower,
                    "upper": ci_rate.ci_upper,
                },
                "linear_convergence_fraction": sum(
                    1 for r in seed_results if r["is_linear"]
                ) / num_seeds,
            }

            print(
                f"  τ={tau:.2f}: NashConv={ci_exp.mean:.6f} "
                f"[{ci_exp.ci_lower:.6f}, {ci_exp.ci_upper:.6f}], "
                f"rate={ci_rate.mean:.4f}"
            )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp1_mmd_convergence.json", "w") as f:
        json.dump(_sanitize(results), f, indent=2)

    return results


def experiment_2_qre_path(output_dir: Path) -> dict:
    """Experiment 2: QRE path tracing (homotopy).

    Traces the QRE correspondence as λ varies from 0 to ∞,
    demonstrating the smooth path from uniform play to Nash.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: QRE Path Tracing")
    print("=" * 60)

    games = {
        "matching_pennies": matching_pennies(),
        "rock_paper_scissors": rock_paper_scissors(),
    }

    results: dict = {}
    for game_name, game in games.items():
        path = qre_path(game, max_iter=2000, tol=1e-10)

        path_data = []
        for r in path:
            path_data.append({
                "rationality": r.rationality,
                "strategy_1": r.strategy_1.tolist(),
                "strategy_2": r.strategy_2.tolist(),
                "nash_conv": r.nash_conv,
                "converged": r.converged,
                "iterations": r.iterations,
            })

        results[game_name] = path_data
        print(f"  {game_name}: traced {len(path)} QRE points, "
              f"final NashConv={path[-1].nash_conv:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp2_qre_path.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def experiment_3_deq_solvers(
    output_dir: Path,
    num_seeds: int = 10,
) -> dict:
    """Experiment 3: DEQ solver comparison.

    Compares Anderson acceleration, Broyden, and Picard iteration
    on convergence speed and accuracy.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: DEQ Solver Comparison")
    print("=" * 60)

    import torch.nn as nn

    from kinetic_ai.models.deq_layer import DEQLayer

    hidden_dim = 32
    results: dict = {"anderson": [], "broyden": [], "picard": []}

    for seed in range(num_seeds):
        torch.manual_seed(seed)

        # Create a shared transformation
        linear = nn.Linear(hidden_dim * 2, hidden_dim)
        with torch.no_grad():
            linear.weight.data *= 0.3
            linear.bias.data *= 0.1

        def transform(z: torch.Tensor, x: torch.Tensor, _lin=linear) -> torch.Tensor:
            return torch.tanh(_lin(torch.cat([z, x], dim=-1)))

        x = torch.randn(4, hidden_dim)

        for solver_name, solver_type in [
            ("anderson", SolverType.ANDERSON),
            ("broyden", SolverType.BROYDEN),
            ("picard", SolverType.PICARD),
        ]:
            config = DEQConfig(solver=solver_type, max_iter=100, tol=1e-6)
            deq = DEQLayer(transform, config)

            t0 = time.time()
            with torch.no_grad():
                z_star = deq(x)
            elapsed = time.time() - t0

            # Check fixed-point residual
            residual = torch.norm(transform(z_star, x) - z_star).item()

            results[solver_name].append({
                "seed": seed,
                "residual": residual,
                "wall_time": elapsed,
            })

    # Compare solvers statistically
    for solver_name in results:
        residuals = [r["residual"] for r in results[solver_name]]
        ci = bootstrap_ci(residuals)
        print(f"  {solver_name}: residual={ci.mean:.2e} [{ci.ci_lower:.2e}, {ci.ci_upper:.2e}]")

    # Pairwise comparison: Anderson vs Picard
    anderson_res = [r["residual"] for r in results["anderson"]]
    picard_res = [r["residual"] for r in results["picard"]]
    test = wilcoxon_signed_rank(anderson_res, picard_res)
    print(f"  Anderson vs Picard: p={test.p_value:.4f}, significant={test.significant}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp3_deq_solvers.json", "w") as f:
        json.dump(_sanitize(results), f, indent=2)

    return results


def experiment_4_statistical_summary(
    output_dir: Path,
    exp1_results: dict,
) -> dict:
    """Experiment 4: Statistical comparison across methods.

    Compares different τ values for MMD convergence using proper
    paired statistical tests with multiple testing correction.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Statistical Comparison Summary")
    print("=" * 60)

    from kinetic_ai.eval.statistical import holm_bonferroni_correction

    summary: dict = {}

    for game_name, game_results in exp1_results.items():
        tau_keys = list(game_results.keys())
        comparisons = []

        for i, key_a in enumerate(tau_keys):
            for key_b in tau_keys[i + 1:]:
                data_a = game_results[key_a]["seed_results"]
                data_b = game_results[key_b]["seed_results"]

                rates_a = [r["convergence_rate"] for r in data_a]
                rates_b = [r["convergence_rate"] for r in data_b]

                test = paired_bootstrap_test(rates_a, rates_b)
                comparisons.append({
                    "comparison": f"{key_a} vs {key_b}",
                    "p_value": test.p_value,
                    "effect_size": test.effect_size,
                    "significant": test.significant,
                    "mean_a": test.method_a_mean,
                    "mean_b": test.method_b_mean,
                })

        # Apply Holm-Bonferroni correction
        p_values = [c["p_value"] for c in comparisons]
        corrected = holm_bonferroni_correction(p_values)
        for c, sig in zip(comparisons, corrected, strict=False):
            c["significant_corrected"] = sig

        summary[game_name] = comparisons
        print(f"\n  {game_name}:")
        for c in comparisons:
            print(
                f"    {c['comparison']}: p={c['p_value']:.4f}, "
                f"d={c['effect_size']:.2f}, "
                f"sig={c['significant_corrected']}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp4_statistical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _sanitize(obj):
    """Recursively convert numpy/torch types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    output = Path("outputs/experiments")
    output.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Kinetic AI — Full Experiment Suite               ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    exp1 = experiment_1_mmd_convergence(output, num_seeds=10, num_steps=500)
    exp2 = experiment_2_qre_path(output)
    exp3 = experiment_3_deq_solvers(output, num_seeds=10)
    exp4 = experiment_4_statistical_summary(output, exp1)

    print("\n" + "=" * 60)
    print("All experiments complete. Results saved to:", output)
    print("=" * 60)
