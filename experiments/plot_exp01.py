#!/usr/bin/env python
"""Plotting script for Experiment 1: MMD vs GDA.

Generates:
  - Trajectory plot (log-y NashConv, one panel per game)
  - Final bar chart with CIs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito palette
COLORS = {
    "GDA": "#0072B2",       # Blue
    "MMD-Fixed": "#D55E00",  # Orange
    "MMD-RND": "#009E73",   # Green
}

def plot_trajectories(results: dict, output_dir: Path) -> None:
    """Plot NashConv trajectories for each game."""
    games = list(results["data"].keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, game_name in zip(axes, games, strict=False):
        game_data = results["data"][game_name]

        for method_name, method_config in results["config"]["methods"].items():
            method_label = method_config["name"]
            trajectories = game_data[method_name]["trajectories"]

            # Plot all seeds with some transparency
            for traj in trajectories:
                steps = np.array(traj["steps"])
                nash_conv = np.array(traj["nash_conv"])

                # Filter out zeros for log scale
                valid_idx = nash_conv > 1e-8
                steps_valid = steps[valid_idx]
                nash_conv_valid = nash_conv[valid_idx]

                ax.semilogy(
                    steps_valid,
                    nash_conv_valid,
                    color=COLORS[method_label],
                    alpha=0.2,
                    linewidth=0.8,
                )

            # Plot mean trajectory
            # Get common steps from all trajectories
            first_traj = trajectories[0]
            common_steps = first_traj["steps"]

            means_list = []
            for step in common_steps:
                values = []
                for traj in trajectories:
                    if step in traj["steps"]:
                        idx = traj["steps"].index(step)
                        values.append(traj["nash_conv"][idx])
                if values:
                    means_list.append(np.mean(values))

            # Filter for log scale
            valid_idx = np.array(means_list) > 1e-8
            steps_valid = np.array(common_steps)[valid_idx]
            means_valid = np.array(means_list)[valid_idx]

            ax.semilogy(
                steps_valid,
                means_valid,
                color=COLORS[method_label],
                label=method_label,
                linewidth=2.5,
            )

        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("NashConv", fontsize=11)
        ax.set_title(game_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "fig_trajectories.pdf"
    plt.savefig(output_path, format="pdf", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_final_bars(results: dict, output_dir: Path) -> None:
    """Plot final NashConv with confidence intervals."""
    games = list(results["data"].keys())
    methods = list(results["config"]["methods"].keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x_positions = np.arange(len(methods))
    bar_width = 0.6

    for ax, game_name in zip(axes, games, strict=False):
        game_data = results["data"][game_name]

        means = []
        ci_lowers = []
        ci_uppers = []

        for method_name, _method_config in results["config"]["methods"].items():
            agg = game_data[method_name]["aggregate"]
            means.append(agg["final_nash_conv_mean"])
            ci_lowers.append(agg["final_nash_conv_mean"] - agg["final_nash_conv_ci_lower"])
            ci_uppers.append(agg["final_nash_conv_ci_upper"] - agg["final_nash_conv_mean"])

        # Plot bars
        method_labels = [results["config"]["methods"][m]["name"] for m in methods]
        colors = [COLORS[label] for label in method_labels]

        ax.bar(
            x_positions,
            means,
            bar_width,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            capsize=5,
        )

        # Add error bars
        ax.errorbar(
            x_positions,
            means,
            yerr=[ci_lowers, ci_uppers],
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=1.5,
            alpha=0.7,
        )

        # Add value labels on bars
        for i, (mean, _ci_l, ci_u) in enumerate(zip(means, ci_lowers, ci_uppers, strict=False)):
            ax.text(
                i,
                mean + ci_u + max(means) * 0.05,
                f"{mean:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(method_labels, fontsize=10)
        ax.set_ylabel("Final NashConv", fontsize=11)
        ax.set_title(game_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Use log scale if values span multiple orders of magnitude
        pos = [m for m in means if m > 0]
        if max(means) > 0 and pos and min(pos) > 0 and max(means) / min(pos) > 100:
            ax.set_yscale("log")

    plt.tight_layout()
    output_path = output_dir / "fig_final_bars.pdf"
    plt.savefig(output_path, format="pdf", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    results_path = Path("/home/sharaths/projects/game-llm/results/exp01_mmd_vs_gda/results.json")
    output_dir = results_path.parent

    with open(results_path) as f:
        results = json.load(f)

    print("Generating plots...")
    plot_trajectories(results, output_dir)
    plot_final_bars(results, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
