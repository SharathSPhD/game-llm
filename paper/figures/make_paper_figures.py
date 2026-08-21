#!/usr/bin/env python3
"""Publication-quality figures for the Kinetic AI paper.

Self-contained and reproducible: all numbers are FROZEN constants hardcoded
below (no checkpoint/JSON parsing). Emits vector PDFs into this directory.

Run:  python3 make_paper_figures.py

PLACEHOLDER: This script currently raises SystemExit and does not generate figures.
Replace PLACEHOLDER constants with validated findings from research/memory/findings.md
as experimental results are finalized. Each figure must be backed by a validated,
peer-reviewed finding with confidence intervals or error bars.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Design system: Okabe-Ito colorblind-safe palette
# --------------------------------------------------------------------------- #
C_TREATMENT = "#0072B2"   # blue   — treatment / primary / equilibrium
C_CONTROL = "#D55E00"     # vermillion — control / baseline
C_NEUTRAL = "#999999"     # grey   — neutral / auxiliary
C_POSITIVE = "#009E73"    # green  — positive / convergence
C_HIGHLIGHT = "#E69F00"   # amber  — highlight / critical
C_PURPLE = "#CC79A7"      # purple — secondary

HERE = os.path.dirname(os.path.abspath(__file__))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.titlepad": 9,
        }
    )


def despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def ygrid(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)


def save(fig, name: str) -> None:
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------- #
# Figure placeholders: each must be backed by validated findings
# --------------------------------------------------------------------------- #

def fig_convergence_mmd() -> None:
    """F1: MMD convergence of learned distribution to equilibrium."""
    # PLACEHOLDER: Replace with real data from validated findings
    # Expected: EqLM fixed-point iteration converges to equilibrium
    # with faster rate than baseline gradient descent.
    raise SystemExit(
        "figure not yet backed by validated findings"
    )


def fig_matrix_game() -> None:
    """F2: EqLM vs baseline on synthetic 2x2 matrix games."""
    # PLACEHOLDER: Replace with real data from validated findings
    # Expected: EqLM recovers QRE exactly; baseline converges slower.
    raise SystemExit(
        "figure not yet backed by validated findings"
    )


def fig_babylm_comparison() -> None:
    """F3: BabyLM pretraining: EqLM vs GPT-2 vs BERT."""
    # PLACEHOLDER: Replace with real data from validated findings
    # Expected: token-matched comparison on standard benchmarks.
    raise SystemExit(
        "figure not yet backed by validated findings"
    )


def fig_deq_memory() -> None:
    """F4: Implicit depth and memory: DEQ iterations vs explicit layers."""
    # PLACEHOLDER: Replace with real data from validated findings
    # Expected: implicit layers reduce memory at cost of convergence time.
    raise SystemExit(
        "figure not yet backed by validated findings"
    )


def fig_dpo_auction() -> None:
    """F5: Auction-based decoding: DPO vs MPO alignment."""
    # PLACEHOLDER: Replace with real data from validated findings
    # Expected: mechanism design view improves alignment efficiency.
    raise SystemExit(
        "figure not yet backed by validated findings"
    )


if __name__ == "__main__":
    apply_style()
    print("ERROR: figures not yet backed by validated findings.")
    print("See research/memory/findings.md for details.")
    sys.exit(1)
