"""exp04_auction_truthfulness: Empirical regret of truthful bidding vs best misreport.

Pre-registered per SPEC 0003:
- Agents n in [3,5], 10 seeds x 200 auctions
- Random valuations in [0,1], vocab 50
- For each mechanism (SECOND_PRICE, WEIGHTED_AGGREGATION):
  - Compute empirical regret of truthful bidding vs best misreport
  - Over grid [0.25v, 0.5v, 0.75v, v, 1.25v, 1.5v, 2v]
  - Utility = valuation-weighted allocation minus payment
- Aggregate mean±CI regret per mechanism
- Figure: regret distribution per mechanism

Pre-registered findings:
  - second-price truthful-regret <= epsilon (truthfulness by Vickrey theory)
  - weighted aggregation reported as measured (document deviation magnitude)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from kinetic_ai.config import AuctionConfig, AuctionType
from kinetic_ai.eval.statistical import bootstrap_ci
from kinetic_ai.mechanisms.auctions import TokenAuction


@dataclass
class AuctionRegretResult:
    """Results from regret analysis of a single auction."""
    mechanism: str
    n_agents: int
    seed: int
    auction_idx: int
    agent_idx: int
    truthful_valuation: float
    truthful_utility: float
    best_misreport_bid: float
    best_misreport_utility: float
    regret: float  # max(misreport utilities) - truthful utility
    misreport_bids_tested: list[float]
    misreport_utilities: list[float]


def generate_random_auction(
    n_agents: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray]:
    """Generate a random auction with agent valuations and distributions.

    Args:
        n_agents: Number of agents
        vocab_size: Size of token vocabulary
        rng: Random number generator

    Returns:
        (valuations, distributions) where:
        - valuations: (n_agents,) random values in [0, 1]
        - distributions: (n_agents, vocab_size) softmax-normalized distributions
    """
    valuations = rng.uniform(0, 1, size=n_agents)
    logits = rng.normal(0, 1, size=(n_agents, vocab_size))
    distributions = torch.from_numpy(logits).float()
    distributions = F.softmax(distributions, dim=-1)

    return valuations, distributions.numpy()


def compute_allocation_and_payment(
    bids: NDArray,
    distributions: NDArray,
    mechanism: str,
    config: AuctionConfig,
) -> tuple[NDArray, NDArray]:
    """Compute allocation weights and payments for all agents.

    Args:
        bids: (n_agents,) agent bids
        distributions: (n_agents, vocab_size) agent distributions
        mechanism: "second_price" or "weighted_aggregation"
        config: Auction configuration

    Returns:
        (allocations, payments) where:
        - allocations: (n_agents,) bid-weighted allocation
        - payments: (n_agents,) payment per agent
    """
    bids_t = torch.from_numpy(bids).float()
    dists_t = torch.from_numpy(distributions).float()

    auction = TokenAuction(config)
    result = auction.run_auction(bids_t, dists_t)

    if mechanism == "second_price":
        # Allocation: winner gets 1.0, others get 0.0
        allocations = np.zeros(len(bids))
        if result.winner_id >= 0:
            allocations[result.winner_id] = 1.0
    else:  # weighted_aggregation
        # Allocation: softmax of bids (normalized bidding power)
        allocations = F.softmax(bids_t / max(config.aggregation_temp, 1e-8), dim=0).numpy()

    payments = result.payments.numpy()

    return allocations, payments


def compute_utility(
    valuation: float,
    allocation: float,
    payment: float,
) -> float:
    """Compute utility: valuation * allocation - payment.

    This is the consistent definition across both mechanisms.
    """
    return valuation * allocation - payment


def compute_regret_for_agent(
    agent_idx: int,
    valuations: NDArray,
    distributions: NDArray,
    mechanism: str,
    config: AuctionConfig,
    misreport_grid: list[float],
    rng: np.random.Generator,
) -> tuple[float, float, float, list[float], list[float]]:
    """Compute regret of truthful bidding vs best misreport for one agent.

    Returns:
        (truthful_utility, best_misreport_bid, best_misreport_utility,
         tested_bids, tested_utilities)
    """
    truthful_bid = valuations[agent_idx]

    # Truthful utility: all agents bid truthfully
    bids_truthful = valuations.copy()
    allocations, payments = compute_allocation_and_payment(
        bids_truthful, distributions, mechanism, config
    )
    truthful_utility = compute_utility(
        truthful_bid,
        allocations[agent_idx],
        payments[agent_idx]
    )

    # Test misreports: agent i deviates while others bid truthfully
    misreport_bids = [truthful_bid * factor for factor in misreport_grid]
    misreport_utilities = []

    for misreport_bid in misreport_bids:
        bids_misreport = valuations.copy()
        bids_misreport[agent_idx] = misreport_bid

        allocations, payments = compute_allocation_and_payment(
            bids_misreport, distributions, mechanism, config
        )
        utility = compute_utility(
            truthful_bid,  # Valuation is unchanged
            allocations[agent_idx],
            payments[agent_idx]
        )
        misreport_utilities.append(utility)

    best_misreport_idx = int(np.argmax(misreport_utilities))
    best_misreport_bid = misreport_bids[best_misreport_idx]
    best_misreport_utility = misreport_utilities[best_misreport_idx]

    # Regret: benefit of best misreport over truthful
    max(0, best_misreport_utility - truthful_utility)

    return (
        truthful_utility,
        best_misreport_bid,
        best_misreport_utility,
        misreport_bids,
        misreport_utilities,
    )


def run_experiment(config_path: str | Path) -> dict:
    """Run the full auction truthfulness experiment.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with all results and metadata
    """
    config_path = Path(config_path)

    # Load config
    import yaml
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    exp_config = config_dict["experiment"]
    auction_config_dict = config_dict["auction"]
    misreport_grid = config_dict["misreport_grid"]

    num_seeds = exp_config["num_seeds"]
    num_auctions = exp_config["num_auctions_per_seed"]
    agent_counts = exp_config["agent_counts"]
    vocab_size = exp_config["vocab_size"]

    output_dir = Path(config_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute config hash
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

    # Get git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=config_path.parent.parent,
            text=True
        ).strip()
    except Exception:
        git_hash = "unknown"

    # Collect all results
    all_results: list[AuctionRegretResult] = []

    # Run seeds
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)

        for n_agents in agent_counts:
            for auction_idx in range(num_auctions):
                # Generate random auction
                valuations, distributions = generate_random_auction(
                    n_agents, vocab_size, rng
                )

                # Test both mechanisms
                for mechanism_str in auction_config_dict["mechanisms"]:
                    auction_type = (
                        AuctionType.SECOND_PRICE
                        if mechanism_str == "second_price"
                        else AuctionType.WEIGHTED_AGGREGATION
                    )
                    auction_cfg = AuctionConfig(
                        auction_type=auction_type,
                        vocab_size=vocab_size,
                        reserve_price=auction_config_dict["reserve_price"],
                        aggregation_temp=auction_config_dict["aggregation_temp"],
                    )

                    # Compute regret for each agent
                    for agent_idx in range(n_agents):
                        (
                            truthful_util,
                            best_misreport_bid,
                            best_misreport_util,
                            tested_bids,
                            tested_utils,
                        ) = compute_regret_for_agent(
                            agent_idx,
                            valuations,
                            distributions,
                            mechanism_str,
                            auction_cfg,
                            misreport_grid,
                            rng,
                        )

                        regret = best_misreport_util - truthful_util

                        result = AuctionRegretResult(
                            mechanism=mechanism_str,
                            n_agents=n_agents,
                            seed=seed,
                            auction_idx=auction_idx,
                            agent_idx=agent_idx,
                            truthful_valuation=float(valuations[agent_idx]),
                            truthful_utility=float(truthful_util),
                            best_misreport_bid=float(best_misreport_bid),
                            best_misreport_utility=float(best_misreport_util),
                            regret=float(regret),
                            misreport_bids_tested=[float(b) for b in tested_bids],
                            misreport_utilities=[float(u) for u in tested_utils],
                        )
                        all_results.append(result)

    # Aggregate results
    results_by_mechanism_n = {}
    for mechanism in auction_config_dict["mechanisms"]:
        for n in agent_counts:
            key = f"{mechanism}_n{n}"
            regrets = [
                r.regret for r in all_results
                if r.mechanism == mechanism and r.n_agents == n
            ]
            ci = bootstrap_ci(regrets, confidence=0.95, n_bootstrap=10000)
            results_by_mechanism_n[key] = {
                "mechanism": mechanism,
                "n_agents": n,
                "regrets_count": len(regrets),
                "mean_regret": float(ci.mean),
                "regret_ci_lower": float(ci.ci_lower),
                "regret_ci_upper": float(ci.ci_upper),
                "regret_std_error": float(ci.std_error),
            }

    # Prepare summary for results.json
    summary = {
        "experiment": "exp04_auction_truthfulness",
        "config_file": str(config_path),
        "config_hash": config_hash,
        "git_commit": git_hash,
        "timestamp": str(np.datetime64('now')),
        "spec_version": "0003",
        "pre_registered": True,
        "configuration": {
            "num_seeds": num_seeds,
            "num_auctions_per_seed": num_auctions,
            "agent_counts": agent_counts,
            "vocab_size": vocab_size,
            "mechanisms": auction_config_dict["mechanisms"],
            "misreport_grid": misreport_grid,
            "utility_definition": config_dict.get("utility_definition", "valuation * allocation - payment"),
        },
        "results": results_by_mechanism_n,
        "raw_results_count": len(all_results),
        "pre_registered_hypotheses": {
            "H_second_price": {
                "claim": "Second-price truthful regret <= epsilon",
                "interpretation": "By Vickrey's theorem, truthful bidding is a dominant strategy in second-price auctions. Empirical regret should be ~0.",
                "actual_mean_regret": results_by_mechanism_n.get("second_price_n3", {}).get("mean_regret"),
            },
            "H_weighted_aggregation": {
                "claim": "Weighted aggregation regret reported as measured",
                "interpretation": "Weighted aggregation with VCG payments is NOT truthful. Report deviation magnitude honestly.",
                "actual_mean_regret_n3": results_by_mechanism_n.get("weighted_aggregation_n3", {}).get("mean_regret"),
                "actual_mean_regret_n5": results_by_mechanism_n.get("weighted_aggregation_n5", {}).get("mean_regret"),
            },
        }
    }

    # Write results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw results as CSV for analysis
    raw_csv = output_dir / "raw_results.csv"
    with open(raw_csv, "w") as f:
        # CSV header
        f.write("mechanism,n_agents,seed,auction_idx,agent_idx,truthful_valuation,"
                "truthful_utility,best_misreport_bid,best_misreport_utility,regret\n")
        for r in all_results:
            f.write(
                f"{r.mechanism},{r.n_agents},{r.seed},{r.auction_idx},"
                f"{r.agent_idx},{r.truthful_valuation:.6f},"
                f"{r.truthful_utility:.6f},{r.best_misreport_bid:.6f},"
                f"{r.best_misreport_utility:.6f},{r.regret:.6f}\n"
            )

    print(f"Results written to {results_file}")
    print(f"Raw results written to {raw_csv}")
    print(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/exp04_auction_truthfulness.yaml"
    run_experiment(config_file)
