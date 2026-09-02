"""Tests for simulate.py reproducibility and seed handling.

Tests that the simulation script properly supports random seeding for
reproducibility across multiple runs, which is essential for the research
harness requirement of ≥3 md5-distinct seeds.
"""

import re
import subprocess
import sys
from pathlib import Path


class TestSimulateSeeding:
    """Tests for random seed support in simulate.py."""

    def test_simulate_deterministic_with_same_seed(self) -> None:
        """Phase 2 (DEQ agent) produces identical outputs with same seed."""
        outputs = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "simulate.py", "--seed", "42"],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            assert result.returncode == 0, f"simulate.py failed: {result.stderr}"
            # Extract "P(token 3):" values from Phase 2 output
            match = re.search(r"P\(token \d+\): ([\d.]+)", result.stdout)
            if match:
                outputs.append(float(match.group(1)))

        assert len(outputs) == 2, "Could not extract P(token) values from output"
        assert outputs[0] == outputs[1], (
            f"Non-deterministic with same seed: {outputs[0]} vs {outputs[1]}"
        )

    def test_simulate_different_seeds_produce_different_results(self) -> None:
        """Different seeds should produce different Phase 2 results."""
        outputs = []
        for seed in [42, 123]:
            result = subprocess.run(
                [sys.executable, "simulate.py", "--seed", str(seed)],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            assert result.returncode == 0, f"simulate.py failed: {result.stderr}"
            # Extract "Bid value:" from Phase 3 output
            match = re.search(r"Bid value: ([\d.]+)", result.stdout)
            if match:
                outputs.append(float(match.group(1)))

        assert len(outputs) == 2, "Could not extract bid values from output"
        assert outputs[0] != outputs[1], (
            f"Different seeds should produce different results: {outputs[0]} vs {outputs[1]}"
        )

    def test_simulate_accepts_seed_argument(self) -> None:
        """simulate.py should accept --seed argument."""
        result = subprocess.run(
            [sys.executable, "simulate.py", "--seed", "42"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, f"simulate.py failed with --seed: {result.stderr}"
        assert "Simulation complete" in result.stdout

    def test_simulate_runs_without_seed_argument(self) -> None:
        """simulate.py should still work without --seed argument (backward compat)."""
        result = subprocess.run(
            [sys.executable, "simulate.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=30,
        )
        assert result.returncode == 0, f"simulate.py failed without --seed: {result.stderr}"
        assert "Simulation complete" in result.stdout

    def test_phase1_nashconv_deterministic_across_seeds(self) -> None:
        """Phase 1 NashConv should be identical regardless of seed (it uses fixed init)."""
        nashconv_values = []
        for seed in [42, 123, 999]:
            result = subprocess.run(
                [sys.executable, "simulate.py", "--seed", str(seed)],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            assert result.returncode == 0, f"simulate.py failed: {result.stderr}"
            # Extract final NashConv from Phase 1
            match = re.search(r"Final.*?\n  NashConv: ([\d.]+)", result.stdout, re.DOTALL)
            if match:
                nashconv_values.append(float(match.group(1)))

        assert (
            len(nashconv_values) == 3
        ), f"Could not extract NashConv values (got {len(nashconv_values)})"
        # Phase 1 is deterministic - same seed produces same NashConv
        assert (
            nashconv_values[0] == nashconv_values[1] == nashconv_values[2]
        ), f"Phase 1 should be deterministic: {nashconv_values}"
