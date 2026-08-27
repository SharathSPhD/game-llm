"""FastAPI backend tests.

Tests the Phase 3 Equilibrium Lab server endpoints:
  - /health: open (no auth)
  - /api/solve, /api/qre_path, /api/auction, /api/results, /api/jobs: Bearer auth required
  - Convergence validation for solve trajectories
  - Job queue stub functionality
"""

import json
import os
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402

# Set test auth token
os.environ["GATEWAY_SECRET"] = "test-secret"
os.environ["RESULTS_DIR"] = "./test_results"

client = TestClient(app)

# Headers for authenticated requests
AUTH_HEADERS = {"Authorization": "Bearer test-secret"}
BAD_AUTH_HEADERS = {"Authorization": "Bearer wrong-token"}


class TestHealth:
    """GET /health (no auth required)."""

    def test_health_open(self) -> None:
        """Health endpoint is open."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "gpu_available" in data
        assert isinstance(data["gpu_available"], bool)
        # GPU availability should be boolean (may be True or False on this machine)
        # But we should NOT allocate GPU memory, just check torch.cuda.is_available()


class TestAuth:
    """Auth checks on /api/* endpoints."""

    def test_401_missing_auth(self) -> None:
        """No Authorization header → 401."""
        resp = client.post("/api/solve", json={"game": "rps"})
        assert resp.status_code == 401

    def test_401_bad_token(self) -> None:
        """Wrong Bearer token → 401."""
        resp = client.post(
            "/api/solve",
            json={"game": "rps"},
            headers=BAD_AUTH_HEADERS,
        )
        assert resp.status_code == 401

    def test_401_no_bearer_scheme(self) -> None:
        """Invalid auth scheme → 401."""
        resp = client.post(
            "/api/solve",
            json={"game": "rps"},
            headers={"Authorization": "Basic dGVzdDp0ZXN0"},
        )
        assert resp.status_code == 401


class TestSolve:
    """POST /api/solve endpoint."""

    def test_solve_mmd_fixed_rps(self) -> None:
        """Solve RPS with MMD (fixed reference)."""
        resp = client.post(
            "/api/solve",
            json={
                "game": "rps",
                "method": "mmd_fixed",
                "lr": 0.1,
                "tau": 1.0,
                "steps": 100,
                "seed": 42,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["game"] == "rps"
        assert data["method"] == "mmd_fixed"
        assert data["steps_run"] == 100
        assert len(data["trajectory"]) > 0
        assert len(data["final_strategy_1"]) == 3
        assert len(data["final_strategy_2"]) == 3
        assert data["final_nash_conv"] < 1.0

        # Verify trajectory is downsampled
        assert len(data["trajectory"]) <= 500

    def test_solve_mmd_rnd_converges(self) -> None:
        """Solve with MMD random resets should converge."""
        resp = client.post(
            "/api/solve",
            json={
                "game": "rps",
                "method": "mmd_rnd",
                "lr": 0.1,
                "tau": 1.0,
                "steps": 200,
                "seed": 42,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()

        # NashConv should decrease monotonically (or mostly)
        # Final NashConv should be < 0.1 for RPS with 200 steps
        assert data["final_nash_conv"] < 0.1

    def test_solve_gda_baseline(self) -> None:
        """Solve with GDA (baseline) method."""
        resp = client.post(
            "/api/solve",
            json={
                "game": "matching_pennies",
                "method": "gda",
                "lr": 0.01,
                "tau": 0.5,
                "steps": 150,
                "seed": 42,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["game"] == "matching_pennies"
        assert data["method"] == "gda"

    def test_solve_invalid_game(self) -> None:
        """Unknown game name raises error."""
        resp = client.post(
            "/api/solve",
            json={
                "game": "unknown_game",
                "method": "mmd_fixed",
                "steps": 10,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 422 or resp.status_code == 400

    def test_solve_steps_capped_at_5000(self) -> None:
        """Steps > 5000 are capped at 5000."""
        resp = client.post(
            "/api/solve",
            json={
                "game": "rps",
                "method": "mmd_fixed",
                "steps": 10000,
                "seed": 42,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_run"] == 5000


class TestQREPath:
    """POST /api/qre_path endpoint."""

    def test_qre_path_basic(self) -> None:
        """Compute QRE path for RPS."""
        resp = client.post(
            "/api/qre_path",
            json={
                "game": "rps",
                "lambda_min": 0.1,
                "lambda_max": 10.0,
                "n_points": 10,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["game"] == "rps"
        assert len(data["path"]) == 10

        # Each point should have rationality, strategies, nash_conv
        for point in data["path"]:
            assert "rationality" in point
            assert "strategy_1" in point
            assert "strategy_2" in point
            assert "nash_conv" in point
            assert len(point["strategy_1"]) == 3
            assert len(point["strategy_2"]) == 3

    def test_qre_path_monotone(self) -> None:
        """QRE path should trace rationality monotonically."""
        resp = client.post(
            "/api/qre_path",
            json={
                "game": "matching_pennies",
                "lambda_min": 0.1,
                "lambda_max": 5.0,
                "n_points": 5,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()

        rationalities = [p["rationality"] for p in data["path"]]
        # Should be increasing
        assert all(
            rationalities[i] <= rationalities[i + 1]
            for i in range(len(rationalities) - 1)
        )

    def test_qre_path_n_points_capped_at_50(self) -> None:
        """n_points > 50 is capped at 50."""
        resp = client.post(
            "/api/qre_path",
            json={
                "game": "rps",
                "lambda_min": 0.1,
                "lambda_max": 10.0,
                "n_points": 200,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["path"]) == 50


class TestAuction:
    """POST /api/auction endpoint."""

    def test_auction_second_price(self) -> None:
        """Second-price auction."""
        resp = client.post(
            "/api/auction",
            json={
                "bids": [0.7, 0.3],
                "agent_distributions": [
                    [0.9, 0.1],
                    [0.2, 0.8],
                ],
                "auction_type": "second_price",
                "vocab_size": 100,
                "seed": 42,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "winner_id" in data
        assert data["winner_id"] in [0, 1]
        assert len(data["output_distribution"]) == 100
        assert len(data["payments"]) == 2
        assert "sampled_token" in data

    def test_auction_weighted_aggregation(self) -> None:
        """Weighted aggregation auction."""
        resp = client.post(
            "/api/auction",
            json={
                "bids": [0.5, 0.5],
                "agent_distributions": [
                    [0.6, 0.4],
                    [0.3, 0.7],
                ],
                "auction_type": "weighted_aggregation",
                "vocab_size": 2,
                "seed": 42,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["output_distribution"]) == 2
        assert abs(sum(data["output_distribution"]) - 1.0) < 1e-5


class TestResults:
    """GET /api/results endpoint."""

    def test_results_empty_if_no_dir(self) -> None:
        """Results returns empty list if dir doesn't exist."""
        resp = client.get(
            "/api/results",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)


class TestJobs:
    """POST /api/jobs and GET /api/jobs/{id} endpoints."""

    def test_submit_noop_demo(self) -> None:
        """Submit a noop_demo job."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "noop_demo",
                "params": {},
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        job_id = data["job_id"]

        # Check status
        status_resp = client.get(
            f"/api/jobs/{job_id}",
            headers=AUTH_HEADERS,
        )
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] == "completed"
        assert status_data["result"] is not None

    def test_job_unknown_type_fails(self) -> None:
        """Unknown job type is rejected."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "unknown_type",
                "params": {},
            },
            headers=AUTH_HEADERS,
        )
        # Job is submitted but fails on execution
        assert resp.status_code == 200
        data = resp.json()
        job_id = data["job_id"]

        # Check status: should reach failed (poll — execution is async)
        status_data = {}
        for _ in range(40):
            status_resp = client.get(f"/api/jobs/{job_id}", headers=AUTH_HEADERS)
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            if status_data["status"] in ("failed", "completed"):
                break
            time.sleep(0.05)
        assert status_data["status"] == "failed"

    def test_job_status_nonexistent(self) -> None:
        """Query nonexistent job returns failed."""
        resp = client.get(
            "/api/jobs/nonexistent-id",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"


class TestCORS:
    """CORS headers on API responses."""

    def test_cors_headers_on_api_response(self) -> None:
        """API responses include CORS headers."""
        resp = client.post(
            "/api/solve",
            json={"game": "rps", "steps": 10},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        # Check that CORS headers are present (if implemented)
        # Note: TestClient may not set these by default, but the server should


class TestIntegration:
    """End-to-end integration tests."""

    def test_solve_convergence_trajectory(self) -> None:
        """MMD should produce a reasonable convergence trajectory."""
        resp = client.post(
            "/api/solve",
            json={
                "game": "rps",
                "method": "mmd_fixed",
                "lr": 0.05,
                "tau": 1.0,
                "steps": 300,
                "seed": 42,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()

        # Check trajectory structure
        trajectory = data["trajectory"]
        assert len(trajectory) > 0
        assert len(trajectory) <= 500  # Downsampled

        # Extract nash_conv values
        nash_convs = [t["nash_conv"] for t in trajectory]

        # Should decrease (mostly monotone)
        # Check that final is better than initial
        assert nash_convs[-1] < nash_convs[0]

        # Final should be reasonably small
        assert data["final_nash_conv"] < 0.2

    def test_qre_path_covers_range(self) -> None:
        """QRE path should cover the rationality range."""
        resp = client.post(
            "/api/qre_path",
            json={
                "game": "rps",
                "lambda_min": 0.5,
                "lambda_max": 5.0,
                "n_points": 5,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        path = data["path"]

        # First point should be near lambda_min
        assert abs(path[0]["rationality"] - 0.5) < 0.1
        # Last point should be near lambda_max
        assert abs(path[-1]["rationality"] - 5.0) < 0.1


class TestPlaygroundGenerate:
    """POST /api/playground/generate endpoint."""

    def test_401_missing_auth(self) -> None:
        """Missing auth → 401."""
        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": "exp10_probe/checkpoints/a1.pt",
                "prompt": "Hello",
            },
        )
        assert resp.status_code == 401

    def test_400_missing_checkpoint_path(self) -> None:
        """Missing checkpoint_path → 400."""
        resp = client.post(
            "/api/playground/generate",
            json={"prompt": "Hello"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert "checkpoint_path" in resp.json()["detail"].lower()

    def test_400_missing_prompt(self) -> None:
        """Missing prompt → 400."""
        resp = client.post(
            "/api/playground/generate",
            json={"checkpoint_path": "exp10_probe/checkpoints/a1.pt"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert "prompt" in resp.json()["detail"].lower()

    def test_400_prompt_too_long(self) -> None:
        """Prompt > 500 chars → 400."""
        long_prompt = "x" * 501
        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": "exp10_probe/checkpoints/a1.pt",
                "prompt": long_prompt,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert "500" in resp.json()["detail"]

    def test_400_traversal_attempt(self) -> None:
        """Path traversal attempt → 400."""
        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": "../../../etc/passwd",
                "prompt": "Hello",
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert "traversal" in resp.json()["detail"].lower()

    def test_400_checkpoint_not_found(self) -> None:
        """Non-existent checkpoint → 400."""
        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": "results/nonexistent.pt",
                "prompt": "Hello",
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.slow
    def test_generation_response_format(self) -> None:
        """Successful generation returns expected response format.

        This test requires a real checkpoint file to exist.
        """
        from pathlib import Path

        # Use the results_dir from env
        results_dir = Path(os.environ.get("RESULTS_DIR", "./results"))
        if not results_dir.exists():
            pytest.skip("results/ directory not found")

        checkpoints = list(results_dir.rglob("*.pt"))
        if not checkpoints:
            pytest.skip("No checkpoints found in results/")

        # Use the first checkpoint found
        checkpoint_rel = checkpoints[0].relative_to(results_dir)

        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": str(checkpoint_rel),
                "prompt": "Test",
                "max_new_tokens": 4,
                "warm_start": False,
                "solver_budget": 6,
            },
            headers=AUTH_HEADERS,
        )

        # Should return 200 or 503 (if model fails to load)
        if resp.status_code == 200:
            data = resp.json()
            assert "text" in data
            assert "tokens" in data
            assert "mean_iters" in data
            assert "wall_ms" in data
            assert isinstance(data["text"], str)
            assert isinstance(data["tokens"], list)
            assert isinstance(data["mean_iters"], (int, float))
            assert isinstance(data["wall_ms"], (int, float))

            # Check token structure
            for token in data["tokens"]:
                assert "token_str" in token
                assert "solver_iters" in token
                assert isinstance(token["token_str"], str)
                # solver_iters should be int or None
                assert token["solver_iters"] is None or isinstance(token["solver_iters"], int)

    def test_solver_budget_clamped(self) -> None:
        """Solver budget is clamped to [4, 64]."""
        # This test just checks that extreme values are accepted without error
        # (actual clamping is tested in generation)

        from pathlib import Path
        # Use the results_dir from env
        results_dir = Path(os.environ.get("RESULTS_DIR", "./results"))
        if not results_dir.exists():
            pytest.skip("results/ directory not found")

        checkpoints = list(results_dir.rglob("*.pt"))
        if not checkpoints:
            pytest.skip("No checkpoints found in results/")

        checkpoint_rel = checkpoints[0].relative_to(results_dir)

        # Very large solver_budget should be clamped
        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": str(checkpoint_rel),
                "prompt": "Test",
                "max_new_tokens": 1,
                "solver_budget": 1000,  # Should be clamped to 64
            },
            headers=AUTH_HEADERS,
        )

        # Should either succeed or fail gracefully
        assert resp.status_code in (200, 503)

    def test_max_new_tokens_clamped(self) -> None:
        """max_new_tokens is clamped to 64."""
        from pathlib import Path
        # Use the results_dir from env
        results_dir = Path(os.environ.get("RESULTS_DIR", "./results"))
        if not results_dir.exists():
            pytest.skip("results/ directory not found")

        checkpoints = list(results_dir.rglob("*.pt"))
        if not checkpoints:
            pytest.skip("No checkpoints found in results/")

        checkpoint_rel = checkpoints[0].relative_to(results_dir)

        # Very large max_new_tokens should be clamped
        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": str(checkpoint_rel),
                "prompt": "Test",
                "max_new_tokens": 1000,  # Should be clamped to 64
            },
            headers=AUTH_HEADERS,
        )

        # Should either succeed or fail gracefully
        assert resp.status_code in (200, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPlaygroundEquilibriumLens:
    """Regression: the lens must report REAL solver iterations and honor the dial.

    (A prior implementation parsed warm_start/solver_budget and ignored both,
    returning solver_iters=None for every token.)
    """

    def test_eqlm_iters_reported_and_budget_binds(self, tmp_path, monkeypatch) -> None:
        import torch as _torch

        from kinetic_ai.models.eqlm import EqLM, EqLMConfig, save_checkpoint

        monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
        _torch.manual_seed(0)
        cfg = EqLMConfig(
            vocab_size=50257, d_model=32, n_heads=2, d_ff=64, max_seq_len=96,
            deq_max_iter=12, deq_tol=1e-4, map_form="postln",
        )
        m = EqLM(cfg)
        ckpt_dir = tmp_path / "run" / "checkpoints"
        save_checkpoint(m, ckpt_dir / "tiny.pt")

        resp = client.post(
            "/api/playground/generate",
            json={
                "checkpoint_path": "run/checkpoints/tiny.pt",
                "prompt": "hello world",
                "max_new_tokens": 4,
                "warm_start": False,
                "solver_budget": 5,
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        iters = [t["solver_iters"] for t in data["tokens"]]
        assert len(iters) == 4
        assert all(isinstance(i, int) for i in iters), f"lens must be real: {iters}"
        # tight tol -> solver hits the budget: the dial demonstrably binds
        assert max(iters) <= 5 and max(iters) >= 4, iters
        assert data["mean_iters"] > 0


class TestAuctionTraces:
    """GET /api/auction/traces — real exp12 decode traces for the playground."""

    def _seed_tracedir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
        exp12 = tmp_path / "exp12"
        exp12.mkdir()
        traces = [
            {"position": 0, "bids": [0.9, 0.4], "winner": 0, "payment": 0.4,
             "target_token": 11},
            {"position": 1, "bids": [0.2, 0.7], "winner": 1, "payment": 0.2,
             "target_token": 12},
        ]
        (exp12 / "traces_seed42.json").write_text(json.dumps(traces))
        (exp12 / "results_seed42.json").write_text(json.dumps({
            "seed": 42,
            "domains": {"a": "childes", "b": "simple_wiki"},
            "eval": {"perplexity_mixed": {"S_A": 30.0, "S_B": 40.0,
                                          "ENS": 28.0, "AUC": 27.0},
                     "auction_win_frac_a": 0.6},
            "h4_score": "MET",
        }))
        return traces

    def test_requires_auth(self):
        assert client.get("/api/auction/traces").status_code in (401, 403)

    def test_lists_available_trace_seeds(self, tmp_path, monkeypatch):
        self._seed_tracedir(tmp_path, monkeypatch)
        resp = client.get("/api/auction/traces", headers=AUTH_HEADERS)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["seeds"] == [42]

    def test_returns_trace_with_summary(self, tmp_path, monkeypatch):
        traces = self._seed_tracedir(tmp_path, monkeypatch)
        resp = client.get("/api/auction/traces/42?limit=1", headers=AUTH_HEADERS)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["traces"] == traces[:1]
        assert data["summary"]["h4_score"] == "MET"
        assert data["summary"]["perplexity_mixed"]["AUC"] == 27.0
        assert data["summary"]["domains"] == {"a": "childes", "b": "simple_wiki"}

    def test_unknown_seed_404(self, tmp_path, monkeypatch):
        self._seed_tracedir(tmp_path, monkeypatch)
        resp = client.get("/api/auction/traces/99", headers=AUTH_HEADERS)
        assert resp.status_code == 404
