"""Kinetic Studio API tests (Capability 1: Runs).

Tests the new experiment job submission, template listing, log polling, and runs registry.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

# Set test auth token and results dir BEFORE importing app.server
os.environ["GATEWAY_SECRET"] = "test-secret"
os.environ["KINETIC_MOCK_EXPERIMENTS"] = "1"
test_results_dir = tempfile.mkdtemp(prefix="kinetic_studio_test_")
os.environ["RESULTS_DIR"] = test_results_dir

from app.server import app  # noqa: E402


@pytest.fixture(autouse=True)
def _studio_env(monkeypatch):
    """Env is read at request time; other test modules mutate it at import.

    Pin this module's env per-test so suite ordering cannot leak a different
    RESULTS_DIR (or disable mock mode) into these requests.
    """
    monkeypatch.setenv("GATEWAY_SECRET", "test-secret")
    monkeypatch.setenv("KINETIC_MOCK_EXPERIMENTS", "1")
    monkeypatch.setenv("RESULTS_DIR", test_results_dir)

client = TestClient(app)

AUTH_HEADERS = {"Authorization": "Bearer test-secret"}


class TestExperimentTemplates:
    """GET /api/experiments endpoint."""

    def test_list_templates(self) -> None:
        """List available experiment templates."""
        resp = client.get("/api/experiments", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        templates = data["templates"]
        assert len(templates) > 0

        # Check structure
        for tmpl in templates:
            assert "id" in tmpl
            assert "name" in tmpl
            assert "description" in tmpl
            assert "script" in tmpl
            # id should be in the ALLOWLIST
            assert tmpl["id"] in ["exp05_eqlm_pretrain", "exp08_solver_aware"]

    def test_templates_have_required_fields(self) -> None:
        """Each template should have all required fields."""
        resp = client.get("/api/experiments", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        templates = resp.json()["templates"]

        for tmpl in templates:
            assert isinstance(tmpl["id"], str)
            assert isinstance(tmpl["name"], str)
            assert isinstance(tmpl["description"], str)
            assert isinstance(tmpl["script"], str)
            assert len(tmpl["id"]) > 0
            assert len(tmpl["name"]) > 0
            assert len(tmpl["script"]) > 0


class TestExperimentJobSubmission:
    """POST /api/jobs with type='experiment'."""

    def test_submit_experiment_with_overrides(self) -> None:
        """Submit experiment job with valid overrides."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "experiment",
                "template_id": "exp05_eqlm_pretrain",
                "overrides": {
                    "training.num_steps": 10,
                    "training.seed": 123,
                },
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    def test_submit_experiment_invalid_template(self) -> None:
        """Invalid template_id is rejected."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "experiment",
                "template_id": "exp_invalid",
                "overrides": {},
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_submit_experiment_no_template(self) -> None:
        """Missing template_id is rejected."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "experiment",
                "overrides": {},
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_submit_experiment_unknown_override_key(self) -> None:
        """Unknown override key is rejected."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "experiment",
                "template_id": "exp05_eqlm_pretrain",
                "overrides": {
                    "unknown.key": 123,
                },
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400
        assert "Unknown override key" in resp.text

    def test_submit_experiment_override_out_of_range(self) -> None:
        """Override value out of range is rejected."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "experiment",
                "template_id": "exp05_eqlm_pretrain",
                "overrides": {
                    "training.num_steps": 50000,  # > 25000 max
                },
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_submit_experiment_override_wrong_type(self) -> None:
        """Override with wrong type is rejected."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "experiment",
                "template_id": "exp05_eqlm_pretrain",
                "overrides": {
                    "training.num_steps": "not_an_int",
                },
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_experiment_config_resolved_correctly(self) -> None:
        """Resolved config is written to job output dir."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "experiment",
                "template_id": "exp05_eqlm_pretrain",
                "overrides": {
                    "training.num_steps": 500,
                    "training.seed": 999,
                },
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Check that config.yaml exists in output dir (created immediately)
        job_output_dir = Path(test_results_dir) / "studio_runs" / job_id
        config_file = job_output_dir / "config.yaml"

        # Config file should exist immediately
        assert config_file.exists(), f"Config file not found at {config_file}"

        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        # Check that overrides were applied
        assert config.get("training", {}).get("num_steps") == 500
        assert config.get("training", {}).get("seed") == 999


class TestGPULocking:
    """GPU lock enforcement."""

    def test_submit_experiment_gpu_locked(self, tmp_path: Path) -> None:
        """Job submission is refused if GPU is locked."""
        # Create a fake state.json with gpu_lock=true
        state_file = tmp_path / "state.json"
        with open(state_file, "w") as f:
            json.dump({"gpu_lock": True}, f)

        # Create a new executor with this state file
        from kinetic_ai.serve.executor import LocalExecutor

        local_executor = LocalExecutor(state_file=str(state_file))

        # Try to submit a job
        from kinetic_ai.serve.executor import JobInput

        job = JobInput(type="noop_demo")
        with pytest.raises(RuntimeError, match="GPU is locked"):
            local_executor.submit(job)


class TestJobLog:
    """GET /api/jobs/{id}/log endpoint."""

    def test_get_job_log_nonexistent(self) -> None:
        """Log for nonexistent job returns empty."""
        resp = client.get(
            "/api/jobs/nonexistent-id/log",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["lines"] == []
        assert data["total_lines"] == 0
        assert data["offset"] == 0

    def test_get_job_log_with_offset(self) -> None:
        """Log polling respects offset parameter."""
        # Create a fake log file
        job_dir = Path(test_results_dir) / "studio_runs" / "test-job"
        job_dir.mkdir(parents=True, exist_ok=True)
        log_file = job_dir / "run.log"

        with open(log_file, "w") as f:
            for i in range(10):
                f.write(f"Line {i}\n")

        # Fetch all lines
        resp = client.get(
            "/api/jobs/test-job/log?offset=0",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_lines"] == 10
        assert len(data["lines"]) == 10
        assert data["lines"][0] == "Line 0"

        # Fetch with offset
        resp = client.get(
            "/api/jobs/test-job/log?offset=5",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["offset"] == 5
        assert len(data["lines"]) == 5
        assert data["lines"][0] == "Line 5"


class TestRunsRegistry:
    """GET /api/runs endpoint."""

    def test_get_runs_registry_empty(self) -> None:
        """Registry returns empty if no results found."""
        resp = client.get(
            "/api/runs",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_get_runs_registry_with_results(self) -> None:
        """Registry includes runs with results.json."""
        # Create a fake results.json
        run_dir = Path(test_results_dir) / "studio_runs" / "test-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        results_file = run_dir / "results.json"

        fake_results = {
            "experiment": "stub_exp",
            "config_hash": "abc123",
            "git_commit": "deadbeef",
            "metrics": {
                "final_loss": 2.5,
                "final_accuracy": 0.42,
            },
        }

        with open(results_file, "w") as f:
            json.dump(fake_results, f)

        # Fetch registry
        resp = client.get(
            "/api/runs",
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        runs = data["runs"]

        # Should include at least our test run
        assert any(r["experiment"] == "stub_exp" for r in runs)

        # Check structure of one run
        test_run = next(r for r in runs if r["experiment"] == "stub_exp")
        assert "dir" in test_run
        assert "config_hash" in test_run
        assert "git_commit" in test_run
        assert "metrics" in test_run
        assert test_run["config_hash"] == "abc123"
        assert test_run["metrics"]["final_loss"] == 2.5


class TestAuthRequirements:
    """Auth checks on studio endpoints."""

    def test_experiments_requires_auth(self) -> None:
        """GET /api/experiments requires Bearer auth."""
        resp = client.get("/api/experiments")
        assert resp.status_code == 401

    def test_experiments_rejects_bad_token(self) -> None:
        """GET /api/experiments rejects bad token."""
        resp = client.get(
            "/api/experiments",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_job_submit_requires_auth(self) -> None:
        """POST /api/jobs requires Bearer auth."""
        resp = client.post(
            "/api/jobs",
            json={"type": "experiment", "template_id": "exp05_eqlm_pretrain"},
        )
        assert resp.status_code == 401

    def test_job_log_requires_auth(self) -> None:
        """GET /api/jobs/{id}/log requires Bearer auth."""
        resp = client.get("/api/jobs/test-id/log")
        assert resp.status_code == 401

    def test_runs_registry_requires_auth(self) -> None:
        """GET /api/runs requires Bearer auth."""
        resp = client.get("/api/runs")
        assert resp.status_code == 401


class TestBackwardCompatibility:
    """Backward compatibility for generic job types."""

    def test_submit_generic_job(self) -> None:
        """Legacy noop_demo job still works."""
        resp = client.post(
            "/api/jobs",
            json={
                "type": "noop_demo",
                "params": {},
            },
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_get_generic_job_status(self) -> None:
        """Getting status of generic job still works."""
        # Submit a noop_demo job
        submit_resp = client.post(
            "/api/jobs",
            json={
                "type": "noop_demo",
                "params": {},
            },
            headers=AUTH_HEADERS,
        )
        assert submit_resp.status_code == 200
        job_id = submit_resp.json()["job_id"]

        # Get status
        status_resp = client.get(
            f"/api/jobs/{job_id}",
            headers=AUTH_HEADERS,
        )
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["queued", "running", "completed", "failed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
