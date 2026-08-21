"""Execution abstraction: protocol-based job submission and status tracking.

Defines the Executor protocol (job submission/status/result) and two implementations:
1. LocalExecutor: In-process with thread pool and file-based GPU lock
2. RunpodServerlessExecutor: Remote execution via RunPod serverless endpoints

GPU Lock Convention
===================
The GB10 enforces the hard rule: never two GPU jobs at once. The GPU lock is
stored in research/memory/state.json at the key "gpu_lock" (boolean):
  - LocalExecutor checks (read-only) and refuses submission if locked
  - Set by hand or by the EFE autoresearch loop (research/cycles/run.md)
  - The executor does NOT acquire/release the lock; that's the harness's job

RunPod Remote Execution
=======================
The RunpodServerlessExecutor stub shape hints at the eventual contract:
  1. User provides RUNPOD_API_KEY and a serverless endpoint URL
  2. Calls handler(job_input: JobInput) -> JobOutput on the remote endpoint
  3. Job input is JSON-serializable (game params, solver config, etc.)
  4. Result is returned in the same call (request-response; no queue polling)

Example remote handler (RunPod serverless contract):
    def handler(job_input: dict) -> dict:
        # Deserialize job_input to JobInput
        job = JobInput(**job_input)
        # Execute (e.g., solve an equilibrium)
        # Return dict matching JobOutput schema
        return {"id": job.id, "status": "completed", "result": {...}}
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests

# ─── Shared Models ───────────────────────────────────────────────────────────


@dataclass
class JobInput:
    """Job specification for executor.

    Attributes:
        id: Unique job identifier (generated if not provided).
        type: Job type: 'noop_demo', 'solve', 'qre_path', etc.
        params: Job-specific parameters (game type, solver config, etc.).
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class JobOutput:
    """Result from job execution.

    Attributes:
        id: The job ID.
        status: 'completed' or 'failed'.
        result: The actual result (varies by job type).
        error: Error message if status == 'failed'.
    """

    id: str
    status: str
    result: Any = None
    error: str | None = None


class JobStatus:
    """Job status type enumeration (string literals)."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── Protocol ────────────────────────────────────────────────────────────────


class Executor(ABC):
    """Abstract executor for job submission, status tracking, and result retrieval.

    Implementations:
        - LocalExecutor: In-process thread pool with GPU lock check
        - RunpodServerlessExecutor: Remote endpoint via HTTP
    """

    @abstractmethod
    def submit(self, job: JobInput) -> str:
        """Submit a job for execution.

        Args:
            job: Job specification.

        Returns:
            Job ID.

        Raises:
            RuntimeError: If GPU is locked (LocalExecutor only).
        """

    @abstractmethod
    def status(self, job_id: str) -> str:
        """Get job status.

        Returns:
            One of JobStatus constants: 'queued', 'running', 'completed', 'failed'.
        """

    @abstractmethod
    def result(self, job_id: str) -> JobOutput:
        """Retrieve job result.

        Must only be called after status() returns 'completed' or 'failed'.

        Returns:
            JobOutput with result or error.
        """


# ─── LocalExecutor ───────────────────────────────────────────────────────────


class LocalExecutor(Executor):
    """In-process executor with thread pool and file-based GPU lock.

    The GPU lock is read-only: LocalExecutor refuses to execute if
    research/memory/state.json has gpu_lock=true. It does not attempt
    to acquire or release the lock.

    Args:
        max_workers: Thread pool size (default 2; typically 1 for GPU jobs).
        state_file: Path to research/memory/state.json (GPU lock source).
    """

    def __init__(
        self,
        max_workers: int = 2,
        state_file: str | None = None,
    ) -> None:
        self.max_workers = max_workers
        if state_file is None:
            state_file = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "research",
                "memory",
                "state.json",
            )
        self.state_file = state_file
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _is_gpu_locked(self) -> bool:
        """Check if GPU is locked in state.json."""
        if not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file) as f:
                state: dict = json.load(f)
                return bool(state.get("gpu_lock", False))
        except (json.JSONDecodeError, OSError):
            return False

    def _run_job(self, job: JobInput) -> JobOutput:
        """Execute a single job (runs in thread pool)."""
        try:
            result = self._dispatch_job(job)
            return JobOutput(id=job.id, status=JobStatus.COMPLETED, result=result)
        except Exception as e:
            return JobOutput(id=job.id, status=JobStatus.FAILED, error=str(e))

    def _dispatch_job(self, job: JobInput) -> Any:
        """Dispatch to job-specific handler."""
        if job.type == "noop_demo":
            # Demo job: sleep 5s and return a message
            time.sleep(5)
            return {"message": "noop_demo completed"}
        else:
            raise ValueError(f"Unknown job type: {job.type}")

    def submit(self, job: JobInput) -> str:
        """Submit a job. Refuses if GPU is locked."""
        if self._is_gpu_locked():
            raise RuntimeError(
                "GPU is locked (gpu_lock=true in state.json). "
                "Wait for current job to complete."
            )
        with self._lock:
            self._jobs[job.id] = {
                "job": job,
                "status": JobStatus.QUEUED,
                "output": None,
            }
        # In production, this would submit to a thread pool executor
        # For now, run synchronously (simulating immediate execution)
        output = self._run_job(job)
        with self._lock:
            self._jobs[job.id]["status"] = output.status
            self._jobs[job.id]["output"] = output
        return job.id

    def status(self, job_id: str) -> str:
        """Get job status."""
        with self._lock:
            if job_id not in self._jobs:
                return JobStatus.FAILED
            return str(self._jobs[job_id]["status"])

    def result(self, job_id: str) -> JobOutput:
        """Retrieve job result."""
        with self._lock:
            if job_id not in self._jobs:
                return JobOutput(
                    id=job_id,
                    status=JobStatus.FAILED,
                    error="Job not found",
                )
            output = self._jobs[job_id]["output"]
            return output or JobOutput(
                id=job_id,
                status=JobStatus.FAILED,
                error="Result not available",
            )


# ─── RunpodServerlessExecutor ────────────────────────────────────────────────


class RunpodServerlessExecutor(Executor):
    """Remote executor for RunPod serverless endpoints.

    Submits jobs to a RunPod serverless endpoint via HTTP. The endpoint
    must implement the RunPod serverless handler contract:
        def handler(job_input: dict) -> dict:
            # job_input matches JobInput schema
            # returns dict matching JobOutput schema

    Remote Endpoint Reference
    ==========================
    The handler path is inferred from RUNPOD_ENDPOINT_URL env var:
        https://api.runpod.io/v1/{endpoint_id}/runsync
    or provided directly via endpoint_url param.

    Args:
        endpoint_url: Full URL to the RunPod serverless endpoint.
                      If None, constructs from RUNPOD_ENDPOINT_ID.
        api_key: RunPod API key (for authentication if needed).
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if endpoint_url is None:
            endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
            if not endpoint_id:
                raise ValueError(
                    "RUNPOD_ENDPOINT_ID not set. Provide endpoint_url or set env var."
                )
            endpoint_url = f"https://api.runpod.io/v1/{endpoint_id}/runsync"
        self.endpoint_url = endpoint_url
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._jobs: dict[str, JobOutput] = {}

    def submit(self, job: JobInput) -> str:
        """Submit job to remote endpoint."""
        payload = {
            "id": job.id,
            "type": job.type,
            "params": job.params,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,  # type: ignore[arg-type]
                headers=headers,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            output = JobOutput(
                id=data.get("id", job.id),
                status=data.get("status", JobStatus.COMPLETED),
                result=data.get("result"),
                error=data.get("error"),
            )
            self._jobs[job.id] = output
            return job.id
        except requests.RequestException as e:
            output = JobOutput(
                id=job.id,
                status=JobStatus.FAILED,
                error=f"Remote execution failed: {str(e)}",
            )
            self._jobs[job.id] = output
            return job.id

    def status(self, job_id: str) -> str:
        """Get job status from cache."""
        if job_id not in self._jobs:
            return JobStatus.FAILED
        return self._jobs[job_id].status

    def result(self, job_id: str) -> JobOutput:
        """Retrieve cached result."""
        if job_id not in self._jobs:
            return JobOutput(
                id=job_id,
                status=JobStatus.FAILED,
                error="Job not found",
            )
        return self._jobs[job_id]
