"""Execution abstraction for job scheduling and GPU management.

Provides a protocol-based executor interface for local (GB10) and remote (RunPod)
execution, with built-in GPU lock management to enforce the hard rule:
"never two GPU jobs at once" (see CLAUDE.md).

Local execution uses a file-based GPU lock at research/memory/state.json;
remote execution via RunPod serverless endpoints respects the same semantics.
"""

from kinetic_ai.serve.executor import (
    Executor,
    JobInput,
    JobOutput,
    JobStatus,
    LocalExecutor,
    RunpodServerlessExecutor,
)

__all__ = [
    "Executor",
    "JobInput",
    "JobOutput",
    "JobStatus",
    "LocalExecutor",
    "RunpodServerlessExecutor",
]
