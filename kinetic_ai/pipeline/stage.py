"""LLM construction pipeline stages.

Defines Stage specification and machine types for the full arc from research
through serving, with inputs/outputs, machine placement, cost, and preconditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MachineType(str, Enum):
    """Target machine types for stage execution.

    RTX 5090: 32GB VRAM, for training (pretrain, posttrain, distillation).
    GB10: 121GB unified memory, for evaluation and serving.
    CPU: CPU-only work (data prep, analysis, research).
    """

    GPU_32GB = "gpu_32gb"  # RTX 5090: training
    GPU_120GB = "gpu_120gb"  # GB10: eval, serving
    CPU = "cpu"  # Lightweight work


class StageType(str, Enum):
    """Types of stages in the LLM construction arc.

    RESEARCH: Hypothesis selection, probe design, result analysis.
    PRETRAIN: Language model pretraining on unlabeled data.
    POSTTRAIN: Post-training (SFT, DPO, preference optimization).
    DISTILL: Knowledge distillation (top-down or bottom-up).
    EVALUATE: Benchmark evaluation against the baseline ladder.
    SERVE: Deploy the model or ensemble as an API.
    """

    RESEARCH = "research"
    PRETRAIN = "pretrain"
    POSTTRAIN = "posttrain"
    DISTILL = "distill"
    EVALUATE = "eval"
    SERVE = "serve"


@dataclass
class Machine:
    """Machine specification for stage execution.

    Attributes:
        name: Unique identifier (e.g., 'rtx5090', 'gb10', 'cpu_worker').
        type: MachineType classification.
        hostname: Network hostname for remote execution.
        gpu_count: Number of GPUs (0 for CPU).
        memory_gb: Total memory available.
        supported_tasks: List of task types this machine can run.
    """

    name: str
    type: MachineType
    hostname: str
    gpu_count: int
    memory_gb: int
    supported_tasks: list[str] = field(default_factory=list)

    def is_available_for(self, task: str) -> bool:
        """Check if this machine can run a given task type."""
        return task in self.supported_tasks


@dataclass(frozen=True)
class DataArtifact:
    """Data artifact specification: model checkpoint, dataset, or results.

    Attributes:
        name: Unique identifier (e.g., 'base_model_checkpoint').
        type: Artifact type (e.g., 'model_checkpoint', 'dataset', 'metrics').
        size_gb: Approximate size in gigabytes.
        location: Physical location (e.g., 'rtx5090:/checkpoint/model.pt').
        required_by: Stages that consume this artifact.
    """

    name: str
    type: str
    size_gb: float
    location: str
    required_by: list[str] = field(default_factory=list)


@dataclass
class Stage:
    """Composable stage in the LLM construction pipeline.

    A stage declares its inputs, outputs, machine requirements, cost, and
    preconditions. Stages are composed into a DAG where outputs of one stage
    become inputs to later stages.

    Attributes:
        name: Unique stage identifier.
        stage_type: Classification (research, pretrain, posttrain, distill, eval, serve).
        machine_required: MachineType on which to run.
        inputs: Names of artifacts required as input.
        outputs: Names of artifacts produced as output.
        cost_gpu_hours: Approximate GPU compute cost (0 for non-GPU stages).
        estimated_duration_hours: Wall-clock time estimate.
        preconditions: Human-readable preconditions for execution.
        variant: Optional variant classification (e.g., 'top_down', 'bottom_up' for distill).
    """

    name: str
    stage_type: StageType
    machine_required: MachineType
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    cost_gpu_hours: float = 0.0
    estimated_duration_hours: float = 0.0
    preconditions: list[str] = field(default_factory=list)
    variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by stage name (unique identifier)."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Equality by stage name."""
        if not isinstance(other, Stage):
            return NotImplemented
        return self.name == other.name
