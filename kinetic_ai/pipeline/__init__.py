"""LLM construction pipeline abstraction.

Composable typed stages for the full arc: research, pretrain, posttrain,
distillation, evaluation, and serving. Each stage declares inputs, outputs,
machine placement, cost, and preconditions. Pipeline validates the DAG,
rejects circular dependencies and GPU conflicts, and generates execution plans.

Public API:
    Machine: Machine type specification.
    MachineType: Enum for GPU/CPU choices.
    Stage: Individual stage in the pipeline.
    StageType: Enum for research/pretrain/posttrain/distill/eval/serve.
    DataArtifact: Checkpoint/dataset/results specification.
    Pipeline: DAG container, validator, and execution planner.
    ExecutionPlan: Generated plan with topological order and costs.
    PipelineError: Exception for validation failures.
"""

from kinetic_ai.pipeline.dag import (
    ExecutionPlan,
    Pipeline,
    PipelineError,
)
from kinetic_ai.pipeline.stage import (
    DataArtifact,
    Machine,
    MachineType,
    Stage,
    StageType,
)

__all__ = [
    "DataArtifact",
    "ExecutionPlan",
    "Machine",
    "MachineType",
    "Pipeline",
    "PipelineError",
    "Stage",
    "StageType",
]
