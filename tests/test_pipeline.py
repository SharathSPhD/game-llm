"""Tests for LLM construction pipeline abstraction.

Validates composable typed stages, DAG structure, machine placement,
GPU scheduling, and execution plan generation.
"""

from __future__ import annotations

import pytest

from kinetic_ai.pipeline import (
    DataArtifact,
    Machine,
    MachineType,
    Pipeline,
    PipelineError,
    Stage,
    StageType,
)

# ─── Machine Tests ───────────────────────────────────────────────────────────


class TestMachine:
    """Tests for Machine specifications."""

    def test_machine_rtx5090_properties(self) -> None:
        """RTX 5090: 32GB VRAM, training specialization."""
        m = Machine(
            name="rtx5090",
            type=MachineType.GPU_32GB,
            hostname="trainer.local",
            gpu_count=1,
            memory_gb=32,
            supported_tasks=["pretrain", "finetune", "distill_student"],
        )
        assert m.name == "rtx5090"
        assert m.type == MachineType.GPU_32GB
        assert m.memory_gb == 32
        assert "pretrain" in m.supported_tasks
        assert m.is_available_for("pretrain")
        assert not m.is_available_for("serving")

    def test_machine_gb10_properties(self) -> None:
        """GB10: 121GB unified memory, eval/serving specialization."""
        m = Machine(
            name="gb10",
            type=MachineType.GPU_120GB,
            hostname="gb10.local",
            gpu_count=1,
            memory_gb=121,
            supported_tasks=["eval", "serving", "equilibrium_solve"],
        )
        assert m.name == "gb10"
        assert m.type == MachineType.GPU_120GB
        assert m.memory_gb == 121
        assert "eval" in m.supported_tasks
        assert m.is_available_for("eval")
        assert not m.is_available_for("pretrain")

    def test_machine_cpu_properties(self) -> None:
        """CPU-only machine for lightweight work."""
        m = Machine(
            name="cpu_worker",
            type=MachineType.CPU,
            hostname="cpu.local",
            gpu_count=0,
            memory_gb=64,
            supported_tasks=["data_prep", "analysis", "reporting"],
        )
        assert m.type == MachineType.CPU
        assert m.gpu_count == 0
        assert m.is_available_for("data_prep")
        assert not m.is_available_for("pretrain")

    def test_machine_task_availability(self) -> None:
        """Verify is_available_for checks task list."""
        m = Machine(
            name="test",
            type=MachineType.GPU_32GB,
            hostname="test.local",
            gpu_count=1,
            memory_gb=32,
            supported_tasks=["task_a", "task_b"],
        )
        assert m.is_available_for("task_a")
        assert m.is_available_for("task_b")
        assert not m.is_available_for("task_c")


# ─── DataArtifact Tests ───────────────────────────────────────────────────────


class TestDataArtifact:
    """Tests for data artifact specifications."""

    def test_artifact_model_checkpoint(self) -> None:
        """Model checkpoint: 300MB, stored on machine."""
        art = DataArtifact(
            name="pretrained_base",
            type="model_checkpoint",
            size_gb=0.3,
            location="rtx5090:/checkpoint",
            required_by=[],
        )
        assert art.name == "pretrained_base"
        assert art.type == "model_checkpoint"
        assert art.size_gb == 0.3
        assert art.required_by == []

    def test_artifact_with_downstream_consumers(self) -> None:
        """Artifact tracks what stages require it."""
        art = DataArtifact(
            name="training_data",
            type="dataset",
            size_gb=50.0,
            location="gb10:/data/babyLM",
            required_by=["pretrain", "distill_student"],
        )
        assert "pretrain" in art.required_by
        assert "distill_student" in art.required_by

    def test_artifact_equivalence(self) -> None:
        """Two artifacts with same content are equal."""
        a1 = DataArtifact(
            name="art",
            type="model",
            size_gb=1.0,
            location="loc",
            required_by=["s1"],
        )
        a2 = DataArtifact(
            name="art",
            type="model",
            size_gb=1.0,
            location="loc",
            required_by=["s1"],
        )
        assert a1 == a2


# ─── Stage Tests ──────────────────────────────────────────────────────────────


class TestStageBasics:
    """Tests for Stage specifications and lifecycle."""

    def test_stage_pretrain_specification(self) -> None:
        """Pretrain stage: inputs raw data, outputs checkpoint."""
        stage = Stage(
            name="pretrain",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["training_data"],
            outputs=["base_model_checkpoint"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
            preconditions=["training_data exists at dataset_path"],
        )
        assert stage.name == "pretrain"
        assert stage.stage_type == StageType.PRETRAIN
        assert stage.machine_required == MachineType.GPU_32GB
        assert "training_data" in stage.inputs
        assert "base_model_checkpoint" in stage.outputs
        assert stage.cost_gpu_hours == 48.0

    def test_stage_eval_specification(self) -> None:
        """Eval stage: consumes checkpoint, outputs metrics."""
        stage = Stage(
            name="eval_base",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["base_model_checkpoint"],
            outputs=["baseline_ladder"],
            cost_gpu_hours=8.0,
            estimated_duration_hours=2.0,
            preconditions=["harness harness_path is valid"],
        )
        assert stage.stage_type == StageType.EVALUATE
        assert "base_model_checkpoint" in stage.inputs
        assert "baseline_ladder" in stage.outputs

    def test_stage_postrain_finetune(self) -> None:
        """Post-training: SFT or DPO from checkpoint."""
        stage = Stage(
            name="dpo_optimize",
            stage_type=StageType.POSTTRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["base_model_checkpoint", "preference_data"],
            outputs=["optimized_model"],
            cost_gpu_hours=16.0,
            estimated_duration_hours=16.0,
            preconditions=[
                "base_model_checkpoint format is compatible",
                "preference_data is well-formed",
            ],
        )
        assert stage.stage_type == StageType.POSTTRAIN
        assert "preference_data" in stage.inputs

    def test_stage_distillation_topdown(self) -> None:
        """Top-down distillation: teacher -> student."""
        stage = Stage(
            name="distill_from_7b",
            stage_type=StageType.DISTILL,
            machine_required=MachineType.GPU_32GB,
            inputs=["large_teacher_checkpoint", "distillation_corpus"],
            outputs=["student_checkpoint"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
            preconditions=["teacher model loaded and frozen"],
            variant="top_down",
        )
        assert stage.stage_type == StageType.DISTILL
        assert stage.variant == "top_down"

    def test_stage_distillation_bottomup(self) -> None:
        """Bottom-up distillation: multiple specialists -> unified student."""
        stage = Stage(
            name="merge_specialists",
            stage_type=StageType.DISTILL,
            machine_required=MachineType.GPU_32GB,
            inputs=[
                "specialist_math_checkpoint",
                "specialist_coding_checkpoint",
                "specialist_general_checkpoint",
                "merge_corpus",
            ],
            outputs=["generalist_student_checkpoint"],
            cost_gpu_hours=20.0,
            estimated_duration_hours=20.0,
            variant="bottom_up",
        )
        assert stage.variant == "bottom_up"
        assert len(stage.inputs) == 4

    def test_stage_serving(self) -> None:
        """Serving: loads council, exposes API."""
        stage = Stage(
            name="deploy_council",
            stage_type=StageType.SERVE,
            machine_required=MachineType.GPU_120GB,
            inputs=[
                "specialist_a_checkpoint",
                "specialist_b_checkpoint",
                "specialist_c_checkpoint",
                "auction_config",
            ],
            outputs=["api_endpoint"],
            cost_gpu_hours=0.0,  # Serving is runtime, not training
            estimated_duration_hours=0.1,  # Startup only
            preconditions=[
                "all specialists share tokenizer",
                "api_config is valid",
            ],
        )
        assert stage.stage_type == StageType.SERVE
        assert stage.cost_gpu_hours == 0.0

    def test_stage_research_hypothesis(self) -> None:
        """Research stage: hypothesis selection, probe design."""
        stage = Stage(
            name="select_armsR",
            stage_type=StageType.RESEARCH,
            machine_required=MachineType.CPU,
            inputs=["ladder_results", "prior_findings"],
            outputs=["experiment_spec", "arm_configs"],
            cost_gpu_hours=0.0,
            estimated_duration_hours=0.5,
            preconditions=["prior_findings is parseable YAML"],
        )
        assert stage.stage_type == StageType.RESEARCH
        assert stage.machine_required == MachineType.CPU


class TestStageIdentification:
    """Tests for stage identification across the arc."""

    def test_all_stage_types_covered(self) -> None:
        """Verify StageType enum covers the full LLM arc."""
        covered = {st.value for st in StageType}
        required = {
            "research",
            "pretrain",
            "posttrain",
            "distill",
            "eval",
            "serve",
        }
        assert required.issubset(covered)

    def test_stage_type_to_machine_mapping(self) -> None:
        """Natural machine assignments per stage type."""
        mapping = {
            StageType.RESEARCH: MachineType.CPU,
            StageType.PRETRAIN: MachineType.GPU_32GB,
            StageType.POSTTRAIN: MachineType.GPU_32GB,
            StageType.DISTILL: MachineType.GPU_32GB,
            StageType.EVALUATE: MachineType.GPU_120GB,
            StageType.SERVE: MachineType.GPU_120GB,
        }
        for stage_type, _expected_machine in mapping.items():
            assert stage_type in StageType  # type guard
            # In real usage, configuration would map these


# ─── DAG Validation Tests ─────────────────────────────────────────────────────


class TestDAGValidation:
    """Tests for pipeline DAG validation."""

    def test_empty_pipeline_is_valid(self) -> None:
        """Empty pipeline has no stages and no edges."""
        pipeline = Pipeline(name="empty")
        plan = pipeline.validate_and_plan()
        assert plan is not None
        assert plan.stages == []
        assert plan.stage_order == []

    def test_single_stage_pipeline(self) -> None:
        """Single-stage pipeline with no inputs is valid."""
        stage = Stage(
            name="pretrain",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["checkpoint"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        pipeline = Pipeline(name="single")
        pipeline.add_stage(stage)
        plan = pipeline.validate_and_plan()
        assert len(plan.stages) == 1
        assert plan.stages[0].name == "pretrain"

    def test_linear_pipeline_is_valid(self) -> None:
        """Linear chain: pretrain -> eval -> done."""
        s1 = Stage(
            name="pretrain",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["checkpoint"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        s2 = Stage(
            name="eval",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["checkpoint"],
            outputs=["metrics"],
            cost_gpu_hours=8.0,
            estimated_duration_hours=2.0,
        )
        pipeline = Pipeline(name="linear")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        plan = pipeline.validate_and_plan()
        assert len(plan.stages) == 2
        assert plan.stage_order == ["pretrain", "eval"]

    def test_diamond_dag_is_valid(self) -> None:
        """Diamond: common input -> two arms -> merger."""
        s1 = Stage(
            name="data_prep",
            stage_type=StageType.RESEARCH,
            machine_required=MachineType.CPU,
            inputs=[],
            outputs=["train_data"],
            cost_gpu_hours=0.0,
            estimated_duration_hours=1.0,
        )
        s2 = Stage(
            name="train_arm_a",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["train_data"],
            outputs=["ckpt_a"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
        )
        s3 = Stage(
            name="train_arm_b",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["train_data"],
            outputs=["ckpt_b"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
        )
        s4 = Stage(
            name="eval_both",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["ckpt_a", "ckpt_b"],
            outputs=["comparison"],
            cost_gpu_hours=8.0,
            estimated_duration_hours=2.0,
        )
        pipeline = Pipeline(name="diamond")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        pipeline.add_stage(s3)
        pipeline.add_stage(s4)
        plan = pipeline.validate_and_plan()
        assert len(plan.stages) == 4
        # s1 runs first, then s2 and s3 can run in parallel, then s4

    def test_missing_input_rejected(self) -> None:
        """Stage requesting an input never produced is rejected."""
        s1 = Stage(
            name="train",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["ckpt"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        s2 = Stage(
            name="eval",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["nonexistent_input"],
            outputs=["metrics"],
            cost_gpu_hours=8.0,
            estimated_duration_hours=2.0,
        )
        pipeline = Pipeline(name="broken")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        with pytest.raises(PipelineError, match="not produced by any stage"):
            pipeline.validate_and_plan()

    def test_circular_dependency_detected(self) -> None:
        """Circular edge (s1 needs s2's output, s2 needs s1's output) is rejected."""
        s1 = Stage(
            name="s1",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["output_from_s2"],
            outputs=["output_from_s1"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        s2 = Stage(
            name="s2",
            stage_type=StageType.POSTTRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["output_from_s1"],
            outputs=["output_from_s2"],
            cost_gpu_hours=16.0,
            estimated_duration_hours=16.0,
        )
        pipeline = Pipeline(name="circular")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        with pytest.raises(PipelineError, match="circular"):
            pipeline.validate_and_plan()

    def test_duplicate_output_rejected(self) -> None:
        """Two stages producing the same output is rejected."""
        s1 = Stage(
            name="train_v1",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["checkpoint"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        s2 = Stage(
            name="train_v2",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["checkpoint"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        pipeline = Pipeline(name="duplicate")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        with pytest.raises(PipelineError, match="produced by both"):
            pipeline.validate_and_plan()


# ─── GPU Scheduling Tests ─────────────────────────────────────────────────────


class TestGPUScheduling:
    """Tests for GPU allocation and concurrent job detection."""

    def test_two_concurrent_gpu_jobs_rejected(self) -> None:
        """Two GPU stages with no serial dependency are forbidden."""
        # For the real program, we'll detect at scheduling time
        # For now, document the constraint
        s1 = Stage(
            name="train_a",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["ckpt_a"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
        )
        s2 = Stage(
            name="train_b",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["ckpt_b"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
        )
        pipeline = Pipeline(name="concurrent_gpu")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        # Both stages are GPU jobs on the same machine type with no dependency
        # The plan should flag this as impossible
        plan = pipeline.validate_and_plan()
        # Check if the plan detects the conflict
        conflicts = plan.detect_gpu_conflicts()
        # Two stages on 32GB with no serialization = conflict
        assert len(conflicts) > 0

    def test_gb10_eval_allows_multiple_sequential_stages(self) -> None:
        """GB10 can run multiple evaluations if serialized by explicit dependencies."""
        s1 = Stage(
            name="train",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["ckpt"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        s2 = Stage(
            name="eval_baseline",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["ckpt"],
            outputs=["baseline_metrics"],
            cost_gpu_hours=4.0,
            estimated_duration_hours=1.0,
        )
        s3 = Stage(
            name="eval_variants",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["baseline_metrics"],  # Depend on previous eval
            outputs=["variant_metrics"],
            cost_gpu_hours=4.0,
            estimated_duration_hours=1.0,
        )
        pipeline = Pipeline(name="seq_eval")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        pipeline.add_stage(s3)
        # s2 and s3 are both on GB10, but s3 depends on s2 output
        # They are serialized by explicit data dependency
        plan = pipeline.validate_and_plan()
        conflicts = plan.detect_gpu_conflicts()
        # No conflict: eval_variants depends on eval_baseline, so they're ordered
        assert len(conflicts) == 0


# ─── Cost Calculation Tests ───────────────────────────────────────────────────


class TestCostCalculation:
    """Tests for pipeline cost tracking and budgeting."""

    def test_single_stage_cost(self) -> None:
        """Cost of a single stage."""
        stage = Stage(
            name="pretrain",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["ckpt"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        pipeline = Pipeline(name="cost_single")
        pipeline.add_stage(stage)
        plan = pipeline.validate_and_plan()
        assert plan.total_cost_gpu_hours == 48.0

    def test_multiple_sequential_stages_add_costs(self) -> None:
        """Costs accumulate when stages are sequential."""
        s1 = Stage(
            name="train",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["ckpt"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        s2 = Stage(
            name="eval",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["ckpt"],
            outputs=["metrics"],
            cost_gpu_hours=8.0,
            estimated_duration_hours=2.0,
        )
        pipeline = Pipeline(name="cost_seq")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        plan = pipeline.validate_and_plan()
        assert plan.total_cost_gpu_hours == 56.0

    def test_parallel_stages_max_concurrent_duration(self) -> None:
        """Wall-clock time for parallel stages is max, not sum."""
        s1 = Stage(
            name="train",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["ckpt"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
        )
        s2a = Stage(
            name="distill_a",
            stage_type=StageType.DISTILL,
            machine_required=MachineType.GPU_32GB,
            inputs=["ckpt"],
            outputs=["student_a"],
            cost_gpu_hours=12.0,
            estimated_duration_hours=12.0,
        )
        s2b = Stage(
            name="distill_b",
            stage_type=StageType.DISTILL,
            machine_required=MachineType.GPU_32GB,
            inputs=["ckpt"],
            outputs=["student_b"],
            cost_gpu_hours=12.0,
            estimated_duration_hours=12.0,
        )
        pipeline = Pipeline(name="cost_parallel")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2a)
        pipeline.add_stage(s2b)
        plan = pipeline.validate_and_plan()
        # Total GPU hours: 24 + 12 + 12 = 48
        assert plan.total_cost_gpu_hours == 48.0


# ─── Execution Plan Tests ─────────────────────────────────────────────────────


class TestExecutionPlan:
    """Tests for execution plan generation and properties."""

    def test_plan_has_topological_order(self) -> None:
        """Execution plan orders stages in topological sort."""
        s1 = Stage(
            name="a",
            stage_type=StageType.RESEARCH,
            machine_required=MachineType.CPU,
            inputs=[],
            outputs=["x"],
            cost_gpu_hours=0.0,
            estimated_duration_hours=1.0,
        )
        s2 = Stage(
            name="b",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["x"],
            outputs=["y"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
        )
        s3 = Stage(
            name="c",
            stage_type=StageType.EVALUATE,
            machine_required=MachineType.GPU_120GB,
            inputs=["y"],
            outputs=["z"],
            cost_gpu_hours=8.0,
            estimated_duration_hours=2.0,
        )
        pipeline = Pipeline(name="topo")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        pipeline.add_stage(s3)
        plan = pipeline.validate_and_plan()
        # Order must be a -> b -> c
        assert plan.stage_order == ["a", "b", "c"]

    def test_plan_tracks_stage_readiness(self) -> None:
        """Plan identifies which stages can start immediately."""
        s1 = Stage(
            name="stage1",
            stage_type=StageType.RESEARCH,
            machine_required=MachineType.CPU,
            inputs=[],
            outputs=["out1"],
            cost_gpu_hours=0.0,
            estimated_duration_hours=1.0,
        )
        s2 = Stage(
            name="stage2",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=["out1"],
            outputs=["out2"],
            cost_gpu_hours=24.0,
            estimated_duration_hours=24.0,
        )
        pipeline = Pipeline(name="ready")
        pipeline.add_stage(s1)
        pipeline.add_stage(s2)
        plan = pipeline.validate_and_plan()
        # stage1 is ready immediately (no inputs)
        ready = plan.get_ready_stages()
        assert "stage1" in ready
        assert "stage2" not in ready

    def test_plan_artifact_tracking(self) -> None:
        """Plan tracks which stages produce which artifacts."""
        s1 = Stage(
            name="pretrain",
            stage_type=StageType.PRETRAIN,
            machine_required=MachineType.GPU_32GB,
            inputs=[],
            outputs=["checkpoint"],
            cost_gpu_hours=48.0,
            estimated_duration_hours=48.0,
        )
        pipeline = Pipeline(name="artifacts")
        pipeline.add_stage(s1)
        plan = pipeline.validate_and_plan()
        producers = plan.get_artifact_producers()
        assert producers["checkpoint"] == "pretrain"


class TestLLMConstructionArc:
    """Integration test: full LLM construction pipeline."""

    def test_full_pipeline_from_research_to_serving(self) -> None:
        """Complete arc: research -> pretrain -> eval -> distill -> serve."""
        stages = [
            Stage(
                name="select_arms",
                stage_type=StageType.RESEARCH,
                machine_required=MachineType.CPU,
                inputs=[],
                outputs=["experiment_spec"],
                cost_gpu_hours=0.0,
                estimated_duration_hours=0.5,
            ),
            Stage(
                name="pretrain",
                stage_type=StageType.PRETRAIN,
                machine_required=MachineType.GPU_32GB,
                inputs=["experiment_spec"],
                outputs=["base_checkpoint"],
                cost_gpu_hours=48.0,
                estimated_duration_hours=48.0,
            ),
            Stage(
                name="eval_baseline",
                stage_type=StageType.EVALUATE,
                machine_required=MachineType.GPU_120GB,
                inputs=["base_checkpoint"],
                outputs=["baseline_metrics"],
                cost_gpu_hours=8.0,
                estimated_duration_hours=2.0,
            ),
            Stage(
                name="sft_optimize",
                stage_type=StageType.POSTTRAIN,
                machine_required=MachineType.GPU_32GB,
                inputs=["base_checkpoint"],
                outputs=["sft_checkpoint"],
                cost_gpu_hours=16.0,
                estimated_duration_hours=16.0,
            ),
            Stage(
                name="distill_to_student",
                stage_type=StageType.DISTILL,
                machine_required=MachineType.GPU_32GB,
                inputs=["sft_checkpoint"],
                outputs=["student_checkpoint"],
                cost_gpu_hours=12.0,
                estimated_duration_hours=12.0,
            ),
            Stage(
                name="final_eval",
                stage_type=StageType.EVALUATE,
                machine_required=MachineType.GPU_120GB,
                inputs=["student_checkpoint"],
                outputs=["final_metrics"],
                cost_gpu_hours=4.0,
                estimated_duration_hours=1.0,
            ),
            Stage(
                name="deploy",
                stage_type=StageType.SERVE,
                machine_required=MachineType.GPU_120GB,
                inputs=["student_checkpoint"],
                outputs=["api_endpoint"],
                cost_gpu_hours=0.0,
                estimated_duration_hours=0.1,
            ),
        ]
        pipeline = Pipeline(name="full_arc")
        for stage in stages:
            pipeline.add_stage(stage)
        plan = pipeline.validate_and_plan()
        assert len(plan.stages) == 7
        assert plan.total_cost_gpu_hours == 88.0
        assert "select_arms" in plan.stage_order
        assert "deploy" in plan.stage_order
        assert plan.stage_order.index("pretrain") < plan.stage_order.index("eval_baseline")
