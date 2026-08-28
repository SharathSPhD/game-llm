"""TDD for KineticLM (SPEC 0011): converting a pretrained HF causal LM into the
EqLMCore topology — explicit outer layers around a weight-tied recursive core.

Uses a tiny randomly-initialized Qwen3 to keep tests fast; the conversion logic
is architecture-generic (anything with model.model.layers).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

transformers = pytest.importorskip("transformers")

from kinetic_ai.models.kinetic_lm import (  # noqa: E402
    KineticConfig,
    convert_to_kinetic,
    count_unique_params,
)


def _tiny_base():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg)


@pytest.fixture(scope="module")
def base():
    return _tiny_base()


@pytest.fixture()
def kinetic(base):
    return convert_to_kinetic(
        _tiny_base(), KineticConfig(n_pre=2, n_post=2, init_strategy="average")
    )


class TestSurgery:
    def test_reduces_unique_parameters(self, base, kinetic) -> None:
        """Tying the middle M layers into one must remove (M-1) layers of params."""
        n_base = count_unique_params(base)
        n_kin = count_unique_params(kinetic)
        assert n_kin < n_base
        per_layer = sum(p.numel() for p in base.model.layers[0].parameters())
        m = 8 - 2 - 2
        assert n_base - n_kin == pytest.approx((m - 1) * per_layer, rel=1e-6)

    def test_core_layers_share_storage(self, kinetic) -> None:
        core = kinetic.model.layers[2:6]
        ref = core[0].mlp.gate_proj.weight
        for layer in core[1:]:
            assert layer.mlp.gate_proj.weight.data_ptr() == ref.data_ptr()

    def test_core_layers_have_distinct_layer_idx(self, kinetic) -> None:
        """Required for KV cache correctness — shared weights, distinct slots."""
        idxs = [layer.self_attn.layer_idx for layer in kinetic.model.layers]
        assert idxs == list(range(len(kinetic.model.layers)))

    def test_outer_layers_are_untouched(self, base, kinetic) -> None:
        torch.testing.assert_close(
            kinetic.model.layers[0].mlp.gate_proj.weight,
            base.model.layers[0].mlp.gate_proj.weight,
        )
        torch.testing.assert_close(
            kinetic.model.layers[-1].mlp.gate_proj.weight,
            base.model.layers[-1].mlp.gate_proj.weight,
        )

    def test_average_init_is_mean_of_middle_layers(self, base) -> None:
        kin = convert_to_kinetic(
            _tiny_base(), KineticConfig(n_pre=2, n_post=2, init_strategy="average")
        )
        expected = torch.stack(
            [layer.mlp.gate_proj.weight for layer in base.model.layers[2:6]]
        ).mean(0)
        torch.testing.assert_close(kin.model.layers[2].mlp.gate_proj.weight, expected)

    def test_stepwise_init_copies_a_single_middle_layer(self, base) -> None:
        kin = convert_to_kinetic(
            _tiny_base(), KineticConfig(n_pre=2, n_post=2, init_strategy="stepwise")
        )
        mids = [base.model.layers[i].mlp.gate_proj.weight for i in range(2, 6)]
        assert any(
            torch.allclose(kin.model.layers[2].mlp.gate_proj.weight, m) for m in mids
        )


class TestForwardAndTraining:
    def test_forward_shape(self, kinetic) -> None:
        ids = torch.randint(0, 256, (2, 12))
        out = kinetic(ids)
        assert out.logits.shape == (2, 12, 256)

    def test_gradients_accumulate_through_all_recursions(self, kinetic) -> None:
        ids = torch.randint(0, 256, (1, 8))
        kinetic(ids, labels=ids).loss.backward()
        g = kinetic.model.layers[2].mlp.gate_proj.weight.grad
        assert g is not None and g.abs().sum() > 0

    def test_generate_with_cache_matches_no_cache(self, kinetic) -> None:
        """Shared weights must not corrupt the KV cache."""
        ids = torch.randint(0, 256, (1, 6))
        torch.manual_seed(0)
        a = kinetic.generate(ids, max_new_tokens=6, do_sample=False, use_cache=True)
        torch.manual_seed(0)
        b = kinetic.generate(ids, max_new_tokens=6, do_sample=False, use_cache=False)
        assert torch.equal(a, b)


class TestBudgetDialAndAnytime:
    def test_set_recursion_depth_changes_applications(self, kinetic) -> None:
        assert kinetic.recursion_depth == 4
        kinetic.set_recursion_depth(2)
        assert kinetic.recursion_depth == 2
        assert len(kinetic.model.layers) == 2 + 2 + 2
        ids = torch.randint(0, 256, (1, 6))
        assert kinetic(ids).logits.shape == (1, 6, 256)

    def test_depth_changes_outputs(self, kinetic) -> None:
        ids = torch.randint(0, 256, (1, 6))
        deep = kinetic(ids).logits.clone()
        kinetic.set_recursion_depth(1)
        shallow = kinetic(ids).logits
        assert not torch.allclose(deep, shallow)

    def test_recursion_depth_bounds(self, kinetic) -> None:
        with pytest.raises(ValueError):
            kinetic.set_recursion_depth(0)

    def test_forward_at_depths_returns_logits_per_depth(self, kinetic) -> None:
        """Anytime supervision (F24/B1) at real scale."""
        ids = torch.randint(0, 256, (1, 6))
        outs = kinetic.forward_at_depths(ids, depths=[1, 2, 4])
        assert sorted(outs) == [1, 2, 4]
        for logits in outs.values():
            assert logits.shape == (1, 6, 256)
        # deeper computation must differ from shallower
        assert not torch.allclose(outs[1], outs[4])

    def test_depth_restored_after_anytime_forward(self, kinetic) -> None:
        kinetic.forward_at_depths(torch.randint(0, 256, (1, 6)), depths=[1, 2])
        assert kinetic.recursion_depth == 4


class TestPersistence:
    def test_save_load_roundtrip_preserves_tying_and_outputs(self, kinetic, tmp_path) -> None:
        from kinetic_ai.models.kinetic_lm import load_kinetic

        ids = torch.randint(0, 256, (1, 6))
        kinetic.eval()
        ref = kinetic(ids).logits
        kinetic.save_pretrained(tmp_path / "k")
        loaded = load_kinetic(tmp_path / "k")
        loaded.eval()
        torch.testing.assert_close(loaded(ids).logits, ref)
        core = loaded.model.layers[2:6]
        assert all(
            layer.mlp.gate_proj.weight.data_ptr()
            == core[0].mlp.gate_proj.weight.data_ptr()
            for layer in core
        )
        assert count_unique_params(loaded) == count_unique_params(kinetic)


class TestDepthBeyondBaseLayerCount:
    """Raising depth past the base's layer count must not overflow per-layer
    config lists (Qwen3 indexes config.layer_types[i] inside its forward)."""

    def test_deeper_than_base_forward_works(self, kinetic) -> None:
        kinetic.set_recursion_depth(12)  # 2 + 12 + 2 = 16 > base 8
        ids = torch.randint(0, 256, (1, 6))
        assert kinetic(ids).logits.shape == (1, 6, 256)

    def test_layer_types_tracks_stack_length(self, kinetic) -> None:
        types = getattr(kinetic.config, "layer_types", None)
        if types is None:
            pytest.skip("architecture has no per-layer type list")
        kinetic.set_recursion_depth(9)
        assert len(kinetic.config.layer_types) == len(kinetic.model.layers)

    def test_num_hidden_layers_tracks_stack_length(self, kinetic) -> None:
        kinetic.set_recursion_depth(7)
        assert kinetic.config.num_hidden_layers == len(kinetic.model.layers) == 2 + 7 + 2


class TestBlockRecursiveSharing:
    """n_cores partitions the middle into several shared blocks — trading
    parameter saving against how much function the collapse destroys."""

    def test_two_cores_saves_half_as_much_as_one(self) -> None:
        base = _tiny_base()
        per_layer = sum(p.numel() for p in base.model.layers[0].parameters())
        n_base = count_unique_params(base)
        one = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=1))
        two = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=2))
        assert n_base - count_unique_params(one) == pytest.approx(3 * per_layer, rel=1e-6)
        assert n_base - count_unique_params(two) == pytest.approx(2 * per_layer, rel=1e-6)

    def test_cores_are_distinct_but_each_group_is_tied(self) -> None:
        m = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=2))
        layers = m.model.layers
        assert layers[2].mlp.gate_proj.weight.data_ptr() == layers[3].mlp.gate_proj.weight.data_ptr()
        assert layers[4].mlp.gate_proj.weight.data_ptr() == layers[5].mlp.gate_proj.weight.data_ptr()
        assert layers[2].mlp.gate_proj.weight.data_ptr() != layers[4].mlp.gate_proj.weight.data_ptr()

    def test_stack_length_preserved_by_default(self) -> None:
        m = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=2))
        assert len(m.model.layers) == 8
        assert m(torch.randint(0, 256, (1, 6))).logits.shape == (1, 6, 256)

    def test_budget_dial_applies_per_core(self) -> None:
        m = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=2))
        assert m.recursion_depth == 2
        m.set_recursion_depth(3)
        assert len(m.model.layers) == 2 + 3 * 2 + 2
        assert m(torch.randint(0, 256, (1, 6))).logits.shape == (1, 6, 256)

    def test_roundtrip_with_multiple_cores(self, tmp_path) -> None:
        from kinetic_ai.models.kinetic_lm import load_kinetic

        m = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=2)).eval()
        ids = torch.randint(0, 256, (1, 6))
        ref = m(ids).logits
        m.save_pretrained(tmp_path / "k2")
        loaded = load_kinetic(tmp_path / "k2").eval()
        torch.testing.assert_close(loaded(ids).logits, ref)
        assert count_unique_params(loaded) == count_unique_params(m)


class TestDepthLoRARelaxation:
    """SPEC 0014 arm A3: per-depth LoRA deltas on the shared core.

    Each recursion gets its own low-rank correction so the tied block can act
    slightly differently at each depth (Relaxed Recursive Transformers). The
    scale anneals to zero during training, so at the end the adapters vanish
    and the pure parameter saving is restored.
    """

    def _model(self):
        from kinetic_ai.models.kinetic_lm import add_depth_lora

        m = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=1))
        add_depth_lora(m, rank=4)
        return m

    def test_adds_parameters_but_far_fewer_than_untying(self, base) -> None:
        from kinetic_ai.models.kinetic_lm import add_depth_lora

        plain = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=1))
        n_plain = count_unique_params(plain)
        lora = self._model()
        n_lora = count_unique_params(lora)
        per_layer = sum(p.numel() for p in base.model.layers[0].parameters())
        assert n_lora > n_plain
        assert n_lora - n_plain < 3 * per_layer  # cheaper than untying the 3 extra layers

    def test_base_weights_still_shared(self) -> None:
        m = self._model()
        core = m.model.layers[2:6]
        ptr = core[0].mlp.gate_proj.base.weight.data_ptr()
        assert all(layer.mlp.gate_proj.base.weight.data_ptr() == ptr for layer in core[1:])

    def test_each_depth_has_its_own_adapter(self) -> None:
        m = self._model()
        core = m.model.layers[2:6]
        ptrs = {layer.mlp.gate_proj.lora_a.data_ptr() for layer in core}
        assert len(ptrs) == len(core)

    def test_zero_scale_reproduces_the_tied_model(self) -> None:
        from kinetic_ai.models.kinetic_lm import add_depth_lora, set_lora_scale

        torch.manual_seed(0)
        plain = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=1)).eval()
        ids = torch.randint(0, 256, (1, 6))
        ref = plain(ids).logits
        torch.manual_seed(0)
        lora = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=1))
        add_depth_lora(lora, rank=4)
        lora.eval()
        set_lora_scale(lora, 0.0)
        torch.testing.assert_close(lora(ids).logits, ref)

    def test_nonzero_scale_changes_outputs(self) -> None:
        from kinetic_ai.models.kinetic_lm import set_lora_scale

        m = self._model().eval()
        ids = torch.randint(0, 256, (1, 6))
        set_lora_scale(m, 0.0)
        off = m(ids).logits.clone()
        for layer in m.model.layers[2:6]:
            torch.nn.init.normal_(layer.mlp.gate_proj.lora_b, std=0.05)
        set_lora_scale(m, 1.0)
        assert not torch.allclose(m(ids).logits, off)

    def test_merge_and_remove_restores_plain_module_and_param_count(self) -> None:
        from kinetic_ai.models.kinetic_lm import merge_and_remove_lora, set_lora_scale

        m = self._model().eval()
        plain = convert_to_kinetic(_tiny_base(), KineticConfig(n_pre=2, n_post=2, n_cores=1))
        n_plain = count_unique_params(plain)
        set_lora_scale(m, 0.0)  # annealed to zero: merging is a no-op on values
        ids = torch.randint(0, 256, (1, 6))
        before = m(ids).logits.clone()
        merge_and_remove_lora(m)
        torch.testing.assert_close(m(ids).logits, before)
        assert count_unique_params(m) == n_plain
        assert isinstance(m.model.layers[2].mlp.gate_proj, torch.nn.Linear)
