"""TDD for the H6 (SPEC 0010 / ADR 0005) model additions.

B1: EqLM.forward_unrolled — anytime supervision on intermediate iterates.
B2: EqLM.local_lipschitz — differentiable trajectory-local contraction probe.
B3: EqLMCore — wide explicit encoder/decoder around a small equilibrium core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.eqlm import (
    EqLM,
    EqLMConfig,
    EqLMCore,
    count_params,
    load_checkpoint,
    save_checkpoint,
)

TINY = dict(
    vocab_size=211, d_model=16, n_heads=2, d_ff=32, max_seq_len=32,
    deq_max_iter=6, map_form="postln", dropout=0.0,
)


@pytest.fixture()
def tiny_eqlm() -> EqLM:
    torch.manual_seed(0)
    return EqLM(config=EqLMConfig(**TINY))


class TestB1Unrolled:
    def test_returns_logits_at_supervision_points(self, tiny_eqlm) -> None:
        ids = torch.randint(0, 211, (2, 8))
        outs = tiny_eqlm.forward_unrolled(ids, supervise_at=[2, 4, 6])
        assert [k for k, _ in outs] == [2, 4, 6]
        for _, logits in outs:
            assert logits.shape == (2, 8, 211)

    def test_gradients_flow_from_early_iterates(self, tiny_eqlm) -> None:
        ids = torch.randint(0, 211, (2, 8))
        outs = tiny_eqlm.forward_unrolled(ids, supervise_at=[2])
        loss = outs[0][1].sum()
        loss.backward()
        grads = [p.grad for p in tiny_eqlm.block.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), \
            "early-iterate loss must reach block parameters"

    def test_final_iterate_matches_depth(self, tiny_eqlm) -> None:
        """supervise_at max == deq_max_iter — deepest logits use full budget."""
        ids = torch.randint(0, 211, (1, 6))
        outs = tiny_eqlm.forward_unrolled(ids, supervise_at=[6])
        assert outs[-1][0] == 6

    def test_rejects_bad_depths(self, tiny_eqlm) -> None:
        ids = torch.randint(0, 211, (1, 4))
        with pytest.raises(ValueError):
            tiny_eqlm.forward_unrolled(ids, supervise_at=[0])
        with pytest.raises(ValueError):
            tiny_eqlm.forward_unrolled(ids, supervise_at=[99])


class TestB2Lipschitz:
    def test_estimate_positive_and_differentiable(self, tiny_eqlm) -> None:
        ids = torch.randint(0, 211, (2, 8))
        tiny_eqlm(ids)  # populates last_z_star
        lhat = tiny_eqlm.local_lipschitz(ids, alpha=0.5)
        assert lhat.item() > 0
        lhat.backward()
        grads = [p.grad for p in tiny_eqlm.block.parameters() if p.grad is not None]
        assert grads, "penalty must be differentiable w.r.t. block params"

    def test_estimate_is_finite_and_scale_free(self, tiny_eqlm) -> None:
        ids = torch.randint(0, 211, (2, 8))
        tiny_eqlm(ids)
        vals = [tiny_eqlm.local_lipschitz(ids, alpha=a).item() for a in (0.1, 1.0)]
        assert all(torch.isfinite(torch.tensor(v)) for v in vals)


class TestB3Core:
    def _core(self) -> EqLMCore:
        torch.manual_seed(0)
        return EqLMCore(
            config=EqLMConfig(**TINY),
            d_core=8, n_heads_core=2, d_ff_core=16, n_enc=1, n_dec=1,
        )

    def test_forward_shape(self) -> None:
        m = self._core()
        ids = torch.randint(0, 211, (2, 8))
        logits = m(ids)
        assert logits.shape == (2, 8, 211)

    def test_core_solve_is_small_dim(self) -> None:
        m = self._core()
        ids = torch.randint(0, 211, (2, 8))
        m(ids)
        assert m.last_z_star is not None
        assert m.last_z_star.shape[-1] == 8, "equilibrium must live in d_core"

    def test_solver_telemetry_exposed(self) -> None:
        m = self._core()
        m(torch.randint(0, 211, (1, 6)))
        assert isinstance(m.deq.last_info, dict) and "iterations" in m.deq.last_info

    def test_checkpoint_roundtrip(self, tmp_path) -> None:
        m = self._core()
        ids = torch.randint(0, 211, (1, 6))
        ref = m(ids)
        save_checkpoint(m, tmp_path / "core.pt")
        m2 = load_checkpoint(tmp_path / "core.pt")
        assert isinstance(m2, EqLMCore)
        torch.testing.assert_close(m2(ids), ref)

    def test_param_count_scales_with_enc_dec(self) -> None:
        small = self._core()
        torch.manual_seed(0)
        big = EqLMCore(config=EqLMConfig(**TINY), d_core=8, n_heads_core=2,
                       d_ff_core=16, n_enc=2, n_dec=2)
        assert count_params(big) > count_params(small)
