"""The SDPA attention path must be the same function as the explicit path.

SPEC 0022 enables the fused kernel for both twin arms; if it computed anything
other than softmax(qk^T/sqrt(d) + causal mask)v the twin comparison would be
run on a different architecture than every finding before it. Equality is
therefore asserted at the block level and at the full-model level, in eval
mode where dropout cannot introduce sampling noise.
"""

from __future__ import annotations

import dataclasses

import torch

from kinetic_ai.models.eqlm import EqLM, EqLMBlock, EqLMConfig, ExplicitLM


def _cfg(**over: object) -> EqLMConfig:
    base = dict(
        vocab_size=211, d_model=64, n_heads=4, d_ff=128, max_seq_len=48,
        deq_max_iter=6, dropout=0.0, spectral_norm=False,
        residual_damping=0.2, map_form="postln",
    )
    base.update(over)
    return EqLMConfig(**base)  # type: ignore[arg-type]


def test_block_sdpa_matches_naive() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    block = EqLMBlock(cfg).eval()
    sdpa_block = EqLMBlock(_cfg(sdpa=True)).eval()
    sdpa_block.load_state_dict(block.state_dict())
    z = torch.randn(2, 17, cfg.d_model)
    x = torch.randn(2, 17, cfg.d_model)
    with torch.no_grad():
        a = block(z, x)
        b = sdpa_block(z, x)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_model_logits_match_under_sdpa() -> None:
    torch.manual_seed(1)
    cfg = _cfg()
    model = EqLM(config=cfg).eval()
    flipped = EqLM(config=dataclasses.replace(cfg, sdpa=True)).eval()
    flipped.load_state_dict(model.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a = model.forward_unrolled(ids, supervise_at=[cfg.deq_max_iter])[-1][1]
        b = flipped.forward_unrolled(ids, supervise_at=[cfg.deq_max_iter])[-1][1]
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_explicit_lm_matches_under_sdpa() -> None:
    torch.manual_seed(2)
    cfg = _cfg()
    model = ExplicitLM(config=cfg, n_layers=3).eval()
    flipped = ExplicitLM(config=dataclasses.replace(cfg, sdpa=True), n_layers=3).eval()
    flipped.load_state_dict(model.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a = model(ids)
        b = flipped(ids)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_default_is_naive_for_old_checkpoints() -> None:
    assert EqLMConfig().sdpa is False
