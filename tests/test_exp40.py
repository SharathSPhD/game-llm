"""The ladder scorer, tested against an oracle that catches misalignment.

The first version of these tests built a fake model whose logits ignored the
input ids, and the scorer's off-by-one — reading logits at each continuation
token's own position instead of one before it — passed them while returning
exact chance on every real benchmark. The fake here is a bigram oracle:
logits at position i favour ``(ids[i] + 1) % vocab``, so the only way to give
a high score to the successor continuation is to read the logits at the
correct shifted position. A scorer with the old defect ties every option and
these tests fail.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from exp40_ladder import (  # noqa: E402
    ModelAdapter,
    eval_lambada,
    eval_multiple_choice,
    score_continuation,
)

VOCAB = 8
HIGH = 5.0


class BigramOracle(nn.Module):
    """logits[b, i, v] = HIGH if v == (ids[b, i] + 1) % VOCAB else 0."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*ids.shape, VOCAB)
        favoured = (ids + 1) % VOCAB
        logits.scatter_(2, favoured.unsqueeze(-1), HIGH)
        return logits


def log_p_favoured() -> float:
    return HIGH - math.log(math.exp(HIGH) + (VOCAB - 1))


def log_p_other() -> float:
    return 0.0 - math.log(math.exp(HIGH) + (VOCAB - 1))


class IdTokenizer:
    """Text is a space-separated list of token ids."""

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in text.split()]


def make_adapter(max_len: int = 32, extra_vocab: int = 0) -> ModelAdapter:
    model = BigramOracle().eval()

    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        logits = model(ids)
        if extra_vocab:
            # Pad columns carry huge values; a correct adapter slices them
            # away before scoring, the way the 50304-vocab checkpoints must.
            pad = torch.full((*ids.shape, extra_vocab), 100.0)
            logits = torch.cat([logits, pad], dim=-1)[:, :, :VOCAB]
        return logits

    tok = IdTokenizer()
    return ModelAdapter(
        name="bigram-oracle", logits_fn=logits_fn, tokenize=tok.encode,
        max_len=max_len, device="cpu",
    )


def test_successor_continuation_scores_exactly() -> None:
    adapter = make_adapter()
    score, n = adapter.score_ids([1, 2], [3, 4])
    assert n == 2
    assert abs(score - 2 * log_p_favoured()) < 1e-5


def test_non_successor_continuation_scores_low() -> None:
    adapter = make_adapter()
    score, n = adapter.score_ids([1, 2], [5, 6])
    assert n == 2
    assert abs(score - (log_p_other() + log_p_favoured())) < 1e-5


def test_score_continuation_string_path() -> None:
    adapter = make_adapter()
    score, n = score_continuation(adapter, "1 2", "3 4")
    assert n == 2
    assert abs(score - 2 * log_p_favoured()) < 1e-5


def test_multiple_choice_picks_the_successor() -> None:
    adapter = make_adapter()
    examples = [
        {"context": "0 1", "options": ["2 3", "5 6", "4 4"], "gold": 0},
        {"context": "3 4", "options": ["0 0", "5 6", "2 2"], "gold": 1},
    ]
    res = eval_multiple_choice(adapter, examples)
    assert res["acc"] == 1.0
    assert res["n"] == 2


def test_acc_norm_normalises_by_bytes() -> None:
    adapter = make_adapter()
    # Both options start with the favoured successor; the longer one adds a
    # second favoured token, so raw sum prefers... both are favoured chains,
    # raw prefers the shorter (fewer negative terms), byte-normalisation
    # divides by length and must keep the answer stable here.
    examples = [{"context": "0 1", "options": ["2", "2 3"], "gold": 0}]
    res = eval_multiple_choice(adapter, examples)
    assert res["acc"] == 1.0
    assert 0.0 <= res["acc_norm"] <= 1.0


def test_lambada_greedy_exact_match() -> None:
    adapter = make_adapter()
    examples = [
        {"context": "1 2 3", "continuation": "4 5"},
        {"context": "1 2 3", "continuation": "6 6"},
    ]
    res = eval_lambada(adapter, examples)
    assert res["acc"] == 0.5
    assert res["n"] == 2


def test_left_truncation_keeps_alignment() -> None:
    adapter = make_adapter(max_len=4)
    # Context of 6 tokens truncates to 2; the surviving suffix still ends in
    # 5, so the successor continuation stays favoured and exactly scored.
    score, n = adapter.score_ids([0, 1, 2, 3, 4, 5], [6, 7])
    assert n == 2
    assert abs(score - 2 * log_p_favoured()) < 1e-5


def test_truncation_never_drops_all_context() -> None:
    adapter = make_adapter(max_len=3)
    score, n = adapter.score_ids([1, 2], [3, 4, 5])
    assert n == 3
    assert abs(score - 3 * log_p_favoured()) < 1e-5


def test_empty_context_is_refused() -> None:
    adapter = make_adapter()
    try:
        adapter.score_ids([], [1, 2])
        raise AssertionError("empty context must be refused")
    except ValueError:
        pass


def test_padded_vocab_is_sliced_before_scoring() -> None:
    adapter = make_adapter(extra_vocab=4)
    score, n = adapter.score_ids([1, 2], [3, 4])
    assert n == 2
    assert abs(score - 2 * log_p_favoured()) < 1e-5


def test_winogrande_option_fills_prefix() -> None:
    from exp40_ladder import winogrande_example

    ex = winogrande_example(
        sentence="The ball hit the _ hard.",
        option1="wall", option2="floor", answer="2",
    )
    assert ex["gold"] == 1
    assert ex["contexts"][0] == "The ball hit the wall"
    assert ex["contexts"][1] == "The ball hit the floor"
    assert ex["options"] == [" hard.", " hard."]


def test_bos_prefix_prepends_to_context_only() -> None:
    import dataclasses

    adapter = dataclasses.replace(make_adapter(), bos_prefix=(2,))
    # With BOS token 2 prepended, context [2] alone makes 3 the favoured
    # successor of the last context token 2 — the continuation [3, 4] must
    # score as a fully favoured chain, proving BOS lands in the context and
    # never in the continuation.
    score, n = adapter.score_ids([2], [3, 4])
    assert n == 2
    assert abs(score - 2 * log_p_favoured()) < 1e-5
