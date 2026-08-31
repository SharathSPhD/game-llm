"""Does solver effort track expectancy? The cheap test behind a halting idea.

The architecture analysis (docs/kineticaiprabhasaarchitectureanalysis.md §5)
records an idea: replace EqLM's numerical halting criterion — stop when the
iterate's residual falls below a tuned epsilon — with a semantic one, stop
when the model's expectancy at the position is saturated. Before that idea is
worth any engineering, one correlation must exist: positions where the
model's next-token distribution is uncertain (high entropy, expectancy open)
should be the positions the fixed-point solver works hardest on, and
positions where the distribution has collapsed (expectancy closed, as at a
sentence-final period) should settle early. If solver effort and expectancy
are uncorrelated, the halting idea dies here for the cost of an hour of GB10
time; if they correlate, RQ-3b gains a semantic dial worth testing at scale.

The measurement reuses exp31's per-position adaptive solve on checkpoints
that already exist, evaluates natural sentences from the SPEC 0022 holdout,
and reports Spearman correlations plus the sentence-final contrast. This is
an exploratory probe, recorded as an observation, not a gated experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp31_adaptive_depth import adaptive_forward, calibrate_tolerance, load_model  # noqa: E402


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation, computed without scipy."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def sentences_from_holdout(pack_dir: str, tok: Any, n: int, max_tokens: int) -> list[list[int]]:
    """Natural sentences from the byte-identical holdout the twins never train on."""
    from kinetic_ai.data.pack import PackReader

    reader = PackReader(pack_dir)
    batches = reader.holdout_batches(2048, 4, 8)
    text = "".join(tok.decode(row.tolist()) for b in batches for row in b)
    out: list[list[int]] = []
    for raw in text.split(". "):
        s = raw.strip()
        if not (20 < len(s) < 300) or "\n" in s:
            continue
        ids = tok(s + ".")["input_ids"]
        if 8 <= len(ids) <= max_tokens:
            out.append(ids)
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--checkpoint", default="results/scale/ckpt/eqlm_anytime_seed42.pt")
    ap.add_argument("--pack-dir", default="data/pack_1b")
    ap.add_argument("--sentences", type=int, default=400)
    ap.add_argument("--max-iter", type=int, default=36)
    ap.add_argument("--target-mean-iters", type=float, default=12.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/scale/exp41_halting_semantics.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = load_model(Path(args.checkpoint), args.device)
    max_tokens = int(model.config.max_seq_len)
    sents = sentences_from_holdout(args.pack_dir, tok, args.sentences, max_tokens)
    if len(sents) < 50:
        raise SystemExit(f"only {len(sents)} usable sentences; need more holdout")

    probe = torch.randint(
        0, model.config.vocab_size - 1, (4, min(64, max_tokens)), device=args.device
    )
    tol, achieved = calibrate_tolerance(model, probe, args.target_mean_iters, args.max_iter)
    print(f"calibrated tol {tol:.3e} -> mean iters {achieved:.2f}", flush=True)

    per_pos_iters: list[float] = []
    per_pos_entropy: list[float] = []
    per_sent_rho: list[float] = []
    final_iters: list[float] = []
    nonfinal_iters: list[float] = []

    with torch.no_grad():
        for ids_list in sents:
            ids = torch.tensor([ids_list], device=args.device)
            logits, iters = adaptive_forward(model, ids, tol, args.max_iter)
            # Expectancy openness at position t is the entropy of the
            # next-token distribution the model holds there.
            lp = F.log_softmax(logits[0].float(), dim=-1)
            ent = -(lp.exp() * lp).sum(-1)
            it = iters[0]
            # The first position has no context and both quantities are
            # degenerate there; drop it.
            it, ent = it[1:], ent[1:]
            per_pos_iters += it.tolist()
            per_pos_entropy += ent.tolist()
            if len(it) >= 5:
                per_sent_rho.append(spearman(it, ent))
            final_iters.append(float(it[-1]))
            nonfinal_iters += it[:-1].tolist()

    it_t = torch.tensor(per_pos_iters)
    en_t = torch.tensor(per_pos_entropy)
    rho_pooled = spearman(it_t, en_t)
    rho_sent = torch.tensor(per_sent_rho)
    fin = torch.tensor(final_iters)
    non = torch.tensor(nonfinal_iters)

    report = {
        "checkpoint": args.checkpoint,
        "n_sentences": len(sents),
        "n_positions": len(per_pos_iters),
        "calibrated_tol": tol,
        "mean_iters": round(float(it_t.mean()), 3),
        "spearman_pooled_iters_vs_entropy": round(rho_pooled, 4),
        "spearman_per_sentence_mean": round(float(rho_sent.mean()), 4),
        "spearman_per_sentence_ci95": round(
            float(1.96 * rho_sent.std() / max(len(rho_sent), 1) ** 0.5), 4
        ),
        "sentence_final_mean_iters": round(float(fin.mean()), 3),
        "non_final_mean_iters": round(float(non.mean()), 3),
        "final_minus_nonfinal": round(float(fin.mean() - non.mean()), 3),
        "note": "positive rho means the solver works longer where expectancy "
                "is open; a negative final-minus-nonfinal gap means "
                "sentence-final positions settle early, the direction the "
                "halting idea requires",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
