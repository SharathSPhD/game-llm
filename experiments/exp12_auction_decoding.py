"""Experiment 12 — H4 (SPEC 0008): truthful token-auction decoding of specialists.

Train two ~30M explicit LMs on disjoint BabyLM subdomains (child-directed
speech: childes; written/wiki: simple_wiki), then evaluate on a 50/50
interleaved held-out mixed-domain stream:

  S_A   specialist A alone
  S_B   specialist B alone
  ENS   uniform logit-average ensemble
  AUC   second-price token auction (bid = own max-prob confidence;
        winner's distribution scores the token; winner pays second price)

H4 scoring (pre-registered): AUC beats BEST single specialist on mixed-domain
perplexity => MET; beats worst but not best => PARTIAL; else MISSED.

The per-position auction is computed vectorized (argmax bid / second-highest
payment — closed form of the validated F6 mechanism); a trace sample is also
run through kinetic_ai.mechanisms.auctions.TokenAuction to (a) cross-check
the vectorized path against the validated implementation and (b) produce
real bid/payment traces for the Auction playground.

Usage:
    python exp12_auction_decoding.py --config configs/exp12.yaml --output results/exp12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.config import AuctionConfig, AuctionType  # noqa: E402
from kinetic_ai.mechanisms.auctions import TokenAuction  # noqa: E402
from kinetic_ai.models.eqlm import EqLMConfig, ExplicitLM, save_checkpoint  # noqa: E402


def get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def compute_config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def create_tokenizer_fn():
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    return lambda text: tok.encode(text)


def load_and_split_domain(
    path: str, heldout_frac: float, seed: int, max_chars: int | None = None
) -> tuple[str, str]:
    """Line-level split of one domain file into (train_text, heldout_text)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    n_held = max(1, int(len(lines) * heldout_frac))
    held, train = lines[:n_held], lines[n_held:]
    train_text = "\n".join(train)
    held_text = "\n".join(held)
    if max_chars is not None:
        train_text = train_text[:max_chars]
        held_text = held_text[: max(1000, max_chars // 20)]
    return train_text, held_text


def text_to_windows(
    text: str, tokenizer_fn, seq_len: int, max_windows: int | None = None
) -> torch.Tensor:
    """Tokenize and chunk into [N, seq_len] windows (drop remainder)."""
    ids = tokenizer_fn(text)
    n = len(ids) // seq_len
    if max_windows is not None:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError(f"Text too short for one window of {seq_len}")
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)


def train_specialist(
    model_cfg: dict,
    windows: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> tuple[ExplicitLM, list[float]]:
    torch.manual_seed(seed)
    model = ExplicitLM(
        config=EqLMConfig(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            d_ff=model_cfg["d_ff"],
            max_seq_len=model_cfg["max_seq_len"],
        ),
        n_layers=model_cfg["n_layers"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    loss_curve = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, windows.shape[0], (batch_size,), generator=rng)
        batch = windows[idx].to(device)
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"    step {step + 1}/{steps} loss {loss.item():.3f}", flush=True)
    return model, loss_curve


def build_mixed_stream(
    held_a: torch.Tensor, held_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """50/50 interleave of held-out windows; returns (windows, domain_labels)."""
    n = min(held_a.shape[0], held_b.shape[0])
    windows, labels = [], []
    for i in range(n):
        windows += [held_a[i], held_b[i]]
        labels += [0, 1]
    return torch.stack(windows), torch.tensor(labels)


@torch.no_grad()
def evaluate_systems(
    model_a: ExplicitLM,
    model_b: ExplicitLM,
    windows: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    batch_size: int = 8,
    trace_positions: int = 200,
) -> dict:
    """Per-position NLLs for all four systems + auction traces/cross-check."""
    model_a.eval()
    model_b.eval()
    sums = {k: 0.0 for k in ("S_A", "S_B", "ENS", "AUC")}
    dom_sums = {k: [0.0, 0.0] for k in ("S_A", "S_B", "ENS", "AUC")}
    dom_toks = [0, 0]
    total_toks = 0
    win_a = 0
    payment_sum = 0.0
    traces: list[dict] = []
    auction = TokenAuction(AuctionConfig(auction_type=AuctionType.SECOND_PRICE))
    mismatches = 0

    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device)
        labs = labels[start : start + batch_size]
        logits_a = model_a(batch)[:, :-1, :]
        logits_b = model_b(batch)[:, :-1, :]
        targets = batch[:, 1:]

        logp_a = F.log_softmax(logits_a, dim=-1)
        logp_b = F.log_softmax(logits_b, dim=-1)
        # uniform logit-average ensemble
        logp_ens = F.log_softmax((logits_a + logits_b) / 2.0, dim=-1)

        nll_a = -logp_a.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll_b = -logp_b.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll_ens = -logp_ens.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # auction: bid = own max prob; winner = argmax bid; pay second price
        bid_a = logp_a.max(dim=-1).values.exp()
        bid_b = logp_b.max(dim=-1).values.exp()
        a_wins = bid_a >= bid_b
        nll_auc = torch.where(a_wins, nll_a, nll_b)
        payment = torch.where(a_wins, bid_b, bid_a)

        for k, nll in (("S_A", nll_a), ("S_B", nll_b), ("ENS", nll_ens), ("AUC", nll_auc)):
            sums[k] += nll.sum().item()
            for d in (0, 1):
                mask = labs == d
                if mask.any():
                    dom_sums[k][d] += nll[mask].sum().item()
        for d in (0, 1):
            dom_toks[d] += int((labs == d).sum().item()) * targets.shape[1]
        total_toks += targets.numel()
        win_a += int(a_wins.sum().item())
        payment_sum += payment.sum().item()

        # trace sample + cross-check against the validated mechanism
        if len(traces) < trace_positions:
            probs_a = logp_a[0].exp()
            probs_b = logp_b[0].exp()
            for t in range(min(targets.shape[1], trace_positions - len(traces))):
                bids = torch.stack([bid_a[0, t], bid_b[0, t]]).cpu()
                dists = torch.stack([probs_a[t], probs_b[t]]).cpu()
                res = auction.run_auction(bids, dists)
                vec_winner = 0 if bool(a_wins[0, t]) else 1
                if res.winner_id != vec_winner and bids[0] != bids[1]:
                    mismatches += 1
                traces.append(
                    {
                        "position": len(traces),
                        "bids": [round(float(bids[0]), 6), round(float(bids[1]), 6)],
                        "winner": int(res.winner_id),
                        "payment": round(float(res.payments[res.winner_id]), 6),
                        "target_token": int(targets[0, t]),
                    }
                )

    def ppl(total_nll: float, toks: int) -> float:
        return float(torch.exp(torch.tensor(total_nll / max(toks, 1))))

    return {
        "perplexity_mixed": {k: ppl(v, total_toks) for k, v in sums.items()},
        "perplexity_domain_a": {k: ppl(dom_sums[k][0], dom_toks[0]) for k in sums},
        "perplexity_domain_b": {k: ppl(dom_sums[k][1], dom_toks[1]) for k in sums},
        "auction_win_frac_a": win_a / max(total_toks, 1),
        "auction_mean_payment": payment_sum / max(total_toks, 1),
        "num_tokens": total_toks,
        "trace_mechanism_mismatches": mismatches,
        "traces": traces,
    }


def score_h4(ppls: dict[str, float]) -> str:
    best = min(ppls["S_A"], ppls["S_B"])
    worst = max(ppls["S_A"], ppls["S_B"])
    if ppls["AUC"] < best:
        return "MET"
    if ppls["AUC"] < worst:
        return "PARTIAL"
    return "MISSED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tr = cfg["training"]
    if isinstance(tr.get("lr"), str):
        tr["lr"] = float(tr["lr"])
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["seed"]
    seq_len = cfg["model"]["max_seq_len"]

    seed_file = out_dir / f"results_seed{seed}.json"
    if seed_file.exists():
        print(f"[resume] seed {seed} already complete")
        return

    tokenizer_fn = create_tokenizer_fn()
    print("loading + splitting domains...")
    train_a, held_a = load_and_split_domain(
        cfg["domains"]["a"]["file"], cfg["data"]["heldout_frac"], seed,
        cfg["data"].get("max_chars"),
    )
    train_b, held_b = load_and_split_domain(
        cfg["domains"]["b"]["file"], cfg["data"]["heldout_frac"], seed,
        cfg["data"].get("max_chars"),
    )
    w_train_a = text_to_windows(train_a, tokenizer_fn, seq_len)
    w_train_b = text_to_windows(train_b, tokenizer_fn, seq_len)
    max_eval = cfg["data"].get("max_eval_windows")
    w_held_a = text_to_windows(held_a, tokenizer_fn, seq_len, max_eval)
    w_held_b = text_to_windows(held_b, tokenizer_fn, seq_len, max_eval)
    print(
        f"windows: A train {w_train_a.shape[0]} held {w_held_a.shape[0]} | "
        f"B train {w_train_b.shape[0]} held {w_held_b.shape[0]}"
    )

    results: dict[str, Any] = {
        "experiment": "exp12_auction_decoding",
        "spec": "0008",
        "seed": seed,
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(),
        "domains": {
            "a": cfg["domains"]["a"]["name"],
            "b": cfg["domains"]["b"]["name"],
        },
    }

    specialists = {}
    for name, windows in (("a", w_train_a), ("b", w_train_b)):
        print(f"training specialist {name.upper()} ({cfg['domains'][name]['name']})...")
        t0 = time.time()
        model, curve = train_specialist(
            cfg["model"], windows, tr["steps"], tr["batch_size"], tr["lr"],
            seed, device,
        )
        specialists[name] = model
        results[f"specialist_{name}"] = {
            "num_params": sum(p.numel() for p in model.parameters()),
            "final_loss": curve[-1],
            "train_time_sec": time.time() - t0,
        }
        if cfg.get("save_checkpoints", False):
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            save_checkpoint(model, ckpt_dir / f"specialist_{name}_seed{seed}.pt")

    print("evaluating on mixed-domain stream...")
    mixed, labels = build_mixed_stream(w_held_a, w_held_b)
    ev = evaluate_systems(
        specialists["a"], specialists["b"], mixed, labels, device,
        trace_positions=cfg["eval"].get("trace_positions", 200),
    )
    traces = ev.pop("traces")
    (out_dir / f"traces_seed{seed}.json").write_text(json.dumps(traces, indent=2))
    results["eval"] = ev
    results["h4_score"] = score_h4(ev["perplexity_mixed"])

    seed_file.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["eval"]["perplexity_mixed"], indent=2))
    print(f"H4 score (seed {seed}): {results['h4_score']}")


if __name__ == "__main__":
    main()
