"""Experiment 14 — H5 (SPEC 0009): autoregressive auction decoding.

F22 showed second-price auction SELECTION beats the best single specialist
under teacher forcing. H5 asks whether the advantage survives closed-loop
generation, where each system's own sampled token becomes its next input.

Systems (per seed, reusing the exp12 specialists): S_A, S_B, uniform
logit-average ensemble, second-price auction (bid = own max-prob at the
CURRENT self-generated context; winner emits its argmax token). Greedy
decoding everywhere for determinism.

Primary metric (pre-registered, ADR 0005): mean NLL/token of the generated
continuation under a frozen independent judge LM (exp10 seed-42 124M
explicit), prompt tokens excluded. H5 MET if auction < best single on 3/3
seeds; PARTIAL 2/3; else MISSED. Secondary: domain consistency, 3-gram
repetition rate, per-position winner traces.

Usage:
    python exp14_autoregressive_auction.py --config configs/exp14_seed42.yaml \
        --output results/exp14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp12_auction_decoding import (  # noqa: E402
    compute_config_hash,
    create_tokenizer_fn,
    get_git_commit,
    load_and_split_domain,
    text_to_windows,
)
from kinetic_ai.models.eqlm import load_checkpoint  # noqa: E402


@torch.no_grad()
def generate(
    system: str,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    prefixes: torch.Tensor,
    gen_len: int,
    device: str,
) -> tuple[torch.Tensor, list[dict]]:
    """Greedy closed-loop generation for one system.

    Returns (full sequences [N, prefix+gen], per-position auction traces —
    empty for non-auction systems).
    """
    ctx = prefixes.to(device)
    traces: list[dict] = []
    for pos in range(gen_len):
        logits_a = model_a(ctx)[:, -1, :]
        logits_b = model_b(ctx)[:, -1, :]
        if system == "S_A":
            nxt = logits_a.argmax(dim=-1)
        elif system == "S_B":
            nxt = logits_b.argmax(dim=-1)
        elif system == "ENS":
            nxt = ((logits_a + logits_b) / 2).argmax(dim=-1)
        elif system == "AUC":
            pa = F.softmax(logits_a, dim=-1)
            pb = F.softmax(logits_b, dim=-1)
            bid_a = pa.max(dim=-1).values
            bid_b = pb.max(dim=-1).values
            a_wins = bid_a >= bid_b
            nxt = torch.where(a_wins, logits_a.argmax(dim=-1), logits_b.argmax(dim=-1))
            if pos < 64:  # sample of early positions, first sequence only
                traces.append(
                    {
                        "position": pos,
                        "bids": [bid_a[0].item(), bid_b[0].item()],
                        "winner": 0 if a_wins[0] else 1,
                        "payment": min(bid_a[0].item(), bid_b[0].item()),
                        "token": int(nxt[0].item()),
                    }
                )
        else:
            raise ValueError(f"Unknown system: {system}")
        ctx = torch.cat([ctx, nxt.unsqueeze(1)], dim=1)
    return ctx.cpu(), traces


@torch.no_grad()
def judge_nll(
    judge: torch.nn.Module, sequences: torch.Tensor, prefix_len: int, device: str
) -> float:
    """Mean NLL/token of the generated part under the judge (prompt excluded)."""
    seqs = sequences.to(device)
    logits = judge(seqs)
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = seqs[:, 1:]
    tok_nll = -logp.gather(2, targets.unsqueeze(2)).squeeze(2)  # [N, L-1]
    gen_nll = tok_nll[:, prefix_len - 1 :]  # predictions of generated tokens
    return gen_nll.mean().item()


@torch.no_grad()
def domain_consistency(
    model_own: torch.nn.Module,
    model_other: torch.nn.Module,
    sequences: torch.Tensor,
    prefix_len: int,
    device: str,
) -> float:
    """Fraction of continuations the own-domain specialist scores better."""
    seqs = sequences.to(device)

    def per_seq_nll(m: torch.nn.Module) -> torch.Tensor:
        logp = F.log_softmax(m(seqs)[:, :-1, :], dim=-1)
        tok = -logp.gather(2, seqs[:, 1:].unsqueeze(2)).squeeze(2)
        return tok[:, prefix_len - 1 :].mean(dim=1)

    return (per_seq_nll(model_own) < per_seq_nll(model_other)).float().mean().item()


def repetition_rate(sequences: torch.Tensor, prefix_len: int) -> float:
    """Fraction of repeated 3-grams in generated parts (degeneration check)."""
    total, repeated = 0, 0
    for seq in sequences:
        gen = seq[prefix_len:].tolist()
        grams = [tuple(gen[i : i + 3]) for i in range(len(gen) - 2)]
        total += len(grams)
        repeated += len(grams) - len(set(grams))
    return repeated / max(total, 1)


def score_h5(judge_scores: dict[str, float]) -> str:
    best_single = min(judge_scores["S_A"], judge_scores["S_B"])
    return "MET" if judge_scores["AUC"] < best_single else "MISSED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["seed"]

    result_file = out_dir / f"results_seed{seed}.json"
    if result_file.exists():
        print(f"[resume] seed {seed} already complete")
        return

    model_a = load_checkpoint(cfg["specialists"]["a"]).to(device).eval()
    model_b = load_checkpoint(cfg["specialists"]["b"]).to(device).eval()
    judge = load_checkpoint(cfg["judge"]).to(device).eval()

    tokenizer_fn = create_tokenizer_fn()
    prefix_len = cfg["generation"]["prefix_len"]
    gen_len = cfg["generation"]["gen_len"]
    n_per_domain = cfg["generation"]["prompts_per_domain"]

    print("building held-out prefixes (exp12 split reproduced)...")
    prefixes, prefix_domains = [], []
    for label, dom in enumerate(("a", "b")):
        _, held = load_and_split_domain(
            cfg["domains"][dom]["file"],
            cfg["data"]["heldout_frac"],
            seed,
            cfg["data"].get("max_chars"),
        )
        w = text_to_windows(held, tokenizer_fn, prefix_len, max_windows=n_per_domain)
        prefixes.append(w)
        prefix_domains += [label] * w.shape[0]
    prefix_tensor = torch.cat(prefixes)
    domains_tensor = torch.tensor(prefix_domains)
    print(f"  {prefix_tensor.shape[0]} prefixes x {prefix_len} tokens, gen {gen_len}")

    runs: dict = {}
    judge_scores: dict[str, float] = {}
    auc_traces: list[dict] = []
    for system in ("S_A", "S_B", "ENS", "AUC"):
        print(f"[{system}] generating...")
        seqs, traces = generate(
            system, model_a, model_b, prefix_tensor, gen_len, device
        )
        if system == "AUC":
            auc_traces = traces
        jnll = judge_nll(judge, seqs, prefix_len, device)
        judge_scores[system] = jnll
        cons_a = domain_consistency(
            model_a, model_b, seqs[domains_tensor == 0], prefix_len, device
        )
        cons_b = domain_consistency(
            model_b, model_a, seqs[domains_tensor == 1], prefix_len, device
        )
        runs[system] = {
            "judge_nll_per_token": jnll,
            "domain_consistency": {"a": cons_a, "b": cons_b},
            "repetition_3gram": repetition_rate(seqs, prefix_len),
        }
        print(
            f"  judge NLL {jnll:.4f} | consistency a {cons_a:.2f} b {cons_b:.2f} "
            f"| rep {runs[system]['repetition_3gram']:.3f}"
        )

    results = {
        "experiment": "exp14_autoregressive_auction",
        "spec": "0009",
        "seed": seed,
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(),
        "n_prompts": int(prefix_tensor.shape[0]),
        "systems": runs,
        "h5_score": score_h5(judge_scores),
    }
    result_file.write_text(json.dumps(results, indent=2))
    (out_dir / f"gen_traces_seed{seed}.json").write_text(json.dumps(auc_traces))
    print(f"\nH5 seed {seed}: {results['h5_score']} "
          f"(AUC {judge_scores['AUC']:.4f} vs best single "
          f"{min(judge_scores['S_A'], judge_scores['S_B']):.4f})")


if __name__ == "__main__":
    main()
