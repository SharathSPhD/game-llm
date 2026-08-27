"""Experiment 13 — H6 (SPEC 0010, ADR 0005): contraction that survives width.

Three TRIZ-derived arms against the exp10 A3 control (BLiMP 0.537 at 121M):

  B1 anytime  : unrolled training with CE on iterates z4/z8/z12
                (P11 — every truncated budget is a usable model).
  B2 trajpen  : standard DEQ training + penalty on a finite-difference local
                Lipschitz estimate along the solve ray (P35 + separation by
                condition — contractive where the solver travels).
  B3 core     : bottleneck-core EqLMCore — equilibrium solved in d_core=256
                between wide explicit encoder/decoder (P24 — capacity outside
                the loop).

All arms are evaluated identically: BLiMP subset + solver telemetry
(certified convergence rate, mean iterations at rel-tol within the same
12-iteration budget) measured with the STANDARD DEQ forward, plus loss,
peak memory, wall time. Uses the exp10 token cache for the identical data
stream.

Usage:
    python exp13_contraction_at_width.py --config configs/exp13_seed42.yaml \
        --output results/exp13_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # repo layout
    from experiments.exp05_eqlm_pretrain import (  # noqa: E402
        compute_config_hash,
        create_gpt2_tokenizer_fn,
        get_git_commit,
    )
except ImportError:  # flat remote job layout
    from exp05_eqlm_pretrain import (  # type: ignore[no-redef]  # noqa: E402
        compute_config_hash,
        create_gpt2_tokenizer_fn,
        get_git_commit,
    )
from kinetic_ai.eval.blimp import evaluate_blimp_subset, load_blimp_subset  # noqa: E402
from kinetic_ai.models.eqlm import (  # noqa: E402
    EqLM,
    EqLMConfig,
    EqLMCore,
    count_params,
    save_checkpoint,
)


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    tr = cfg["training"]
    for k in ("lr", "weight_decay", "grad_clip"):
        if isinstance(tr.get(k), str):
            tr[k] = float(tr[k])
    for arm in cfg["arms"].values():
        for k in ("lambda_c", "gamma"):
            if isinstance(arm.get(k), str):
                arm[k] = float(arm[k])
        mc = arm.get("model_config", {})
        if isinstance(mc.get("deq_tol"), str):
            mc["deq_tol"] = float(mc["deq_tol"])
    return cfg


def build_model(arm_cfg: dict, device: str) -> Any:
    mc = dict(arm_cfg["model_config"])
    kind = arm_cfg["kind"]
    if kind in ("anytime", "trajpen"):
        return EqLM(config=EqLMConfig(**mc)).to(device)
    if kind == "core":
        core_keys = {"d_core", "n_heads_core", "d_ff_core", "n_enc", "n_dec"}
        core_kwargs = {k: mc.pop(k) for k in list(mc) if k in core_keys}
        return EqLMCore(config=EqLMConfig(**mc), **core_kwargs).to(device)
    raise ValueError(f"Unknown arm kind: {kind}")


def ce_loss(logits: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1),
    )


def train_arm(
    arm_name: str,
    arm_cfg: dict,
    cfg: dict,
    tokens: torch.Tensor,
    device: str,
    out_dir: Path,
) -> dict:
    arm_file = out_dir / f"arm_{arm_name}.json"
    if arm_file.exists():
        print(f"[resume] {arm_name} complete, skipping")
        return json.loads(arm_file.read_text())

    tr = cfg["training"]
    seed = cfg["seed"]
    torch.manual_seed(seed)
    model = build_model(arm_cfg, device)
    n_params = count_params(model)
    print(f"[{arm_name}] kind={arm_cfg['kind']} params={n_params:,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
    )
    rng = torch.Generator().manual_seed(seed)
    kind = arm_cfg["kind"]
    sup = arm_cfg.get("supervise_at", [4, 8, 12])
    sup_w = arm_cfg.get("supervise_weights", [0.15, 0.3, 1.0])
    alpha_rng = torch.Generator().manual_seed(seed + 1)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loss_curve: list[dict] = []
    pen_curve: list[float] = []
    t0 = time.time()
    model.train()
    for step in range(tr["num_steps"]):
        idx = torch.randint(0, tokens.shape[0], (cfg["data"]["batch_size"],), generator=rng)
        batch = tokens[idx].to(device)

        if kind == "anytime":
            outs = model.forward_unrolled(batch, supervise_at=sup)
            parts = [
                w * ce_loss(lg, batch)
                for w, (_, lg) in zip(sup_w, outs, strict=True)
            ]
            loss = torch.stack(parts).sum() / sum(sup_w)
        elif kind == "trajpen":
            loss = ce_loss(model(batch), batch)
            alpha = torch.rand(1, generator=alpha_rng).item()
            lhat = model.local_lipschitz(batch, alpha=alpha)
            pen = torch.relu(lhat - arm_cfg["gamma"])
            pen_curve.append(lhat.item())
            loss = loss + arm_cfg["lambda_c"] * pen
        else:  # core
            loss = ce_loss(model(batch), batch)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()
        if step % tr["log_every"] == 0 or step == tr["num_steps"] - 1:
            loss_curve.append({"step": step, "loss": float(loss.item())})
            print(f"  [{arm_name}] step {step} loss {loss.item():.4f}", flush=True)
    train_time = time.time() - t0
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
    )

    # ---- Solver telemetry with the STANDARD DEQ forward (all kinds) ----
    model.eval()
    conv, iters = [], []
    with torch.no_grad():
        for _ in range(cfg["eval"]["telemetry_batches"]):
            idx = torch.randint(0, tokens.shape[0], (8,), generator=rng)
            model(tokens[idx].to(device))
            info = model.deq.last_info if isinstance(model.deq.last_info, dict) else {}
            conv.append(bool(info.get("converged", False)))
            iters.append(int(info.get("iterations", -1)))

    # ---- BLiMP ----
    tokenizer_fn = create_gpt2_tokenizer_fn("gpt2")
    blimp = load_blimp_subset(
        num_phenomena=cfg["eval"]["blimp"]["num_phenomena"],
        pairs_per_phenomenon=cfg["eval"]["blimp"]["pairs_per_phenomenon"],
    )
    eval_res = evaluate_blimp_subset(model, blimp, tokenizer_fn, device=device)

    if cfg.get("save_checkpoints", False):
        ckpt = out_dir / "checkpoints"
        ckpt.mkdir(exist_ok=True)
        save_checkpoint(model, ckpt / f"{arm_name}.pt")

    result = {
        "arm": arm_name,
        "kind": kind,
        "num_params": n_params,
        "blimp_accuracy": eval_res["accuracy"],
        "blimp_num_correct": eval_res["num_correct"],
        "blimp_num_total": eval_res["num_total"],
        "solver_convergence_rate": sum(conv) / max(len(conv), 1),
        "solver_mean_iterations": sum(iters) / max(len(iters), 1),
        "final_loss": loss_curve[-1]["loss"],
        "loss_curve": loss_curve,
        "lipschitz_curve_tail": pen_curve[-20:],
        "peak_memory_mb": peak_mb,
        "train_time_sec": train_time,
    }
    arm_file.write_text(json.dumps(result, indent=2))
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    cache = torch.load(cfg["data"]["token_cache_file"], weights_only=True)
    tokens = cache["tensor"]
    print(f"token stream: {tokens.shape[0]} seqs x {tokens.shape[1]}")

    results: dict[str, Any] = {
        "experiment": "exp13_contraction_at_width",
        "spec": "0010",
        "seed": cfg["seed"],
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(),
        "control": cfg.get("control", {}),
        "arms": {},
    }
    for arm_name, arm_cfg in cfg["arms"].items():
        results["arms"][arm_name] = train_arm(
            arm_name, arm_cfg, cfg, tokens, device, out_dir
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("\n=== exp13 summary ===")
    for name, r in results["arms"].items():
        print(
            f"  {name:8} {r['kind']:8} blimp {r['blimp_accuracy']:.3f} "
            f"conv {r['solver_convergence_rate']:.2f} "
            f"iters {r['solver_mean_iterations']:.1f} "
            f"loss {r['final_loss']:.3f} "
            f"mem {r['peak_memory_mb'] and round(r['peak_memory_mb'])}MB "
            f"time {round(r['train_time_sec'] / 60)}min"
        )


if __name__ == "__main__":
    main()
