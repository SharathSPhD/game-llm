"""Adaptive depth: the thing an explicit stack cannot do (kinetic core, cycle 30).

This returns the programme to its actual subject. The council work established a
conditional eight-point advantage for routing between four separate models, but
that is a systems result, not the paradigm: comparing four models against one is
not a comparison of architectures, and nothing in it requires the depth,
training or decoding of any single model to be an equilibrium computation.

The paradigm claim is a single-model claim and has always been. One weight-tied
block solved to a fixed point, matched in parameters and in compute against a
conventional explicit stack, and asked to be better rather than merely equal.
F24 reached the first half of that: anytime-unrolled training brought the tied
block to a ratio of 0.991 against a twelve-layer explicit transformer at matched
budget, with one seed exceeding it. Parity, not advantage.

What has never been tested is the property that makes the equilibrium
formulation structurally different rather than merely equivalent. An explicit
twelve-layer stack applies twelve layers to every token, always, because its
depth is its architecture. A fixed-point model's depth is a stopping criterion,
so it can iterate until each position's residual falls below a tolerance and
stop there — five iterations on a token whose representation settles quickly,
twenty on one that does not. At a matched *average* number of iterations, the
equilibrium model is therefore spending the same compute unevenly while the
explicit stack must spend it uniformly, and the question this file asks is
whether uneven spending is worth anything.

Three arms on identical evaluation data, all from checkpoints already trained:
the explicit baseline at its fixed twelve layers, the tied block at a uniform
twelve iterations (F24's setting), and the tied block iterating adaptively with
its tolerance calibrated so that its mean iteration count matches twelve. A
fourth arm sweeps the tolerance downward to find what mean depth the adaptive
solver needs to hold parity, which is the compute-saving form of the same claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.eval.blimp import evaluate_blimp_subset, load_blimp_subset  # noqa: E402
from kinetic_ai.models.eqlm import EqLM, EqLMConfig, ExplicitLM  # noqa: E402


@torch.no_grad()
def adaptive_forward(
    model: EqLM, input_ids: torch.Tensor, tol: float, max_iter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterate the tied block, freezing positions once they have converged.

    Each position gets its own stopping time. A position whose residual falls
    below ``tol`` is held fixed while its neighbours continue, so the compute
    spent on a sequence is proportional to how hard its individual tokens are
    rather than to its length times a constant depth. This is the operation an
    explicit stack has no way to express: its depth is baked into its parameter
    list, not decided at inference.

    Returns the logits and the per-position iteration count.
    """
    x = model.embedding(input_ids)
    positions = torch.arange(
        input_ids.shape[1], device=input_ids.device, dtype=torch.long
    )
    x = x + model.pos_embedding(positions)

    z = x
    active = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
    iters = torch.zeros(input_ids.shape[:2], device=input_ids.device)

    for _ in range(max_iter):
        if not bool(active.any()):
            break
        z_new = model.block(z, x)
        # Relative residual per position, which is scale-free and so comparable
        # across positions whose activations differ in magnitude.
        resid = (z_new - z).norm(dim=-1) / (z.norm(dim=-1) + 1e-6)
        # Only unconverged positions advance; the rest keep the state they had.
        upd = active.unsqueeze(-1)
        z = torch.where(upd, z_new, z)
        iters = iters + active.float()
        active = active & (resid > tol)

    h = model.ln_final(z)
    logits = model.lm_head(h) / (model.config.d_model**0.5)
    return logits, iters


def make_adaptive_wrapper(model: EqLM, tol: float, max_iter: int) -> Any:
    """A module exposing the adaptive solve through the standard eval interface.

    The BLiMP harness calls a model and reads logits; wrapping rather than
    modifying the model keeps the uniform-budget arm running exactly the code
    F24 measured, so the two arms differ only in the solve.
    """

    class _Wrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model
            self.config = model.config
            self.iteration_counts: list[float] = []

        def forward(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
            logits, iters = adaptive_forward(self.inner, input_ids, tol, max_iter)
            self.iteration_counts.append(float(iters.mean().item()))
            return logits

    return _Wrapped()


def calibrate_tolerance(
    model: EqLM,
    probe_ids: torch.Tensor,
    target_mean: float,
    max_iter: int,
) -> tuple[float, float]:
    """Find the tolerance whose mean iteration count matches the target.

    Matching mean compute is what makes the comparison a comparison. Bisection
    on the log of the tolerance, because the relationship between tolerance and
    depth is roughly geometric; the probe batch is held separate from the
    evaluation data so the calibration cannot tune to what is being scored.
    """
    lo, hi = -8.0, 0.0
    best = (10 ** ((lo + hi) / 2), 0.0)
    for _ in range(24):
        mid = (lo + hi) / 2
        tol = 10**mid
        _, iters = adaptive_forward(model, probe_ids, tol, max_iter)
        mean_iters = float(iters.mean().item())
        best = (tol, mean_iters)
        if mean_iters > target_mean:
            lo = mid  # looser tolerance -> fewer iterations
        else:
            hi = mid
        if abs(mean_iters - target_mean) < 0.05:
            break
    return best


def load_model(path: Path, device: str) -> Any:
    """Restore a checkpoint, using the architecture it records for itself.

    The class is stored alongside the weights, so it is read rather than guessed
    from the config keys — the two architectures share a config dataclass and a
    guess would silently build the wrong one.
    """
    blob = torch.load(path, map_location=device, weights_only=True)
    state = blob["state_dict"]
    cfg = EqLMConfig(**blob["config_dict"])
    kind = blob.get("model_class", "EqLM")
    if kind == "ExplicitLM":
        model: Any = ExplicitLM(config=cfg, n_layers=int(blob["n_layers"]))
    else:
        model = EqLM(config=cfg)
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqlm", required=True, help="anytime-trained tied-block checkpoint")
    ap.add_argument("--explicit", default=None, help="param-matched explicit baseline")
    ap.add_argument("--out", default="results/scale/exp31_adaptive_depth.json")
    ap.add_argument("--uniform-depth", type=int, default=12)
    ap.add_argument("--max-iter", type=int, default=48)
    ap.add_argument("--phenomena", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(42)

    eqlm = load_model(Path(args.eqlm), device)
    blimp = load_blimp_subset(
        num_phenomena=args.phenomena, pairs_per_phenomenon=args.pairs
    )

    # The same tokenizer the checkpoints were trained under; both arms share it,
    # so any tokenisation quirk affects them identically.
    from kinetic_ai.data.tokenizer import load_gpt2_tokenizer

    gpt2 = load_gpt2_tokenizer()
    if gpt2 is None:
        raise SystemExit("GPT-2 tokenizer unavailable; both arms need it to compare")

    def tokenizer_fn(text: str) -> list[int]:
        return list(gpt2(text)["input_ids"])

    report: dict[str, Any] = {
        "eqlm_checkpoint": args.eqlm,
        "uniform_depth": args.uniform_depth,
        "arms": {},
    }

    # Arm 1: the explicit baseline, if supplied. Its depth is fixed by
    # construction, which is the whole point of the comparison.
    if args.explicit:
        expl = load_model(Path(args.explicit), device)
        res = evaluate_blimp_subset(expl, blimp, tokenizer_fn, device=device)
        report["arms"]["explicit_fixed"] = {
            "blimp": round(res["accuracy"], 4),
            "mean_depth": float(getattr(expl.config, "n_layers", args.uniform_depth)),
        }
        del expl
        torch.cuda.empty_cache()
        print(f"explicit: {report['arms']['explicit_fixed']}", flush=True)

    # Arm 2: the tied block at a uniform budget — F24's measured configuration.
    res = evaluate_blimp_subset(eqlm, blimp, tokenizer_fn, device=device)
    report["arms"]["eqlm_uniform"] = {
        "blimp": round(res["accuracy"], 4),
        "mean_depth": float(args.uniform_depth),
    }
    print(f"eqlm uniform: {report['arms']['eqlm_uniform']}", flush=True)

    # Calibration probe, drawn separately from the scored pairs.
    probe = torch.randint(
        0, max(eqlm.config.vocab_size - 1, 1), (4, 64), device=device
    )
    tol, achieved = calibrate_tolerance(eqlm, probe, float(args.uniform_depth), args.max_iter)
    report["calibration"] = {"tolerance": tol, "probe_mean_iterations": round(achieved, 3)}
    print(f"calibrated tol={tol:.3e} -> mean iters {achieved:.2f}", flush=True)

    # Arm 3: adaptive depth at matched mean compute.
    wrapped = make_adaptive_wrapper(eqlm, tol, args.max_iter)
    res = evaluate_blimp_subset(wrapped, blimp, tokenizer_fn, device=device)
    counts = wrapped.iteration_counts
    report["arms"]["eqlm_adaptive_matched"] = {
        "blimp": round(res["accuracy"], 4),
        "mean_depth": round(sum(counts) / max(len(counts), 1), 3),
        "tolerance": tol,
    }
    print(f"eqlm adaptive: {report['arms']['eqlm_adaptive_matched']}", flush=True)

    # Arm 4: how little depth adaptivity needs to hold its ground.
    sweep = []
    for factor in (2.0, 4.0, 8.0):
        t = tol * factor
        w = make_adaptive_wrapper(eqlm, t, args.max_iter)
        r = evaluate_blimp_subset(w, blimp, tokenizer_fn, device=device)
        c = w.iteration_counts
        sweep.append(
            {
                "tolerance": t,
                "blimp": round(r["accuracy"], 4),
                "mean_depth": round(sum(c) / max(len(c), 1), 3),
            }
        )
        print(f"  sweep tol={t:.2e}: {sweep[-1]}", flush=True)
    report["tolerance_sweep"] = sweep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
