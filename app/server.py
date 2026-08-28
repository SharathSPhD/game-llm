"""Phase 3 Equilibrium Lab FastAPI backend.

Endpoints:
  - GET /health: System health check (version, GPU availability)
  - POST /api/solve: Equilibrium Lab solver (MMD, GDA on game matrices)
  - POST /api/qre_path: QRE homotopy path (λ sweep)
  - POST /api/auction: Token auction result
  - GET /api/results: Research findings feed (results/*.json summaries)
  - POST /api/jobs, GET /api/jobs/{id}: Training Studio job queue STUB

Auth: All /api/* paths require Authorization: Bearer <GATEWAY_SECRET> (env).
CORS: Allowed origins from env ALLOWED_ORIGINS (comma-separated).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2Tokenizer

from kinetic_ai.games.payoff import NormalFormGame
from kinetic_ai.games.qre import nash_conv, qre_path
from kinetic_ai.mechanisms.auctions import AuctionConfig, AuctionType, TokenAuction
from kinetic_ai.optim.bregman import NegativeEntropy
from kinetic_ai.optim.mmd import mmd_strategy_update
from kinetic_ai.serve.executor import JobInput, LocalExecutor
from kinetic_ai.serve.hf_publish import (
    compute_exact_param_count,
    get_checkpoint_metadata,
    get_metrics_from_results_json,
    publish_checkpoint_to_hf,
)

# ─── Config ──────────────────────────────────────────────────────────────────

__version__ = "0.1.0-phase3"

# Executor for job queue
# Enable mock mode for testing (when GATEWAY_SECRET is set to "test-secret")
# Mock mode (tests/CI only) is controlled by an explicit env var, read
# dynamically inside LocalExecutor at submit time — never by magic secrets.
executor = LocalExecutor()

# Model cache for playground (ONE model in memory, keyed by checkpoint path)
_playground_model_cache: tuple[str, Any, str] | None = None
_playground_tokenizer: GPT2Tokenizer | None = None

# Allowed experiment templates (ALLOWLIST for security)
EXPERIMENT_TEMPLATES = {
    "exp05_eqlm_pretrain": {
        "name": "EqLM Pretraining (EXP05)",
        "script": "experiments/exp05_eqlm_pretrain.py",
        "config_yaml": "configs/exp05_smoke.yaml",
        "description": "Train EqLM vs ExplicitLM on BabyLM with parameter matching",
    },
    "exp08_solver_aware": {
        "name": "Solver-Aware Loss (EXP08)",
        "script": "experiments/exp08_solver_aware.py",
        "config_yaml": "configs/exp08_smoke.yaml",
        "description": "Test auxiliary loss for learning contraction in DEQ models",
    },
}

# Schema for experiment overrides (numeric ranges, strict validation)
EXPERIMENT_OVERRIDES_SCHEMA = {
    "training.num_steps": {"type": "int", "min": 1, "max": 25000},
    "training.seed": {"type": "int", "min": 1, "max": 2**31 - 1},
    "data.subset_size": {"type": "int", "min": 1000, "max": 100000000},
    "arms.A1.config.dropout": {"type": "float", "min": 0.0, "max": 0.5},
    "arms.A2.config.dropout": {"type": "float", "min": 0.0, "max": 0.5},
    "arms.A3.config.dropout": {"type": "float", "min": 0.0, "max": 0.5},
    "arms.A2.config.deq_max_iter": {"type": "int", "min": 1, "max": 50},
    "arms.A3.config.deq_max_iter": {"type": "int", "min": 1, "max": 50},
    "arms.A1.lambda_aux": {"type": "float", "min": 0.0, "max": 10.0},
    "arms.A2.lambda_aux": {"type": "float", "min": 0.0, "max": 10.0},
    "arms.A3.lambda_aux": {"type": "float", "min": 0.0, "max": 10.0},
}

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Kinetic AI — Phase 3 Backend",
    version=__version__,
    description="Equilibrium Lab + Training Studio",
)

# Get config at request time (to support testing)
def get_gateway_secret() -> str:
    # Fail closed: no default secret. "dev-secret" is explicitly rejected so a
    # placeholder can never authenticate in any environment.
    secret = os.environ.get("GATEWAY_SECRET", "")
    if not secret or secret == "dev-secret":
        raise RuntimeError(
            "GATEWAY_SECRET must be set to a strong value (openssl rand -hex 32)"
        )
    return secret


def get_allowed_origins() -> list[str]:
    # Never wildcard: credentials are allowed, so origins must be explicit.
    origins_str = os.environ.get(
        "ALLOWED_ORIGINS", "https://kinetic.kinetic-ai.workers.dev"
    )
    return [o.strip() for o in origins_str.split(",") if o.strip() and o.strip() != "*"]


def get_results_dir() -> str:
    return os.environ.get("RESULTS_DIR", "./results")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ─── Auth ─────────────────────────────────────────────────────────────────────


def require_bearer_auth(authorization: str | None = Header(None)) -> None:
    """Require Authorization: Bearer <GATEWAY_SECRET>."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, _, credentials = authorization.partition(" ")
    gateway_secret = get_gateway_secret()
    if scheme.lower() != "bearer" or credentials != gateway_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )


# ─── Models ──────────────────────────────────────────────────────────────────


@dataclass
class TrajectoryPoint:
    """Single step in a solve trajectory."""

    step: int
    nash_conv: float
    utility_1: float
    utility_2: float
    strategy_1: list[float]
    strategy_2: list[float]


@dataclass
class SolveResponse:
    """Result from /api/solve."""

    game: str
    method: str
    steps_run: int
    trajectory: list[TrajectoryPoint]
    final_strategy_1: list[float]
    final_strategy_2: list[float]
    final_nash_conv: float
    final_utility_1: float
    final_utility_2: float


@dataclass
class QREPathResponse:
    """Result from /api/qre_path."""

    game: str
    lambda_min: float
    lambda_max: float
    path: list[dict[str, float | list[float]]]


@dataclass
class AuctionResponse:
    """Result from /api/auction."""

    winner_id: int
    output_distribution: list[float]
    payments: list[float]
    sampled_token: int


@dataclass
class TokenInfo:
    """Per-token info from playground generation."""

    token_str: str
    solver_iters: int | None


@dataclass
class PlaygroundGenerateResponse:
    """Result from /api/playground/generate."""

    text: str
    tokens: list[TokenInfo]
    mean_iters: float
    wall_ms: float


# ─── Helpers ─────────────────────────────────────────────────────────────────


def get_game_by_name(name: str) -> NormalFormGame:
    """Load game by name."""
    if name == "rps":
        # Rock-Paper-Scissors
        return NormalFormGame(
            payoff_1=torch.tensor([
                [0.0, -1.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0],
            ]),
            payoff_2=torch.tensor([
                [0.0, 1.0, -1.0],
                [-1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0],
            ]),
            name="rps",
        )
    elif name == "matching_pennies":
        # Matching Pennies (zero-sum)
        return NormalFormGame(
            payoff_1=torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
            payoff_2=torch.tensor([[-1.0, 1.0], [1.0, -1.0]]),
            name="matching_pennies",
        )
    elif name == "biased_rps":
        # Biased RPS (nontrivial zero-sum)
        return NormalFormGame(
            payoff_1=torch.tensor([
                [0.0, -0.8, 1.2],
                [1.0, 0.0, -1.2],
                [-0.9, 1.1, 0.0],
            ]),
            payoff_2=-torch.tensor([
                [0.0, -0.8, 1.2],
                [1.0, 0.0, -1.2],
                [-0.9, 1.1, 0.0],
            ]),
            name="biased_rps",
        )
    elif name == "kuhn":
        # Kuhn Poker (extensive-form; we use a simplified normal-form reduction)
        # This is a placeholder; real Kuhn poker requires extensive-form solver
        return NormalFormGame(
            payoff_1=torch.tensor([
                [0.0, -1.0, 0.5],
                [1.0, 0.0, -0.5],
                [-0.5, 0.5, 0.0],
            ]),
            payoff_2=torch.tensor([
                [0.0, 1.0, -0.5],
                [-1.0, 0.0, 0.5],
                [0.5, -0.5, 0.0],
            ]),
            name="kuhn",
        )
    else:
        raise ValueError(f"Unknown game: {name}")


def downsample_trajectory(
    trajectory: list[TrajectoryPoint],
    max_points: int = 500,
) -> list[TrajectoryPoint]:
    """Downsample trajectory to max_points."""
    if len(trajectory) <= max_points:
        return trajectory
    step = len(trajectory) // max_points
    return [trajectory[i] for i in range(0, len(trajectory), step)]


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """System health check."""
    return {
        "status": "ok",
        "version": __version__,
        "gpu_available": torch.cuda.is_available(),
        # Note: DO NOT allocate GPU; just check availability
    }


@app.post("/api/solve")
async def solve(
    body: dict,
    authorization: str | None = Header(None),
) -> SolveResponse:
    """Equilibrium Lab solver.

    Request body:
        {
            "game": "rps" | "matching_pennies" | "biased_rps" | "kuhn",
            "method": "mmd_fixed" | "mmd_rnd" | "gda",
            "lr": float (learning rate),
            "tau": float (magnetic strength),
            "steps": int (<=5000),
            "seed": int
        }

    Returns:
        Trajectory (downsampled to <=500 points), final strategies, NashConv.
    """
    require_bearer_auth(authorization)

    # Validate input
    game_name = body.get("game", "rps")
    method = body.get("method", "mmd_fixed")
    lr = float(body.get("lr", 0.1))
    tau = float(body.get("tau", 0.1))  # Default 0.1, max 0.5 for stability
    tau = min(tau, 0.5)  # Clamp tau to avoid divergence with magnetic term too strong
    steps = min(int(body.get("steps", 100)), 5000)
    seed = int(body.get("seed", 42))

    torch.manual_seed(seed)

    # Load game (with error handling)
    try:
        game = get_game_by_name(game_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Initialize strategies with small random perturbation
    # This helps escape the Nash equilibrium when optimization starts
    n1, n2 = game.num_actions_1, game.num_actions_2
    torch.manual_seed(seed)  # Ensure reproducibility for strategy init
    noise_1 = 1.0 + 0.1 * torch.randn(n1)
    noise_2 = 1.0 + 0.1 * torch.randn(n2)
    sigma_1 = torch.clamp(noise_1, min=1e-8)
    sigma_2 = torch.clamp(noise_2, min=1e-8)
    sigma_1 = sigma_1 / sigma_1.sum()
    sigma_2 = sigma_2 / sigma_2.sum()

    trajectory: list[TrajectoryPoint] = []
    bregman = NegativeEntropy()

    if method == "mmd_fixed":
        # Fixed reference (initial uniform)
        ref_1, ref_2 = sigma_1.clone(), sigma_2.clone()

        for t in range(steps):
            # Sequential MMD updates (player 1, then player 2 sees the update)
            # This is more stable than simultaneous updates
            g1 = game.utility_gradient(1, sigma_1, sigma_2)
            sigma_1 = mmd_strategy_update(sigma_1, g1, ref_1, bregman, lr, tau)
            sigma_1 = F.softmax(torch.log(torch.clamp(sigma_1, min=1e-8)), dim=0)

            g2 = game.utility_gradient(2, sigma_2, sigma_1)
            sigma_2 = mmd_strategy_update(sigma_2, g2, ref_2, bregman, lr, tau)
            sigma_2 = F.softmax(torch.log(torch.clamp(sigma_2, min=1e-8)), dim=0)

            # Log trajectory (downsample later)
            u1_val, u2_val = game.expected_payoff(sigma_1, sigma_2)
            nc = nash_conv(game, sigma_1, sigma_2)
            trajectory.append(
                TrajectoryPoint(
                    step=t,
                    nash_conv=float(nc),
                    utility_1=float(u1_val),
                    utility_2=float(u2_val),
                    strategy_1=sigma_1.tolist(),
                    strategy_2=sigma_2.tolist(),
                )
            )

    elif method == "mmd_rnd":
        # Random reference resets (Regularized Nash Dynamics)
        reset_freq = max(1, steps // 10)
        ref_1 = sigma_1.clone()
        ref_2 = sigma_2.clone()

        for t in range(steps):
            if t % reset_freq == 0:
                ref_1 = sigma_1.clone()
                ref_2 = sigma_2.clone()

            # Sequential MMD updates (player 1, then player 2 sees the update)
            g1 = game.utility_gradient(1, sigma_1, sigma_2)
            sigma_1 = mmd_strategy_update(sigma_1, g1, ref_1, bregman, lr, tau)
            sigma_1 = F.softmax(torch.log(torch.clamp(sigma_1, min=1e-8)), dim=0)

            g2 = game.utility_gradient(2, sigma_2, sigma_1)
            sigma_2 = mmd_strategy_update(sigma_2, g2, ref_2, bregman, lr, tau)
            sigma_2 = F.softmax(torch.log(torch.clamp(sigma_2, min=1e-8)), dim=0)

            u1_val, u2_val = game.expected_payoff(sigma_1, sigma_2)
            nc = nash_conv(game, sigma_1, sigma_2)
            trajectory.append(
                TrajectoryPoint(
                    step=t,
                    nash_conv=float(nc),
                    utility_1=float(u1_val),
                    utility_2=float(u2_val),
                    strategy_1=sigma_1.tolist(),
                    strategy_2=sigma_2.tolist(),
                )
            )

    elif method == "gda":
        # Gradient Descent Ascent (baseline)
        for t in range(steps):
            u1 = game.payoff_1 @ sigma_2
            u2 = game.payoff_2.t() @ sigma_1

            grad_1 = u1 - (sigma_1 @ game.payoff_1 @ sigma_2)
            grad_2 = u2 - (sigma_2 @ game.payoff_2.t() @ sigma_1)

            sigma_1 = sigma_1 + lr * grad_1
            sigma_2 = sigma_2 - lr * grad_2

            sigma_1 = F.softmax(torch.log(torch.clamp(sigma_1, min=1e-8)), dim=0)
            sigma_2 = F.softmax(torch.log(torch.clamp(sigma_2, min=1e-8)), dim=0)

            u1_val, u2_val = game.expected_payoff(sigma_1, sigma_2)
            nc = nash_conv(game, sigma_1, sigma_2)
            trajectory.append(
                TrajectoryPoint(
                    step=t,
                    nash_conv=float(nc),
                    utility_1=float(u1_val),
                    utility_2=float(u2_val),
                    strategy_1=sigma_1.tolist(),
                    strategy_2=sigma_2.tolist(),
                )
            )

    else:
        raise ValueError(f"Unknown method: {method}")

    # Downsample and return
    trajectory = downsample_trajectory(trajectory, max_points=500)
    final = trajectory[-1] if trajectory else TrajectoryPoint(
        step=0, nash_conv=0.0, utility_1=0.0, utility_2=0.0,
        strategy_1=sigma_1.tolist(), strategy_2=sigma_2.tolist()
    )

    return SolveResponse(
        game=game_name,
        method=method,
        steps_run=steps,
        trajectory=trajectory,
        final_strategy_1=final.strategy_1,
        final_strategy_2=final.strategy_2,
        final_nash_conv=final.nash_conv,
        final_utility_1=final.utility_1,
        final_utility_2=final.utility_2,
    )


@app.post("/api/qre_path")
async def qre_path_endpoint(
    body: dict,
    authorization: str | None = Header(None),
) -> QREPathResponse:
    """QRE homotopy path (λ sweep).

    Request body:
        {
            "game": "rps" | ...,
            "lambda_min": float,
            "lambda_max": float,
            "n_points": int (<=50)
        }

    Returns:
        Path of (λ, strategy_1, strategy_2) along QRE correspondence.
    """
    require_bearer_auth(authorization)

    game_name = body.get("game", "rps")
    lambda_min = float(body.get("lambda_min", 0.1))
    lambda_max = float(body.get("lambda_max", 10.0))
    n_points = min(int(body.get("n_points", 20)), 50)

    game = get_game_by_name(game_name)

    # Generate rationality values (log scale from lambda_min to lambda_max)
    rationality_values = torch.logspace(
        torch.log10(torch.tensor(lambda_min)).item(),
        torch.log10(torch.tensor(lambda_max)).item(),
        n_points,
    ).tolist()

    path = qre_path(
        game,
        rationality_values=rationality_values,
    )

    return QREPathResponse(
        game=game_name,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        path=[
            {
                "rationality": float(p.rationality),
                "strategy_1": p.strategy_1.tolist(),
                "strategy_2": p.strategy_2.tolist(),
                "nash_conv": float(p.nash_conv),
            }
            for p in path
        ],
    )


@app.get("/api/auction/traces")
async def auction_traces_list(
    authorization: str | None = Header(None),
) -> dict:
    """List seeds with real exp12 auction decode traces available."""
    require_bearer_auth(authorization)
    exp12 = Path(get_results_dir()) / "exp12"
    seeds = sorted(
        int(p.stem.split("traces_seed")[1])
        for p in exp12.glob("traces_seed*.json")
        if p.stem.split("traces_seed")[1].isdigit()
    ) if exp12.is_dir() else []
    return {"seeds": seeds}


@app.get("/api/auction/traces/{seed}")
async def auction_traces_get(
    seed: int,
    limit: int = 200,
    authorization: str | None = Header(None),
) -> dict:
    """Real per-token auction traces (bids/winner/payment) from exp12,
    plus the run's summary metrics — feeds the Auction playground with
    real model data (SPEC 0008 app tie-in)."""
    require_bearer_auth(authorization)
    exp12 = Path(get_results_dir()) / "exp12"
    trace_file = exp12 / f"traces_seed{seed}.json"
    if not trace_file.is_file():
        raise HTTPException(status_code=404, detail=f"No traces for seed {seed}")
    traces = json.loads(trace_file.read_text())[: max(0, min(limit, 1000))]
    summary: dict = {}
    results_file = exp12 / f"results_seed{seed}.json"
    if results_file.is_file():
        r = json.loads(results_file.read_text())
        summary = {
            "h4_score": r.get("h4_score"),
            "domains": r.get("domains"),
            "perplexity_mixed": (r.get("eval") or {}).get("perplexity_mixed"),
            "auction_win_frac_a": (r.get("eval") or {}).get("auction_win_frac_a"),
        }
    return {"seed": seed, "traces": traces, "summary": summary}


@app.post("/api/auction")
async def auction(
    body: dict,
    authorization: str | None = Header(None),
) -> AuctionResponse:
    """Token auction result.

    Request body:
        {
            "bids": [float, float, ...],
            "agent_distributions": [[float, ...], [float, ...], ...],
            "auction_type": "second_price" | "weighted_aggregation",
            "vocab_size": int (<=100),
            "seed": int
        }

    Returns:
        Winner, payments, output distribution, sampled token.
    """
    require_bearer_auth(authorization)

    bids = torch.tensor(body.get("bids", [1.0, 1.0]), dtype=torch.float32)
    dists = torch.tensor(
        body.get("agent_distributions", [[1.0, 0.0], [0.0, 1.0]]),
        dtype=torch.float32,
    )
    auction_type_str = body.get("auction_type", "second_price")
    vocab_size = int(body.get("vocab_size", 100))
    seed = int(body.get("seed", 42))

    torch.manual_seed(seed)

    # Validate dimensions
    if dists.shape[0] != bids.shape[0]:
        raise ValueError(
            f"Bid count {bids.shape[0]} != agent count {dists.shape[0]}"
        )
    if dists.shape[1] > vocab_size:
        raise ValueError(
            f"Distribution vocab {dists.shape[1]} > vocab_size {vocab_size}"
        )

    # Pad distributions to vocab_size if needed
    if dists.shape[1] < vocab_size:
        pad_size = vocab_size - dists.shape[1]
        dists = torch.cat(
            [dists, torch.zeros(dists.shape[0], pad_size)], dim=1
        )

    # Normalize
    bids = bids / (bids.sum() + 1e-8)
    dists = dists / (dists.sum(dim=1, keepdim=True) + 1e-8)

    # Run auction
    auction_type = (
        AuctionType.SECOND_PRICE
        if auction_type_str == "second_price"
        else AuctionType.WEIGHTED_AGGREGATION
    )
    config = AuctionConfig(auction_type=auction_type)
    auction_mech = TokenAuction(config)
    result = auction_mech.run_auction(bids, dists)

    return AuctionResponse(
        winner_id=int(result.winner_id),
        output_distribution=result.output_distribution.tolist(),
        payments=result.payments.tolist(),
        sampled_token=int(result.sampled_token),
    )


@app.post("/api/playground/generate")
async def playground_generate(
    body: dict,
    authorization: str | None = Header(None),
) -> dict:
    """Generate text with playground model (equilibrium lens).

    Request body:
        {
            "checkpoint_path": "exp10_probe/checkpoints/a2.pt",
            "prompt": "The future of AI is",
            "max_new_tokens": 32,
            "warm_start": false,
            "solver_budget": 6
        }

    Returns:
        {
            "text": "Generated text here",
            "tokens": [
                {"token_str": "The", "solver_iters": 6},
                {"token_str": " future", "solver_iters": 4},
                ...
            ],
            "mean_iters": 5.2,
            "wall_ms": 234.5
        }

    Raises:
        400: Invalid checkpoint_path, traversal attempt, prompt too long
        401: Missing/invalid auth
        503: Model load failed
    """
    require_bearer_auth(authorization)

    checkpoint_path_str = body.get("checkpoint_path", "")
    prompt = body.get("prompt", "")
    max_new_tokens = min(int(body.get("max_new_tokens", 16)), 64)
    warm_start = bool(body.get("warm_start", False))
    solver_budget = min(max(int(body.get("solver_budget", 6)), 4), 64)
    temperature = min(max(float(body.get("temperature", 0.8)), 0.0), 2.0)
    top_k = min(max(int(body.get("top_k", 50)), 0), 200)

    # Validate inputs
    if not checkpoint_path_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="checkpoint_path required",
        )

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt required",
        )

    if len(prompt) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt exceeds 500 characters",
        )

    # Security: prevent traversal attacks
    results_dir = Path(get_results_dir()).resolve()
    full_path = (results_dir / checkpoint_path_str).resolve()

    try:
        full_path.relative_to(results_dir)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal detected. checkpoint_path must be inside results/",
        ) from e

    if not full_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Checkpoint not found: {checkpoint_path_str}",
        )

    start_time = time.time()

    try:
        # Get tokenizer (load once)
        global _playground_tokenizer
        if _playground_tokenizer is None:
            _playground_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Load model (cache ONE model)
        global _playground_model_cache
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if _playground_model_cache is None or _playground_model_cache[0] != str(full_path):
            # Load via the canonical (weights_only-safe, tested) loader.
            from kinetic_ai.models.eqlm import load_checkpoint

            model = load_checkpoint(full_path)
            model_class_name = type(model).__name__
            model.eval()
            model.to(device)

            # Cache model (path, model, class name)
            _playground_model_cache = (str(full_path), model, model_class_name)

        model = _playground_model_cache[1]
        model_class_name = _playground_model_cache[2]

        # Tokenize prompt (cap sequence to the model's context window)
        token_ids = _playground_tokenizer.encode(prompt, return_tensors="pt")[0]
        max_ctx = int(getattr(model.config, "max_seq_len", 128))
        budget_room = max_ctx - max_new_tokens - 1
        if budget_room < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_new_tokens too large for the model's context window",
            )
        token_ids = token_ids[-budget_room:].to(device)

        solver_iters_list: list[int | None] = []
        if model_class_name == "EqLM":
            from kinetic_ai.models.eqlm import EqLM as _EqLM

            assert isinstance(model, _EqLM)
            # Equilibrium path: the solver budget is the "think-harder" dial and
            # warm_start reuses the previous token's equilibrium (H1'a).
            original_max_iter = model.deq.config.max_iter
            original_depth = model.config.deq_max_iter
            model.deq.config.max_iter = solver_budget
            # In unrolled decode mode the budget dial IS the unroll depth.
            model.config.deq_max_iter = solver_budget
            try:
                output_ids, gen_info = model.generate(
                    token_ids.unsqueeze(0),
                    max_new_tokens,
                    warm_start=warm_start,
                    return_iter_counts=True,
                    temperature=temperature,
                    top_k=top_k,
                )
            finally:
                model.deq.config.max_iter = original_max_iter
                model.config.deq_max_iter = original_depth
            generated_ids = [int(t) for t in output_ids[0, token_ids.shape[0]:]]
            raw_iters = gen_info.get("iter_counts", [])
            iters = list(raw_iters) if isinstance(raw_iters, (list, tuple)) else []
            solver_iters_list = [int(i) for i in iters[: len(generated_ids)]]
            solver_iters_list += [None] * (len(generated_ids) - len(solver_iters_list))
        else:
            # Explicit stack: fixed depth, no solver iterations to report.
            generated_ids = []
            current_ids = token_ids.clone()
            from kinetic_ai.models.eqlm import sample_next_token

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = model(current_ids.unsqueeze(0))[0, -1, :]
                next_token_id = int(
                    sample_next_token(logits.unsqueeze(0), temperature, top_k)[0, 0].item()
                )
                generated_ids.append(next_token_id)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([next_token_id], device=device)]
                )
                solver_iters_list.append(None)

        # Decode tokens
        token_strings = []
        for token_id in generated_ids:
            token_str = _playground_tokenizer.decode([token_id])
            token_strings.append(token_str)

        # Compute stats
        valid_iters = [it for it in solver_iters_list if it is not None]
        mean_iters = sum(valid_iters) / len(valid_iters) if valid_iters else 0.0

        wall_ms = (time.time() - start_time) * 1000

        # Decode full output
        output_ids = torch.cat([token_ids, torch.tensor(generated_ids, device=device)])
        full_text = _playground_tokenizer.decode(output_ids.tolist())

        return {
            "text": full_text,
            "tokens": [
                {"token_str": ts, "solver_iters": it}
                for ts, it in zip(token_strings, solver_iters_list, strict=True)
            ],
            "mean_iters": float(mean_iters),
            "wall_ms": float(wall_ms),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Generation failed: {str(e)}",
        ) from e


@app.get("/api/results")
async def get_results(
    authorization: str | None = Header(None),
) -> dict:
    """Research findings feed (results/*.json summaries).

    Returns:
        List of all results JSON files found in RESULTS_DIR.
    """
    require_bearer_auth(authorization)

    results = []
    results_path = Path(get_results_dir())

    if results_path.exists():
        for json_file in sorted(results_path.glob("*/results.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    results.append({
                        "experiment": json_file.parent.name,
                        "data": data,
                    })
            except (json.JSONDecodeError, OSError):
                pass

    return {"results": results}


@app.get("/api/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    authorization: str | None = Header(None),
) -> dict:
    """Get job status and result.

    Returns:
        {"status": "queued|running|completed|failed", "result": ...}
    """
    require_bearer_auth(authorization)

    status_str = executor.status(job_id)
    output = executor.result(job_id)

    return {
        "job_id": job_id,
        "status": status_str,
        "result": output.result,
        "error": output.error,
    }


# ─── Experiment Studio Endpoints ─────────────────────────────────────────────


@app.get("/api/experiments")
async def list_experiments(
    authorization: str | None = Header(None),
) -> dict:
    """List available experiment templates.

    Returns:
        [
            {
                "id": "exp05_eqlm_pretrain",
                "name": "EqLM Pretraining (EXP05)",
                "description": "...",
                "config_yaml": "configs/exp05_smoke.yaml",
            },
            ...
        ]
    """
    require_bearer_auth(authorization)

    templates = []
    for template_id, info in EXPERIMENT_TEMPLATES.items():
        # Find default config for this template
        config_dir = Path(get_results_dir()).parent / "configs"
        default_config = None
        for candidate in ["smoke", "dry_run"]:
            candidate_path = config_dir / f"{template_id}_{candidate}.yaml"
            if candidate_path.exists():
                default_config = str(candidate_path.relative_to(Path.cwd()))
                break

        if not default_config:
            # Fallback: list first config matching template_id
            for cfg_file in config_dir.glob(f"{template_id}*.yaml"):
                default_config = str(cfg_file.relative_to(Path.cwd()))
                break

        templates.append({
            "id": template_id,
            "name": info["name"],
            "description": info["description"],
            "config_yaml": default_config,
            "script": info["script"],
        })

    return {"templates": templates}


def _validate_and_apply_overrides(
    config: dict,  # type: ignore[type-arg]
    overrides: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate overrides against schema and apply to config dict.

    Returns:
        (success: bool, error_msg: str | None)
    """
    # Validate: all override keys must be in schema
    for key in overrides:
        if key not in EXPERIMENT_OVERRIDES_SCHEMA:
            return False, f"Unknown override key: {key}"

    # Validate types and ranges
    for key, value in overrides.items():
        schema: dict[str, Any] = EXPERIMENT_OVERRIDES_SCHEMA[key]  # type: ignore[assignment]
        value_type: str = schema["type"]  # type: ignore[index]

        # Type check
        if value_type == "int":
            if not isinstance(value, int):
                return False, f"{key}: expected int, got {type(value).__name__}"
            if "min" in schema and value < schema["min"]:  # type: ignore[operator]
                return False, f"{key}: {value} < minimum {schema['min']}"
            if "max" in schema and value > schema["max"]:  # type: ignore[operator]
                return False, f"{key}: {value} > maximum {schema['max']}"
        elif value_type == "float":
            if not isinstance(value, (int, float)):
                return False, f"{key}: expected float, got {type(value).__name__}"
            fval = float(value)
            if "min" in schema and fval < schema["min"]:  # type: ignore[operator]
                return False, f"{key}: {fval} < minimum {schema['min']}"
            if "max" in schema and fval > schema["max"]:  # type: ignore[operator]
                return False, f"{key}: {fval} > maximum {schema['max']}"

    # Apply overrides to config
    for key, value in overrides.items():
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return True, None


@app.post("/api/jobs")
async def submit_job_experiment(
    body: dict,
    authorization: str | None = Header(None),
) -> dict:
    """Submit experiment job or generic job.

    For experiment jobs:
        {
            "type": "experiment",
            "template_id": "exp05_eqlm_pretrain" | "exp08_solver_aware",
            "overrides": {
                "training.num_steps": 1000,
                "training.seed": 42,
                ...
            }
        }

    For generic jobs (legacy):
        {
            "type": "noop_demo" | ...,
            "params": {...}
        }

    Returns:
        {"job_id": str}

    Raises:
        400: Invalid overrides
        409: GPU locked
        503: Job submission failed
    """
    require_bearer_auth(authorization)

    job_type = body.get("type", "noop_demo")

    # Special handling for experiment jobs
    if job_type == "experiment":
        template_id = body.get("template_id")
        overrides = body.get("overrides", {})

        if not template_id or template_id not in EXPERIMENT_TEMPLATES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid template_id. Allowed: {list(EXPERIMENT_TEMPLATES.keys())}",
            )

        # Validate overrides
        valid, error = _validate_and_apply_overrides({}, overrides)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid overrides: {error}",
            )

        # Load base config
        config_yaml_path = EXPERIMENT_TEMPLATES[template_id]["config_yaml"]
        config_path = Path(config_yaml_path)

        if not config_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Base config not found: {config_yaml_path}",
            )

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Apply overrides
        _validate_and_apply_overrides(config, overrides)

        # Create job output directory
        results_dir = Path(get_results_dir())
        job_id = str(__import__("uuid").uuid4())
        job_output_dir = results_dir / "studio_runs" / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Write resolved config
        resolved_config_path = job_output_dir / "config.yaml"
        with open(resolved_config_path, "w") as f:
            yaml.dump(config, f)

        # Submit job with the same job_id
        job = JobInput(
            id=job_id,
            type="experiment",
            params={
                "template_id": template_id,
                "resolved_config_path": str(resolved_config_path),
                "output_dir": str(job_output_dir),
            },
        )

        try:
            returned_job_id = executor.submit(job)
            return {"job_id": returned_job_id}
        except RuntimeError as e:
            if "GPU is locked" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(e),
                ) from e
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e

    # Fall back to generic job submission
    params = body.get("params", {})
    job = JobInput(type=job_type, params=params)

    try:
        job_id = executor.submit(job)
        return {"job_id": job_id}
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e


@app.get("/api/jobs/{job_id}/log")
async def get_job_log(
    job_id: str,
    offset: int = 0,
    authorization: str | None = Header(None),
) -> dict:
    """Get incremental log for a job (poll-based streaming).

    Args:
        job_id: Job ID
        offset: Line offset (for incremental polling)

    Returns:
        {
            "job_id": str,
            "lines": [str],
            "total_lines": int,
            "offset": int
        }
    """
    require_bearer_auth(authorization)

    # Find log file
    results_dir = Path(get_results_dir())
    log_file = results_dir / "studio_runs" / job_id / "run.log"

    if not log_file.exists():
        # Try generic job dir
        log_file = results_dir / job_id / "run.log"

    if not log_file.exists():
        return {
            "job_id": job_id,
            "lines": [],
            "total_lines": 0,
            "offset": offset,
        }

    try:
        with open(log_file) as f:
            all_lines = f.readlines()

        # Return lines from offset onwards
        lines_to_return = all_lines[offset : offset + 100]
        lines_to_return = [line.rstrip("\n") for line in lines_to_return]

        return {
            "job_id": job_id,
            "lines": lines_to_return,
            "total_lines": len(all_lines),
            "offset": offset,
        }
    except OSError:
        return {
            "job_id": job_id,
            "lines": [],
            "total_lines": 0,
            "offset": offset,
        }


@app.get("/api/runs")
async def get_runs_registry(
    authorization: str | None = Header(None),
) -> dict:
    """Get registry of completed runs.

    Walks results/ and studio_runs/ for results.json files.
    Returns:
        [
            {
                "dir": "results/exp05_...",
                "experiment": "exp05_eqlm_pretrain",
                "config_hash": "abc123...",
                "git_commit": "deadbeef...",
                "metrics": {...}
            },
            ...
        ]
    """
    require_bearer_auth(authorization)

    runs = []
    results_dir = Path(get_results_dir())

    if results_dir.exists():
        # Scan results/ and studio_runs/
        for results_json in results_dir.rglob("results.json"):
            try:
                with open(results_json) as f:
                    data = json.load(f)

                # Extract metadata
                run_dir = results_json.parent
                experiment = data.get("experiment", "unknown")
                config_hash = data.get("config_hash", "unknown")
                git_commit = data.get("git_commit", "unknown")
                metrics = {}

                # Defensively extract key metrics
                if "metrics" in data:
                    metrics = data["metrics"]

                # Try to make path relative to cwd, fallback to absolute
                try:
                    rel_path = str(run_dir.relative_to(Path.cwd()))
                except ValueError:
                    rel_path = str(run_dir)

                runs.append({
                    "dir": rel_path,
                    "experiment": experiment,
                    "config_hash": config_hash,
                    "git_commit": git_commit,
                    "metrics": metrics,
                })
            except (json.JSONDecodeError, OSError):
                pass

    return {"runs": runs}


# ─── Models Registry (Capability 2) ──────────────────────────────────────────


# Cache state: (timestamp, result)
_model_registry_cache: tuple[float, list[dict[str, Any]]] | None = None
_CACHE_TTL_SECONDS = 30


def scan_models_registry() -> list[dict[str, Any]]:
    """Scan results/**/*.pt checkpoints and return metadata with caching.

    Returns:
        List of checkpoint metadata dicts with keys:
          - path: relative path (e.g., "exp09/checkpoints/model.pt")
          - size_mb: file size in MB
          - config: checkpoint config dict
          - model_class: "EqLM" or "ExplicitLM"
          - params_estimate: estimated parameter count
          - run: dict with config_sha and git_commit from sibling results.json
    """
    global _model_registry_cache

    now = time.time()
    if _model_registry_cache is not None:
        timestamp, cached_result = _model_registry_cache
        if now - timestamp < _CACHE_TTL_SECONDS:
            return cached_result

    results = []
    results_dir = Path(get_results_dir())

    if not results_dir.exists():
        _model_registry_cache = (now, [])
        return []

    # Find all .pt files in results directory (bounded depth: max 4 levels)
    # Pattern: results/exp*/checkpoints/*.pt or similar
    for pt_file in sorted(results_dir.rglob("*.pt")):
        # Security: ensure file is within results/
        try:
            relative_path = pt_file.relative_to(results_dir)
        except ValueError:
            continue  # Outside results/

        # Skip if depth too deep (prevent traversal issues)
        if len(relative_path.parts) > 4:
            continue

        # Skip if not in a reasonable checkpoint path
        if "checkpoint" not in str(pt_file).lower() and "model" not in str(pt_file).lower():
            continue

        try:
            # Get file size
            size_bytes = pt_file.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            # Load metadata without instantiating model
            metadata = get_checkpoint_metadata(pt_file)
            config = metadata.get("config")
            model_class = metadata.get("model_class", "unknown")

            # Compute exact parameter count from state_dict
            try:
                params_estimate = compute_exact_param_count(pt_file)
            except Exception:
                # Fallback: 0 if computation fails
                params_estimate = 0

            # Extract metrics from sibling results.json
            run_dir = pt_file.parent.parent  # checkpoint is in run_dir/checkpoints/
            run_metrics = get_metrics_from_results_json(run_dir)
            config_sha = run_metrics.pop("config_sha", "unknown")
            git_commit = run_metrics.pop("git_commit", "unknown")

            results.append({
                "path": str(relative_path),
                "size_mb": round(size_mb, 2),
                "config": {
                    "d_model": getattr(config, "d_model", None),
                    "n_heads": getattr(config, "n_heads", None),
                    "d_ff": getattr(config, "d_ff", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                    "map_form": getattr(config, "map_form", None),
                },
                "model_class": model_class,
                "params_estimate": params_estimate,
                "run": {
                    "config_sha": config_sha,
                    "git_commit": git_commit,
                },
            })
        except Exception:
            # Skip checkpoints that fail to load
            continue

    _model_registry_cache = (now, results)
    return results


@app.get("/api/models")
async def get_models_registry(
    authorization: str | None = Header(None),
) -> list[dict[str, Any]]:
    """Get models registry (cached, 30s TTL).

    Returns:
        [
            {
                "path": "exp09_adaptive/checkpoints/eqlm.pt",
                "size_mb": 45.2,
                "config": {
                    "d_model": 768,
                    "n_heads": 12,
                    "d_ff": 3072,
                    "vocab_size": 50257,
                    "map_form": "residual"
                },
                "model_class": "EqLM",
                "params_estimate": 12345678,
                "run": {
                    "config_sha": "abc123...",
                    "git_commit": "def456..."
                }
            },
            ...
        ]
    """
    require_bearer_auth(authorization)
    return scan_models_registry()


@app.post("/api/models/publish")
async def publish_models_endpoint(
    body: dict,
    authorization: str | None = Header(None),
) -> dict:
    """Publish a checkpoint to Hugging Face Hub.

    Request body:
        {
            "checkpoint_path": "exp09_adaptive/checkpoints/eqlm.pt",
            "repo_id": "kinetic-ai/eqlm-babylm-10m"
        }

    Returns:
        {
            "repo_url": "https://huggingface.co/kinetic-ai/eqlm-babylm-10m"
        }

    Raises:
        400: checkpoint not found, traversal attempt, invalid repo_id
        503: HF auth failed or upload failed
    """
    require_bearer_auth(authorization)

    checkpoint_path = body.get("checkpoint_path", "")
    repo_id = body.get("repo_id", "")

    if not checkpoint_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="checkpoint_path required",
        )

    if not repo_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="repo_id required",
        )

    # Validate repo_id format
    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="repo_id must be in format 'owner/name'",
        )

    # Security: prevent traversal attacks
    results_dir = Path(get_results_dir()).resolve()
    full_path = (results_dir / checkpoint_path).resolve()

    try:
        full_path.relative_to(results_dir)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal detected. checkpoint_path must be inside results/",
        ) from e

    if not full_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Checkpoint not found: {checkpoint_path}",
        )

    # Attempt publish
    try:
        repo_url = publish_checkpoint_to_hf(full_path, repo_id)
        return {"repo_url": repo_url}
    except RuntimeError as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "token" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Hugging Face authentication failed: {error_msg}",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to publish: {error_msg}",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8097,
        log_level="info",
    )
