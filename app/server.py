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
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from kinetic_ai.games.payoff import NormalFormGame
from kinetic_ai.games.qre import nash_conv, qre_path
from kinetic_ai.mechanisms.auctions import AuctionConfig, AuctionType, TokenAuction
from kinetic_ai.optim.bregman import NegativeEntropy
from kinetic_ai.optim.mmd import mmd_strategy_update
from kinetic_ai.serve.executor import JobInput, LocalExecutor

# ─── Config ──────────────────────────────────────────────────────────────────

__version__ = "0.1.0-phase3"

# Executor for job queue
executor = LocalExecutor()

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
        "ALLOWED_ORIGINS", "https://kinetic.sharath-sathish.workers.dev"
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


@app.post("/api/jobs")
async def submit_job(
    body: dict,
    authorization: str | None = Header(None),
) -> dict:
    """Submit a job to the Training Studio queue.

    Request body:
        {
            "type": "noop_demo" | "solve" | "train" | ...,
            "params": {...}
        }

    Returns:
        {"job_id": str}
    """
    require_bearer_auth(authorization)

    job_type = body.get("type", "noop_demo")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8097,
        log_level="info",
    )
