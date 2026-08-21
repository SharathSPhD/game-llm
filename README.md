# Kinetic AI

**A unified library for game-theoretic LLM training: Magnetic Mirror Descent, Deep Equilibrium Models, and Mechanism Design.**

[![Tests](https://github.com/SharathSPhD/game-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/SharathSPhD/game-llm/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Thesis

Current AI systems are trained via **dictatorial optimization** — a single loss function forces updates on every parameter. But real-world deployment environments are **adversarial, multi-agent, and strategic**. Game theory, not optimization, is the correct mathematical framework.

Kinetic AI implements the transition from **static optimization** to **dynamic equilibrium**:

| Component | What it replaces | Validated status (see `research/memory/findings.md`) |
|-----------|-----------------|------------------------------------------------------|
| **Magnetic Mirror Descent** | Simultaneous gradient play | Linear last-iterate convergence to its magnetic fixed point where GDA cycles (F1); RND resets reach Nash (F3); asymmetric-game attractor gap discovered (F2) |
| **Deep Equilibrium Models** | Explicit transformer layers | O(1) activation memory vs O(N) measured (F4); Anderson wins on stiff fixed points (F5) |
| **Token Auctions** | Winner-take-all generation | Second-price empirically truthful, regret exactly 0 (F6); weighted aggregation measurably manipulable |
| **Self-Play (SPPO)** | RLHF / DPO | Policy-weighted self-play loop with convergence tests; LLM-scale runs pending |
| **EqLM** (new architecture) | Stacked GPT-class LMs | Parity with a param-matched explicit baseline at smoke scale (F10); full BabyLM run in progress |

Everything above traces to committed runs under `results/` (config hashes + seeds).
The research process is spec-driven and adversarially reviewed — see `CLAUDE.md`,
`research/specs/`, and `docs/decisions/`. Paper: `paper/kinetic_ai.tex`. Site: `site/`.
Researcher app: `apps/web` + `app/server.py` (see `apps/web/DEPLOY.md`).

## Installation

```bash
pip install -e ".[all]"
```

## Quick Start

### Strategy-Space MMD on Rock-Paper-Scissors

```python
import torch
from kinetic_ai.games.payoff import rock_paper_scissors
from kinetic_ai.games.qre import nash_conv
from kinetic_ai.optim.bregman import NegativeEntropy
from kinetic_ai.optim.mmd import mmd_strategy_update

game = rock_paper_scissors()
bregman = NegativeEntropy()

s1 = torch.tensor([0.7, 0.2, 0.1])  # Biased initial strategy
s2 = torch.tensor([0.1, 0.7, 0.2])
ref = torch.ones(3) / 3  # Uniform reference (magnet)

for step in range(500):
    # Sequential (alternating) updates with reduced learning rate
    # ensure convergence. Simultaneous updates require tighter stepsizes.
    g1 = game.utility_gradient(1, s1, s2)
    s1 = mmd_strategy_update(s1, g1, ref, bregman, lr=0.1, tau=0.05)
    
    g2 = game.utility_gradient(2, s2, s1)
    s2 = mmd_strategy_update(s2, g2, ref, bregman, lr=0.1, tau=0.05)

print(f"NashConv: {nash_conv(game, s1, s2):.6f}")  # Converges to τ-regularized QRE (≈Nash for RPS)
```

### DEQ Layer with Anderson Acceleration

```python
import torch
import torch.nn as nn
from kinetic_ai.config import DEQConfig, SolverType
from kinetic_ai.models.deq_layer import DEQLayer

transform = nn.Linear(32, 16)
def f(z, x):
    return torch.tanh(transform(torch.cat([z, x], dim=-1)))

deq = DEQLayer(f, DEQConfig(solver=SolverType.ANDERSON, max_iter=50))
z_star = deq(torch.randn(1, 16))  # Finds equilibrium state
```

### Token Auction

```python
import torch
from kinetic_ai.config import AuctionConfig, AuctionType
from kinetic_ai.mechanisms.auctions import TokenAuction

auction = TokenAuction(AuctionConfig(
    auction_type=AuctionType.WEIGHTED_AGGREGATION,
    vocab_size=1000,
))

bids = torch.tensor([2.0, 5.0, 1.0])
dists = torch.softmax(torch.randn(3, 1000), dim=-1)
result = auction.run_auction(bids, dists)
print(f"Selected token: {result.sampled_token}")
```

## Architecture

```
kinetic_ai/
├── optim/          # Magnetic Mirror Descent + Bregman divergences
├── models/         # Deep Equilibrium Layers (Anderson, Broyden, Picard)
├── mechanisms/     # Token auctions, mechanism design
├── games/          # Game definitions, QRE computation, self-play
├── eval/           # Convergence diagnostics, statistical testing
└── config.py       # Config-driven experiment system
```

## Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/ -v -m "not slow"      # Skip slow convergence tests
```

## Running the Full Simulation

```bash
python simulate.py
```

## References

1. Sokota et al. "A Unified Approach to RL, QRE, and Two-Player Zero-Sum Games" (NeurIPS 2023)
2. Bai et al. "Deep Equilibrium Models" (NeurIPS 2019)
3. Duetting et al. "Mechanism Design for Large Language Models" (WWW 2024, Best Paper)
4. Wu et al. "Self-Play Preference Optimization for Language Model Alignment" (2024)
5. McKelvey & Palfrey "Quantal Response Equilibria for Normal Form Games" (1995)

## License

MIT
