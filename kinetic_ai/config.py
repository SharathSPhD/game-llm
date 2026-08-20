"""Configuration system for Kinetic AI.

All hyperparameters and experiment settings are defined as dataclasses,
enabling type safety, serialization to/from YAML, and config-driven experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast, get_type_hints

import yaml


class BregmanType(str, Enum):
    """Supported Bregman divergence types."""

    NEGATIVE_ENTROPY = "negative_entropy"  # KL divergence on simplex
    EUCLIDEAN = "euclidean"  # L2 squared / 2
    DILATED_ENTROPY = "dilated_entropy"  # For extensive-form games (treeplex)


class SolverType(str, Enum):
    """Fixed-point solver types for DEQ."""

    PICARD = "picard"  # Simple fixed-point iteration (baseline)
    ANDERSON = "anderson"  # Anderson acceleration (default)
    BROYDEN = "broyden"  # Broyden's method


class AuctionType(str, Enum):
    """Auction mechanism types."""

    SECOND_PRICE = "second_price"  # Vickrey auction
    WEIGHTED_AGGREGATION = "weighted_aggregation"  # Distribution mixing


@dataclass
class MMDConfig:
    """Configuration for Magnetic Mirror Descent optimizer.

    Attributes:
        lr: Learning rate (step size η).
        tau: Magnetic strength parameter. Controls pull toward reference policy.
            High τ → conservative (stays near reference).
            Low τ → aggressive (explores freely).
        bregman_type: Which Bregman divergence to use as the mirror map.
        reference_update_interval: Steps between reference policy updates.
            0 = never update (fixed reference). >0 = Regularized Nash Dynamics.
    """

    lr: float = 1e-2
    tau: float = 0.1
    bregman_type: BregmanType = BregmanType.NEGATIVE_ENTROPY
    reference_update_interval: int = 0


@dataclass
class DEQConfig:
    """Configuration for Deep Equilibrium Layer.

    Attributes:
        solver: Which fixed-point solver to use.
        max_iter: Maximum iterations for the forward pass solver.
        tol: Convergence tolerance (L2 norm of residual).
        anderson_m: History size for Anderson acceleration.
        anderson_beta: Mixing coefficient for Anderson acceleration.
        spectral_norm: Whether to apply spectral normalization for contraction.
        jfb: Jacobian-Free Backprop — use fixed-point iteration for backward
            pass instead of full implicit differentiation. Saves memory.
    """

    solver: SolverType = SolverType.ANDERSON
    max_iter: int = 50
    tol: float = 1e-5
    anderson_m: int = 5
    anderson_beta: float = 1.0
    spectral_norm: bool = True
    jfb: bool = False


@dataclass
class AuctionConfig:
    """Configuration for Token Auction mechanism.

    Attributes:
        auction_type: Which auction mechanism to use.
        vocab_size: Size of the token vocabulary.
        aggregation_temp: Temperature for softmax aggregation of distributions.
        reserve_price: Minimum bid to participate.
    """

    auction_type: AuctionType = AuctionType.WEIGHTED_AGGREGATION
    vocab_size: int = 32000
    aggregation_temp: float = 1.0
    reserve_price: float = 0.0


@dataclass
class SelfPlayConfig:
    """Configuration for SPPO-style self-play training.

    Attributes:
        num_rounds: Number of self-play rounds.
        num_samples_per_round: Synthetic samples generated per round.
        eta: Multiplicative weights step size.
        semantic_calibration: Whether to use S-SPPO semantic gating.
        repulsion_strength: Strength of latent-space repulsive force (S-SPPO).
    """

    num_rounds: int = 10
    num_samples_per_round: int = 1000
    eta: float = 1.0
    semantic_calibration: bool = True
    repulsion_strength: float = 0.1


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Attributes:
        name: Human-readable experiment name.
        seed: Random seed for reproducibility.
        device: Torch device string.
        mmd: MMD optimizer configuration.
        deq: DEQ layer configuration.
        auction: Auction mechanism configuration.
        self_play: Self-play training configuration.
        output_dir: Directory for experiment outputs.
        log_interval: Steps between logging.
    """

    name: str = "default"
    seed: int = 42
    device: str = "cpu"
    mmd: MMDConfig = field(default_factory=MMDConfig)
    deq: DEQConfig = field(default_factory=DEQConfig)
    auction: AuctionConfig = field(default_factory=AuctionConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    output_dir: str = "outputs"
    log_interval: int = 10

    def save(self, path: str | Path) -> None:
        """Serialize config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(_dataclass_to_dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        """Deserialize config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cast(ExperimentConfig, _dict_to_config(data, cls))


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to a dict, handling enums."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for k, _v in obj.__dataclass_fields__.items():
            result[k] = _dataclass_to_dict(getattr(obj, k))
        return result
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


def _dict_to_config(data: dict[str, Any], cls: type) -> Any:
    """Recursively reconstruct a dataclass from a dict.

    Uses get_type_hints() for safe type annotation resolution, which handles
    forward references and string annotations without eval().
    """
    if data is None:
        return cls()

    # get_type_hints() safely resolves string annotations from __future__ import
    field_types = get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        ft = field_types[key]
        # Type annotations are already resolved by get_type_hints()
        if hasattr(ft, "__dataclass_fields__"):
            kwargs[key] = _dict_to_config(value, ft)
        elif isinstance(ft, type) and issubclass(ft, Enum):
            kwargs[key] = ft(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
