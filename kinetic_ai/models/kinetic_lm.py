"""KineticLM (SPEC 0011): equilibrium conversion of a pretrained causal LM.

Converts a standard HuggingFace decoder stack into the EqLMCore topology
validated at 121M (F24/B3): explicit outer layers around a **weight-tied core
applied recursively**. The conversion is deliberately implemented as an
in-place restructuring of ``model.model.layers`` rather than a new
architecture class, because that keeps every downstream HF behaviour intact —
KV cache, rotary embeddings, attention masks, ``generate()``, and third-party
harnesses (lm-evaluation-harness) all work unmodified.

Two details make that safe:

1. **Tying by parameter identity, not module identity.** The core positions
   hold distinct module objects whose parameters are the *same*
   ``nn.Parameter`` objects. Distinct modules give each recursion its own
   ``layer_idx`` (so the KV cache stays correct); shared parameters give the
   memory saving and accumulate gradients from every recursion.
2. **Depth is a property of the layer list**, so the inference budget dial
   (F19/F24) is just rebuilding the list at a different length.

Initialization follows published recursive-uptraining practice (Relaxed
Recursive Transformers, arXiv 2410.20672): ``average`` (mean of the replaced
layers) or ``stepwise`` (adopt one representative layer).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

KINETIC_CONFIG_FILE = "kinetic_config.json"


@dataclass
class KineticConfig:
    """Surgery plan for converting a pretrained stack.

    Attributes:
        n_pre: Explicit layers kept at the bottom (initialized from the base).
        n_post: Explicit layers kept at the top.
        n_cores: Number of DISTINCT shared blocks the middle is partitioned
            into (block-recursive sharing). 1 = one block looped over the whole
            middle (maximum saving, maximum damage); M = no sharing at all.
            Intermediate values trade parameter saving against how much
            function is destroyed by collapsing distinct layers.
        recursion_depth: Applications of EACH core. Defaults to the group size
            it replaces (depth-preserving conversion).
        init_strategy: 'average' (mean of the layers a core replaces) or
            'stepwise' (adopt the middle layer of the group).
    """

    n_pre: int = 6
    n_post: int = 6
    n_cores: int = 1
    recursion_depth: int | None = None
    init_strategy: str = "average"

    def __post_init__(self) -> None:
        if self.init_strategy not in ("average", "stepwise"):
            raise ValueError(
                f"init_strategy must be 'average' or 'stepwise', got {self.init_strategy}"
            )
        if self.n_pre < 0 or self.n_post < 0:
            raise ValueError("n_pre and n_post must be non-negative")
        if self.n_cores < 1:
            raise ValueError(f"n_cores must be >= 1, got {self.n_cores}")


def count_unique_params(model: nn.Module) -> int:
    """Parameter count that respects sharing (each tensor counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def _named_parameter_paths(module: nn.Module) -> list[str]:
    return [name for name, _ in module.named_parameters()]


def _set_by_path(module: nn.Module, path: str, value: nn.Parameter) -> None:
    parts = path.split(".")
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _get_by_path(module: nn.Module, path: str) -> nn.Parameter:
    target: Any = module
    for part in path.split("."):
        target = getattr(target, part)
    return target


def _tie_to(source: nn.Module, target: nn.Module) -> None:
    """Point every parameter of ``target`` at ``source``'s parameter object."""
    for path in _named_parameter_paths(source):
        _set_by_path(target, path, _get_by_path(source, path))


def _set_layer_idx(layer: nn.Module, idx: int) -> None:
    """Give a decoder layer its position, so KV cache slots stay distinct."""
    if hasattr(layer, "layer_idx"):
        object.__setattr__(layer, "layer_idx", idx)
    for child in layer.modules():
        if hasattr(child, "layer_idx"):
            object.__setattr__(child, "layer_idx", idx)


def _sync_config_for_layers(
    model: Any, n_pre: int, depth: int, n_post: int
) -> None:
    """Keep per-layer config lists in step with the rebuilt stack.

    Modern HF configs carry per-layer metadata (e.g. Qwen3's ``layer_types``,
    which the forward indexes by position). Changing the recursion depth
    changes the number of layers, so these lists must be rebuilt or the
    forward indexes past their end.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    total = n_pre + depth + n_post
    if hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = total
    for attr in ("layer_types",):
        types = getattr(cfg, attr, None)
        if isinstance(types, (list, tuple)) and types:
            pre = list(types[:n_pre])
            core_type = types[min(n_pre, len(types) - 1)]
            post = list(types[len(types) - n_post :]) if n_post else []
            setattr(cfg, attr, pre + [core_type] * depth + post)


def _build_core_layer(layers: list[nn.Module], strategy: str) -> nn.Module:
    """Create the single shared block that replaces ``layers``."""
    if strategy == "stepwise":
        return copy.deepcopy(layers[len(layers) // 2])

    core = copy.deepcopy(layers[0])
    with torch.no_grad():
        for path in _named_parameter_paths(core):
            stack = torch.stack([_get_by_path(m, path).detach() for m in layers])
            _get_by_path(core, path).copy_(stack.mean(dim=0))
        # Buffers (e.g. norm running state, if any) take the first layer's values.
    return core


class _KineticMixin:
    """Behaviour mixed into the converted HF model instance."""

    kinetic_config: KineticConfig
    _core_layer: nn.Module

    _core_layers: list[nn.Module]

    @property
    def recursion_depth(self) -> int:
        """Applications of each core (the budget dial)."""
        cfg = self.kinetic_config
        middle = len(self.model.layers) - cfg.n_pre - cfg.n_post  # type: ignore[attr-defined]
        return middle // len(self._core_layers)

    def set_recursion_depth(self, depth: int) -> None:
        """Inference/training budget dial: apply the shared core ``depth`` times."""
        if depth < 1:
            raise ValueError(f"recursion_depth must be >= 1, got {depth}")
        cfg = self.kinetic_config
        layers = self.model.layers  # type: ignore[attr-defined]
        pre = list(layers[: cfg.n_pre])
        post = list(layers[len(layers) - cfg.n_post :]) if cfg.n_post else []
        core: list[nn.Module] = []
        for core_layer in self._core_layers:
            core.append(core_layer)
            core.extend(_clone_tied(core_layer) for _ in range(depth - 1))
        new_layers = nn.ModuleList(pre + core + post)
        for i, layer in enumerate(new_layers):
            _set_layer_idx(layer, i)
        self.model.layers = new_layers  # type: ignore[attr-defined]
        _sync_config_for_layers(
            self, cfg.n_pre, depth * len(self._core_layers), cfg.n_post
        )

    def forward_at_depths(
        self, input_ids: torch.Tensor, depths: list[int], **kwargs: Any
    ) -> dict[int, torch.Tensor]:
        """Anytime supervision (F24/B1): logits at several recursion depths.

        Restores the original depth before returning. Each depth is a separate
        forward, which is the honest cost of supervising truncated computation.
        """
        if not depths:
            raise ValueError("depths must be non-empty")
        original = self.recursion_depth
        out: dict[int, torch.Tensor] = {}
        try:
            for d in sorted(set(int(x) for x in depths)):
                self.set_recursion_depth(d)
                out[d] = self(input_ids, **kwargs).logits  # type: ignore[operator]
        finally:
            self.set_recursion_depth(original)
        return out

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:  # type: ignore[override]
        """Save as a STANDARD dense checkpoint plus the surgery plan.

        The core's shared parameters are materialized once per recursion so the
        artifact loads in any HF-compatible tool (including
        lm-evaluation-harness) with no custom code; ``load_kinetic`` re-ties
        them afterwards to restore the memory saving. The private
        ``_core_layer`` alias is dropped — it duplicates layer entries.
        """
        save_directory = Path(save_directory)
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = {
                k: v.detach().clone()
                for k, v in self.state_dict().items()  # type: ignore[attr-defined]
                if not (k.startswith("_core_layer.") or k.startswith("_core_layers."))
            }
        super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)  # type: ignore[misc]
        (save_directory / KINETIC_CONFIG_FILE).write_text(
            json.dumps(asdict(self.kinetic_config), indent=2)
        )


def _clone_tied(core: nn.Module) -> nn.Module:
    """A module that shares ``core``'s parameters but is a distinct object."""
    clone = copy.deepcopy(core)
    _tie_to(core, clone)
    return clone


def convert_to_kinetic(model: nn.Module, config: KineticConfig) -> Any:
    """Convert a pretrained HF causal LM into the KineticLM topology, in place.

    Args:
        model: A causal LM whose decoder stack lives at ``model.model.layers``.
        config: Surgery plan.

    Returns:
        The same model object, restructured and augmented with the KineticLM
        API (``recursion_depth``, ``set_recursion_depth``, ``forward_at_depths``).
    """
    layers = list(model.model.layers)  # type: ignore[attr-defined]
    n_layers = len(layers)
    n_middle = n_layers - config.n_pre - config.n_post
    if n_middle < 2:
        raise ValueError(
            f"need >= 2 middle layers to tie; got {n_middle} "
            f"({n_layers} layers, n_pre={config.n_pre}, n_post={config.n_post})"
        )

    if config.n_cores > n_middle:
        raise ValueError(f"n_cores ({config.n_cores}) exceeds middle layers ({n_middle})")
    middle = layers[config.n_pre : config.n_pre + n_middle]
    group = n_middle // config.n_cores
    groups = [
        middle[i * group : (i + 1) * group if i < config.n_cores - 1 else n_middle]
        for i in range(config.n_cores)
    ]
    cores = [_build_core_layer(g, config.init_strategy) for g in groups]
    depth = config.recursion_depth or group

    cls = type(model)
    if not isinstance(model, _KineticMixin):
        model.__class__ = type(f"Kinetic{cls.__name__}", (_KineticMixin, cls), {})
    model.kinetic_config = config  # type: ignore[attr-defined]
    model._core_layers = cores  # type: ignore[attr-defined]
    model._core_layer = cores[0]  # type: ignore[attr-defined]  # back-compat alias

    pre = layers[: config.n_pre]
    post = layers[n_layers - config.n_post :] if config.n_post else []
    core_stack: list[nn.Module] = []
    for c in cores:
        core_stack.append(c)
        core_stack.extend(_clone_tied(c) for _ in range(depth - 1))
    new_layers = nn.ModuleList(pre + core_stack + post)
    for i, layer in enumerate(new_layers):
        _set_layer_idx(layer, i)
    model.model.layers = new_layers  # type: ignore[attr-defined]
    _sync_config_for_layers(model, config.n_pre, depth * len(cores), config.n_post)
    return model


def load_kinetic(path: str | Path, **kwargs: Any) -> Any:
    """Load a saved KineticLM, restoring the tying structure."""
    from transformers import AutoModelForCausalLM

    path = Path(path)
    cfg_file = path / KINETIC_CONFIG_FILE
    if not cfg_file.exists():
        raise FileNotFoundError(f"{cfg_file} missing — not a KineticLM checkpoint")
    kin_cfg = KineticConfig(**json.loads(cfg_file.read_text()))

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    layers = list(model.model.layers)
    n_layers = len(layers)
    depth = n_layers - kin_cfg.n_pre - kin_cfg.n_post

    # The saved stack already holds identical (duplicated) core weights; re-tie
    # them to one parameter set so the memory saving survives the round trip.
    per_core = depth // kin_cfg.n_cores
    cores = [layers[kin_cfg.n_pre + i * per_core] for i in range(kin_cfg.n_cores)]
    cls = type(model)
    model.__class__ = type(f"Kinetic{cls.__name__}", (_KineticMixin, cls), {})
    model.kinetic_config = kin_cfg
    model._core_layers = cores
    model._core_layer = cores[0]

    core_stack: list[nn.Module] = []
    for c in cores:
        core_stack.append(c)
        core_stack.extend(_clone_tied(c) for _ in range(per_core - 1))
    new_layers = nn.ModuleList(
        layers[: kin_cfg.n_pre] + core_stack + (layers[n_layers - kin_cfg.n_post :] if kin_cfg.n_post else [])
    )
    for i, layer in enumerate(new_layers):
        _set_layer_idx(layer, i)
    model.model.layers = new_layers
    _sync_config_for_layers(model, kin_cfg.n_pre, per_core * len(cores), kin_cfg.n_post)
    return model
