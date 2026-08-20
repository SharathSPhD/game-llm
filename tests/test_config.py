"""Tests for configuration system and YAML serialization.

Tests that the config system properly handles type resolution, YAML
round-trips, and nested dataclass reconstruction without using eval().
"""

from pathlib import Path
from typing import get_type_hints

from kinetic_ai.config import (
    AuctionConfig,
    AuctionType,
    BregmanType,
    DEQConfig,
    ExperimentConfig,
    MMDConfig,
    SelfPlayConfig,
    SolverType,
    _dict_to_config,
)


class TestConfigTypeResolution:
    """Tests for type resolution in _dict_to_config."""

    def test_get_type_hints_resolves_all_fields(self) -> None:
        """Verify that get_type_hints can resolve all ExperimentConfig fields."""
        hints = get_type_hints(ExperimentConfig)
        expected_keys = {
            "name",
            "seed",
            "device",
            "mmd",
            "deq",
            "auction",
            "self_play",
            "output_dir",
            "log_interval",
        }
        assert set(hints.keys()) == expected_keys
        # Verify nested types are resolved
        assert hints["mmd"] == MMDConfig
        assert hints["deq"] == DEQConfig
        assert hints["auction"] == AuctionConfig
        assert hints["self_play"] == SelfPlayConfig

    def test_dict_to_config_nested_mmd(self) -> None:
        """Test that _dict_to_config correctly resolves nested MMDConfig."""
        data = {
            "name": "test",
            "seed": 42,
            "device": "cpu",
            "mmd": {
                "lr": 0.02,
                "tau": 0.15,
                "bregman_type": "negative_entropy",
                "reference_update_interval": 0,
            },
            "deq": {},
            "auction": {},
            "self_play": {},
            "output_dir": "outputs",
            "log_interval": 10,
        }
        config = _dict_to_config(data, ExperimentConfig)
        assert config.name == "test"
        assert config.seed == 42
        assert isinstance(config.mmd, MMDConfig)
        assert config.mmd.lr == 0.02
        assert config.mmd.tau == 0.15

    def test_dict_to_config_enum_resolution(self) -> None:
        """Test that enums are properly resolved from string values."""
        data = {
            "name": "test",
            "seed": 42,
            "device": "cpu",
            "mmd": {
                "lr": 0.01,
                "tau": 0.1,
                "bregman_type": "negative_entropy",
                "reference_update_interval": 0,
            },
            "deq": {
                "solver": "anderson",
                "max_iter": 50,
                "tol": 1e-5,
                "anderson_m": 5,
                "anderson_beta": 1.0,
                "spectral_norm": True,
                "jfb": False,
            },
            "auction": {
                "auction_type": "weighted_aggregation",
                "vocab_size": 32000,
                "aggregation_temp": 1.0,
                "reserve_price": 0.0,
            },
            "self_play": {},
            "output_dir": "outputs",
            "log_interval": 10,
        }
        config = _dict_to_config(data, ExperimentConfig)
        assert config.mmd.bregman_type == BregmanType.NEGATIVE_ENTROPY
        assert config.deq.solver == SolverType.ANDERSON
        assert config.auction.auction_type == AuctionType.WEIGHTED_AGGREGATION

    def test_dict_to_config_all_nested_configs(self) -> None:
        """Test that all nested dataclass configs load properly."""
        data = {
            "name": "full_test",
            "seed": 100,
            "device": "cuda",
            "mmd": {
                "lr": 0.01,
                "tau": 0.1,
                "bregman_type": "euclidean",
                "reference_update_interval": 5,
            },
            "deq": {
                "solver": "picard",
                "max_iter": 30,
                "tol": 1e-4,
                "anderson_m": 3,
                "anderson_beta": 0.8,
                "spectral_norm": False,
                "jfb": True,
            },
            "auction": {
                "auction_type": "second_price",
                "vocab_size": 50000,
                "aggregation_temp": 0.5,
                "reserve_price": 1.0,
            },
            "self_play": {
                "num_rounds": 20,
                "num_samples_per_round": 500,
                "eta": 2.0,
                "semantic_calibration": False,
                "repulsion_strength": 0.2,
            },
            "output_dir": "my_outputs",
            "log_interval": 5,
        }
        config = _dict_to_config(data, ExperimentConfig)

        # Verify all nested configs
        assert isinstance(config.mmd, MMDConfig)
        assert isinstance(config.deq, DEQConfig)
        assert isinstance(config.auction, AuctionConfig)
        assert isinstance(config.self_play, SelfPlayConfig)

        # Verify values propagated correctly
        assert config.mmd.bregman_type == BregmanType.EUCLIDEAN
        assert config.mmd.reference_update_interval == 5
        assert config.deq.solver == SolverType.PICARD
        assert config.deq.jfb is True
        assert config.auction.auction_type == AuctionType.SECOND_PRICE
        assert config.auction.vocab_size == 50000
        assert config.self_play.num_rounds == 20
        assert config.self_play.semantic_calibration is False
        assert config.output_dir == "my_outputs"
        assert config.log_interval == 5


class TestConfigYAMLRoundTrip:
    """Tests for YAML serialization and deserialization."""

    def test_yaml_roundtrip_simple_config(self) -> None:
        """Test save/load cycle preserves config."""
        config = ExperimentConfig(
            name="roundtrip_test",
            seed=123,
            device="cuda",
        )

        # Save to temp file
        temp_path = Path("/tmp/test_config.yaml")
        config.save(temp_path)

        # Load back
        loaded = ExperimentConfig.load(temp_path)

        # Verify round-trip
        assert loaded.name == config.name
        assert loaded.seed == config.seed
        assert loaded.device == config.device
        assert isinstance(loaded.mmd, MMDConfig)
        assert isinstance(loaded.deq, DEQConfig)

        temp_path.unlink()

    def test_yaml_roundtrip_full_config(self) -> None:
        """Test save/load cycle with customized nested configs."""
        config = ExperimentConfig(
            name="full_roundtrip",
            seed=999,
            device="cpu",
            mmd=MMDConfig(lr=0.05, tau=0.2, bregman_type=BregmanType.EUCLIDEAN),
            deq=DEQConfig(solver=SolverType.BROYDEN, max_iter=100),
            auction=AuctionConfig(
                auction_type=AuctionType.SECOND_PRICE, vocab_size=100000
            ),
            self_play=SelfPlayConfig(num_rounds=30, eta=1.5),
            output_dir="custom_outputs",
            log_interval=20,
        )

        temp_path = Path("/tmp/test_full_config.yaml")
        config.save(temp_path)
        loaded = ExperimentConfig.load(temp_path)

        # Verify all nested values
        assert loaded.mmd.lr == 0.05
        assert loaded.mmd.bregman_type == BregmanType.EUCLIDEAN
        assert loaded.deq.solver == SolverType.BROYDEN
        assert loaded.deq.max_iter == 100
        assert loaded.auction.auction_type == AuctionType.SECOND_PRICE
        assert loaded.auction.vocab_size == 100000
        assert loaded.self_play.num_rounds == 30
        assert loaded.self_play.eta == 1.5
        assert loaded.output_dir == "custom_outputs"
        assert loaded.log_interval == 20

        temp_path.unlink()

    def test_yaml_default_factory_fields_survive_roundtrip(self) -> None:
        """Test that default_factory fields are properly reconstructed."""
        config = ExperimentConfig(name="defaults_test")

        temp_path = Path("/tmp/test_defaults_config.yaml")
        config.save(temp_path)
        loaded = ExperimentConfig.load(temp_path)

        # Verify all defaults are present and correct
        assert loaded.mmd.lr == 1e-2
        assert loaded.mmd.tau == 0.1
        assert loaded.mmd.bregman_type == BregmanType.NEGATIVE_ENTROPY
        assert loaded.deq.solver == SolverType.ANDERSON
        assert loaded.deq.max_iter == 50
        assert loaded.auction.auction_type == AuctionType.WEIGHTED_AGGREGATION
        assert loaded.auction.vocab_size == 32000

        temp_path.unlink()

    def test_dict_to_config_with_none_returns_defaults(self) -> None:
        """Test that None data returns a config with all defaults."""
        config = _dict_to_config(None, ExperimentConfig)
        assert isinstance(config, ExperimentConfig)
        assert config.name == "default"
        assert config.seed == 42
        assert isinstance(config.mmd, MMDConfig)
