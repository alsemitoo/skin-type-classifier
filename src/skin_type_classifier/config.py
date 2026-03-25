"""Structured configuration for FST classification experiments.

Uses Hydra's structured configs (via dataclasses) for type-safe YAML parsing
with CLI override support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    """Data paths and split configuration."""

    csv_path: str = "data/processed/full_dataset.csv"
    data_root: str = "data/processed"
    test_size: float = 0.15
    val_size: float = 0.15
    split_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    backbone: str = "efficientnet_v2_m"
    num_classes: int = 6
    dropout: float = 0.3
    freeze_backbone: bool = True


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4


@dataclass
class LearningCurveConfig:
    """Learning curve experiment parameters."""

    fractions: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    output_dir: str = "reports/learning_curve"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    learning_curve: LearningCurveConfig = field(default_factory=LearningCurveConfig)


cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)
