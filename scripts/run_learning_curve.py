"""CLI entry point for the learning curve experiment."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from skin_type_classifier.config import ExperimentConfig  # noqa: F401 (registers structured config)
from skin_type_classifier.learning_curve import run_learning_curve

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="learning_curve")
def main(cfg: DictConfig) -> None:
    """Run the FST learning curve experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    results_df = run_learning_curve(cfg)
    logger.info(f"Total trials: {len(results_df)}")
    logger.info(f"Best test macro F1: {results_df['test_macro_f1'].max():.4f}")


if __name__ == "__main__":
    main()
