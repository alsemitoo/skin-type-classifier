"""Unit tests for learning curve orchestration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from skin_type_classifier.learning_curve import run_single_trial


def _make_synthetic_df(n_groups: int = 60, images_per_group: int = 2) -> pd.DataFrame:
    """Create a synthetic dataset for learning curve testing."""
    rng = np.random.default_rng(42)
    rows = []
    for g in range(n_groups):
        fst = (g % 6) + 1
        for i in range(images_per_group):
            rows.append(
                {
                    "image_path": f"img_g{g}_{i}.png",
                    "source": "scin",
                    "fitzpatrick_skin_type": fst,
                    "diagnosis": "test",
                    "age": str(rng.integers(20, 80)),
                    "sex": "FEMALE",
                    "group_id": f"G{g}",
                    "revised_fitzpatrick": fst,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestRunSingleTrial:
    """Test the run_single_trial function with mocked model and data loading."""

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        """Result dict should contain all expected metric keys."""
        df = _make_synthetic_df(n_groups=60)
        train_df = df[df["group_id"].isin([f"G{i}" for i in range(42)])].copy()
        val_df = df[df["group_id"].isin([f"G{i}" for i in range(42, 51)])].copy()
        test_df = df[df["group_id"].isin([f"G{i}" for i in range(51, 60)])].copy()

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        from PIL import Image

        for _, row in df.iterrows():
            Image.new("RGB", (32, 32), color="red").save(tmp_path / row["image_path"])

        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "data": {"data_root": str(tmp_path)},
                "model": {"num_classes": 6, "dropout": 0.1, "freeze_backbone": True},
                "training": {
                    "batch_size": 8,
                    "num_workers": 0,
                    "max_epochs": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "scheduler_patience": 1,
                    "scheduler_factor": 0.5,
                    "early_stopping_patience": 10,
                    "early_stopping_min_delta": 1e-4,
                },
            }
        )

        result = run_single_trial(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cfg=cfg,
            fraction=0.5,
            seed=42,
            device=torch.device("cpu"),
            output_dir=tmp_path / "output",
        )

        expected_keys = {
            "fraction",
            "seed",
            "n_train_images",
            "n_train_groups",
            "best_epoch",
            "val_loss",
            "val_macro_f1",
            "val_accuracy",
            "test_accuracy",
            "test_macro_f1",
        }
        expected_keys.update(f"test_per_class_f1_{i}" for i in range(6))
        assert expected_keys.issubset(result.keys())

    def test_fraction_reduces_training_size(self, tmp_path: Path) -> None:
        """Using fraction < 1.0 should reduce the number of training images."""
        df = _make_synthetic_df(n_groups=60)
        train_df = df[df["group_id"].isin([f"G{i}" for i in range(42)])].copy()
        val_df = df[df["group_id"].isin([f"G{i}" for i in range(42, 51)])].copy()
        test_df = df[df["group_id"].isin([f"G{i}" for i in range(51, 60)])].copy()

        from PIL import Image

        for _, row in df.iterrows():
            Image.new("RGB", (32, 32), color="red").save(tmp_path / row["image_path"])

        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "data": {"data_root": str(tmp_path)},
                "model": {"num_classes": 6, "dropout": 0.1, "freeze_backbone": True},
                "training": {
                    "batch_size": 8,
                    "num_workers": 0,
                    "max_epochs": 1,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "scheduler_patience": 1,
                    "scheduler_factor": 0.5,
                    "early_stopping_patience": 10,
                    "early_stopping_min_delta": 1e-4,
                },
            }
        )

        result_half = run_single_trial(train_df, val_df, test_df, cfg, 0.5, 42, torch.device("cpu"), tmp_path / "out1")
        result_full = run_single_trial(train_df, val_df, test_df, cfg, 1.0, 42, torch.device("cpu"), tmp_path / "out2")

        assert result_half["n_train_images"] < result_full["n_train_images"]
