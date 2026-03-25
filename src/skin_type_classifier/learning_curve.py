"""Learning curve experiment: train at various data fractions to estimate data requirements."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig

from skin_type_classifier.data.data import FSTDataset, make_eval_loader, make_train_loader
from skin_type_classifier.data.splits import stratified_group_split, subsample_training_groups
from skin_type_classifier.data.transforms import get_eval_transform, get_train_transform
from skin_type_classifier.evaluate import evaluate_model
from skin_type_classifier.model import FSTClassifier
from skin_type_classifier.train import train_model
from skin_type_classifier.visualize import plot_learning_curve, plot_per_class_f1_heatmap

logger = logging.getLogger(__name__)


def _detect_device() -> torch.device:
    if torch.xpu.is_available():
        device = torch.device("xpu")
        logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def run_single_trial(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: DictConfig,
    fraction: float,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Train and evaluate a single (fraction, seed) trial.

    Args:
        train_df: Full training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        cfg: Experiment configuration.
        fraction: Fraction of training data to use.
        seed: Random seed for subsampling and model init.
        device: Device for training.
        output_dir: Root output directory for checkpoints.

    Returns:
        Dict of results for this trial.
    """
    if fraction < 1.0:
        sub_train_df = subsample_training_groups(train_df, fraction=fraction, random_state=seed)
    else:
        sub_train_df = train_df.copy()

    n_train = len(sub_train_df)
    logger.info(f"Trial fraction={fraction}, seed={seed}: {n_train} training images")

    data_root = Path(cfg.data.data_root)
    train_dataset = FSTDataset(sub_train_df, data_root, transform=get_train_transform())
    val_dataset = FSTDataset(val_df, data_root, transform=get_eval_transform())
    test_dataset = FSTDataset(test_df, data_root, transform=get_eval_transform())

    train_loader = make_train_loader(
        train_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers
    )
    val_loader = make_eval_loader(
        val_dataset, batch_size=cfg.training.batch_size * 2, num_workers=cfg.training.num_workers
    )
    test_loader = make_eval_loader(
        test_dataset, batch_size=cfg.training.batch_size * 2, num_workers=cfg.training.num_workers
    )

    torch.manual_seed(seed)
    model = FSTClassifier(
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)

    trial_dir = output_dir / f"frac_{fraction:.2f}" / f"seed_{seed}"

    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=cfg.training.max_epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        scheduler_patience=cfg.training.scheduler_patience,
        scheduler_factor=cfg.training.scheduler_factor,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        checkpoint_dir=trial_dir,
    )

    if train_result.best_checkpoint_path is not None:
        model.load_state_dict(torch.load(train_result.best_checkpoint_path, weights_only=True))
        model.to(device)

    test_metrics = evaluate_model(model, test_loader, device=device)

    return {
        "fraction": fraction,
        "seed": seed,
        "n_train_images": n_train,
        "n_train_groups": sub_train_df["group_id"].nunique(),
        "best_epoch": train_result.best_epoch,
        "val_loss": train_result.best_val_loss,
        "val_macro_f1": train_result.best_val_macro_f1,
        "val_accuracy": train_result.history[train_result.best_epoch - 1].val_accuracy
        if train_result.best_epoch > 0
        else 0.0,
        "test_accuracy": test_metrics.accuracy,
        "test_macro_f1": test_metrics.macro_f1,
        **{f"test_per_class_f1_{i}": f1 for i, f1 in enumerate(test_metrics.per_class_f1)},
    }


def run_learning_curve(cfg: DictConfig) -> pd.DataFrame:
    """Run the full learning curve experiment.

    Args:
        cfg: Experiment configuration (Hydra DictConfig).

    Returns:
        DataFrame with one row per (fraction, seed) trial.
    """
    output_dir = Path(cfg.learning_curve.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _detect_device()

    df = pd.read_csv(cfg.data.csv_path)
    train_df, val_df, test_df = stratified_group_split(
        df, test_size=cfg.data.test_size, val_size=cfg.data.val_size, random_state=cfg.data.split_seed
    )
    logger.info(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    fractions = list(cfg.learning_curve.fractions)
    seeds = list(cfg.learning_curve.seeds)
    total_trials = len(fractions) * len(seeds)
    results: list[dict] = []

    for frac_idx, fraction in enumerate(fractions):
        for seed_idx, seed in enumerate(seeds):
            current = frac_idx * len(seeds) + seed_idx + 1
            logger.info(f"--- Trial {current}/{total_trials}: fraction={fraction}, seed={seed} ---")

            result = run_single_trial(train_df, val_df, test_df, cfg, fraction, seed, device, output_dir)
            results.append(result)

            results_df = pd.DataFrame(results)
            results_df.to_csv(output_dir / "results.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_learning_curve(results_df, output_dir, metric="test_macro_f1")
    plot_learning_curve(results_df, output_dir, metric="test_accuracy")
    plot_per_class_f1_heatmap(results_df, output_dir)

    summary = results_df.groupby("fraction")[["test_macro_f1", "test_accuracy"]].agg(["mean", "std"])
    logger.info(f"\nLearning Curve Summary:\n{summary}")

    return results_df
