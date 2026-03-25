"""Training loop for FST classification with early stopping and metric tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from skin_type_classifier.evaluate import evaluate_model

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Metrics collected during a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate: float
    epoch_time_seconds: float


@dataclass
class TrainResult:
    """Complete result of a training run."""

    history: list[TrainMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_macro_f1: float = 0.0
    best_checkpoint_path: Path | None = None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Args:
        model: Model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device for tensors.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    max_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 7,
    early_stopping_min_delta: float = 1e-4,
    checkpoint_dir: Path | None = None,
) -> TrainResult:
    """Train model with early stopping on validation loss.

    Args:
        model: The model to train (already on ``device``).
        train_loader: Training DataLoader (with WeightedRandomSampler).
        val_loader: Validation DataLoader.
        class_weights: Optional tensor of shape ``(num_classes,)`` for CrossEntropyLoss.
            If None, uses unweighted loss (recommended when using WeightedRandomSampler).
        device: Device for tensors.
        max_epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for Adam optimizer.
        weight_decay: L2 regularization strength.
        scheduler_patience: Epochs before LR reduction.
        scheduler_factor: Factor to multiply LR on plateau.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_min_delta: Minimum improvement to count as progress.
        checkpoint_dir: If provided, save best model checkpoint here.

    Returns:
        TrainResult with full training history and best metrics.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=scheduler_patience,
        factor=scheduler_factor,
    )

    result = TrainResult()
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics.loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        metrics = TrainMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_metrics.loss,
            val_accuracy=val_metrics.accuracy,
            val_macro_f1=val_metrics.macro_f1,
            learning_rate=current_lr,
            epoch_time_seconds=elapsed,
        )
        result.history.append(metrics)

        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_f1={val_metrics.macro_f1:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics.loss < result.best_val_loss - early_stopping_min_delta:
            result.best_val_loss = val_metrics.loss
            result.best_val_macro_f1 = val_metrics.macro_f1
            result.best_epoch = epoch
            epochs_without_improvement = 0

            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                result.best_checkpoint_path = ckpt_path
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

    return result
