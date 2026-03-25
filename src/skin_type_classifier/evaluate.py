"""Evaluation utilities for FST classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

FST_LABELS = ["FST 1", "FST 2", "FST 3", "FST 4", "FST 5", "FST 6"]
ALL_CLASSES = list(range(6))


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single model evaluation."""

    loss: float
    accuracy: float
    macro_f1: float
    per_class_f1: list[float]
    confusion_matrix: np.ndarray
    classification_report_str: str


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None = None,
    device: torch.device | None = None,
) -> EvalMetrics:
    """Evaluate model on a DataLoader and return comprehensive metrics.

    Args:
        model: Trained model (set to eval mode internally).
        loader: Evaluation DataLoader (val or test).
        criterion: Loss function. If None, loss is reported as 0.0.
        device: Device to run inference on. Defaults to CPU.

    Returns:
        EvalMetrics with loss, accuracy, macro F1, per-class F1, confusion matrix.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion is not None:
            total_loss += criterion(outputs, labels).item()
            n_batches += 1
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=ALL_CLASSES, zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None, labels=ALL_CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CLASSES)
    report = classification_report(y_true, y_pred, labels=ALL_CLASSES, target_names=FST_LABELS, zero_division=0)

    return EvalMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class.tolist(),
        confusion_matrix=cm,
        classification_report_str=report,
    )
