"""Unit tests for evaluation utilities."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skin_type_classifier.evaluate import EvalMetrics, evaluate_model


def _make_eval_loader(n_samples: int = 20, num_classes: int = 6) -> DataLoader:
    """Create a DataLoader with random predictions for testing."""
    images = torch.randn(n_samples, 3, 4, 4)
    labels = torch.arange(n_samples) % num_classes
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def _make_tiny_model(num_classes: int = 6) -> nn.Module:
    """Create a minimal model for testing evaluation."""
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))


@pytest.mark.unit
class TestEvaluateModel:
    """Test the evaluate_model function."""

    def test_returns_eval_metrics(self) -> None:
        """evaluate_model should return an EvalMetrics dataclass."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        result = evaluate_model(model, loader)
        assert isinstance(result, EvalMetrics)

    def test_per_class_f1_has_six_entries(self) -> None:
        """per_class_f1 should always have 6 entries (one per FST)."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        result = evaluate_model(model, loader)
        assert len(result.per_class_f1) == 6

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix should be (6, 6)."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        result = evaluate_model(model, loader)
        assert result.confusion_matrix.shape == (6, 6)

    def test_accuracy_in_valid_range(self) -> None:
        """Accuracy should be between 0 and 1."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        result = evaluate_model(model, loader)
        assert 0.0 <= result.accuracy <= 1.0

    def test_macro_f1_in_valid_range(self) -> None:
        """Macro F1 should be between 0 and 1."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        result = evaluate_model(model, loader)
        assert 0.0 <= result.macro_f1 <= 1.0

    def test_loss_computed_when_criterion_provided(self) -> None:
        """Loss should be non-zero when a criterion is provided."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        criterion = nn.CrossEntropyLoss()
        result = evaluate_model(model, loader, criterion=criterion)
        assert result.loss > 0.0

    def test_loss_zero_when_no_criterion(self) -> None:
        """Loss should be 0.0 when no criterion is provided."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        result = evaluate_model(model, loader, criterion=None)
        assert result.loss == 0.0

    def test_classification_report_is_string(self) -> None:
        """Classification report should be a non-empty string."""
        model = _make_tiny_model()
        loader = _make_eval_loader()
        result = evaluate_model(model, loader)
        assert isinstance(result.classification_report_str, str)
        assert len(result.classification_report_str) > 0
