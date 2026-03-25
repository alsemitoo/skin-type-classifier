"""Unit tests for the training loop."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skin_type_classifier.train import TrainResult, train_model, train_one_epoch


def _make_loader(n_samples: int = 16, num_classes: int = 2) -> DataLoader:
    """Create a simple DataLoader with linearly separable data."""
    torch.manual_seed(0)
    images = torch.randn(n_samples, 3, 4, 4)
    labels = torch.arange(n_samples) % num_classes
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def _make_simple_model(num_classes: int = 2) -> nn.Module:
    """Create a minimal trainable model."""
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))


@pytest.mark.unit
class TestTrainOneEpoch:
    """Test the train_one_epoch function."""

    def test_returns_float_loss(self) -> None:
        """train_one_epoch should return a float loss value."""
        model = _make_simple_model()
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        device = torch.device("cpu")
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_model_in_train_mode_after(self) -> None:
        """Model should be in train mode after train_one_epoch."""
        model = _make_simple_model()
        model.eval()
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        device = torch.device("cpu")
        train_one_epoch(model, loader, criterion, optimizer, device)
        assert model.training


@pytest.mark.unit
class TestTrainModel:
    """Test the full train_model function."""

    def test_returns_train_result(self) -> None:
        """train_model should return a TrainResult with non-empty history."""
        model = _make_simple_model()
        train_loader = _make_loader()
        val_loader = _make_loader()
        device = torch.device("cpu")

        result = train_model(model, train_loader, val_loader, device=device, max_epochs=3, early_stopping_patience=10)
        assert isinstance(result, TrainResult)
        assert len(result.history) == 3

    def test_early_stopping_triggers(self) -> None:
        """Training should stop before max_epochs if validation loss stalls."""
        model = _make_simple_model()
        train_loader = _make_loader()
        val_loader = _make_loader()
        device = torch.device("cpu")

        result = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            max_epochs=100,
            early_stopping_patience=2,
            learning_rate=1e-6,
        )
        assert len(result.history) < 100

    def test_checkpoint_saved(self, tmp_path) -> None:
        """Best model checkpoint should be saved to checkpoint_dir."""
        model = _make_simple_model()
        train_loader = _make_loader()
        val_loader = _make_loader()
        device = torch.device("cpu")

        result = train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            max_epochs=3,
            early_stopping_patience=10,
            checkpoint_dir=tmp_path,
        )
        assert result.best_checkpoint_path is not None
        assert result.best_checkpoint_path.exists()

    def test_best_epoch_tracked(self) -> None:
        """best_epoch should be set to the epoch with best validation loss."""
        model = _make_simple_model()
        train_loader = _make_loader()
        val_loader = _make_loader()
        device = torch.device("cpu")

        result = train_model(model, train_loader, val_loader, device=device, max_epochs=5, early_stopping_patience=10)
        assert 1 <= result.best_epoch <= 5
        assert result.best_val_loss < float("inf")

    def test_with_explicit_class_weights(self) -> None:
        """train_model should accept explicit class weights for backward compat."""
        model = _make_simple_model()
        train_loader = _make_loader()
        val_loader = _make_loader()
        device = torch.device("cpu")
        class_weights = torch.tensor([1.0, 1.0])

        result = train_model(model, train_loader, val_loader, class_weights=class_weights, device=device, max_epochs=2)
        assert isinstance(result, TrainResult)
        assert len(result.history) == 2
