"""Unit tests for FSTClassifier model."""

import pytest
import torch

from skin_type_classifier.model import FSTClassifier


@pytest.mark.unit
class TestFSTClassifier:
    """Test the FSTClassifier architecture and freezing behaviour."""

    def test_output_shape(self) -> None:
        """Output should be (batch_size, num_classes) for a batch of images."""
        model = FSTClassifier(num_classes=6, freeze_backbone=True)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 6)

    def test_frozen_backbone_has_no_grad(self) -> None:
        """Backbone params should have requires_grad=False when frozen."""
        model = FSTClassifier(freeze_backbone=True)
        for p in model.features.parameters():
            assert not p.requires_grad

    def test_unfrozen_backbone_has_grad(self) -> None:
        """Backbone params should have requires_grad=True when not frozen."""
        model = FSTClassifier(freeze_backbone=False)
        for p in model.features.parameters():
            assert p.requires_grad

    def test_classifier_head_always_trainable(self) -> None:
        """Head params should always be trainable regardless of freeze setting."""
        model = FSTClassifier(freeze_backbone=True)
        for p in model.classifier.parameters():
            assert p.requires_grad

    def test_trainable_params_frozen(self) -> None:
        """Frozen model should have very few trainable params (just the head)."""
        model = FSTClassifier(freeze_backbone=True)
        assert model.trainable_params < 10_000

    def test_trainable_params_unfrozen(self) -> None:
        """Unfrozen model should have millions of trainable params."""
        model = FSTClassifier(freeze_backbone=False)
        assert model.trainable_params > 50_000_000

    def test_custom_num_classes(self) -> None:
        """Model should support a custom number of output classes."""
        model = FSTClassifier(num_classes=3, freeze_backbone=True)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 3)
