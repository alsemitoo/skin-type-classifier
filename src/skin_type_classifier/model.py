"""EfficientNetV2-M transfer learning model for FST classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_M_Weights, efficientnet_v2_m


class FSTClassifier(nn.Module):
    """EfficientNetV2-M backbone with a custom classification head.

    Args:
        num_classes: Number of output classes (default 6 for FST 1-6).
        dropout: Dropout probability before the final linear layer.
        freeze_backbone: If True, freeze all backbone (feature extractor) weights.
    """

    BACKBONE_FEATURES = 1280

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        base = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(self.BACKBONE_FEATURES, num_classes),
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features -> pool -> flatten -> classify."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @property
    def trainable_params(self) -> int:
        """Count of parameters with requires_grad=True."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
