"""Fitzpatrick Skin Type classification dataset and DataLoader factories."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class FSTDataset(Dataset):
    """Fitzpatrick Skin Type classification dataset.

    Loads images from disk and returns (image_tensor, label) pairs.
    Labels use the ``revised_fitzpatrick`` column (majority-voted) and
    are 0-indexed (FST 1 -> 0, ..., FST 6 -> 5).
    """

    NUM_CLASSES = 6

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform
        self.labels: np.ndarray = (self.df["revised_fitzpatrick"].values - 1).astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | Image.Image, int]:
        row = self.df.iloc[index]
        img_path = self.data_root / row["image_path"]
        image: torch.Tensor | Image.Image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])

    @property
    def class_counts(self) -> np.ndarray:
        """Return per-class sample count array of shape ``(NUM_CLASSES,)``."""
        return np.bincount(self.labels, minlength=self.NUM_CLASSES)


def compute_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    """Compute sqrt-inverse-frequency class weights for CrossEntropyLoss.

    The sqrt dampening prevents extreme minority classes (e.g. FST 6 with 72
    samples vs FST 2 with 2793) from dominating the loss while still penalising
    minority-class errors more than majority-class errors.

    Returns:
        Float tensor of shape ``(num_classes,)`` with mean weight ~1.0.
    """
    counts = class_counts.astype(np.float64)
    inv_freq = 1.0 / counts
    weights = np.sqrt(inv_freq)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_train_loader(
    dataset: FSTDataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Create a training DataLoader with class-balanced sampling.

    Uses ``WeightedRandomSampler`` with ``replacement=True`` so that minority
    classes (especially FST 6 with only ~72 samples) are oversampled to appear
    roughly equally often per epoch.  Each resampled image receives different
    random augmentations from the transform pipeline.
    """
    class_counts = dataset.class_counts.astype(np.float64)
    sample_weights = 1.0 / class_counts[dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_eval_loader(
    dataset: FSTDataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Create a validation/test DataLoader (sequential, no sampling)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
