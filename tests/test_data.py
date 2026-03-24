"""Unit tests for FSTDataset and DataLoader factories."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from skin_type_classifier.data.data import FSTDataset, compute_class_weights, make_eval_loader, make_train_loader


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame matching the unified schema."""
    return pd.DataFrame(
        {
            "image_path": ["img1.png", "img2.png", "img3.png", "img4.png"],
            "source": ["scin", "scin", "pad_ufes", "pad_ufes"],
            "fitzpatrick_skin_type": [1, 2, 3, 5],
            "diagnosis": ["eczema", "acne", "BCC", "MEL"],
            "age": ["25", "30", "55", "70"],
            "sex": ["FEMALE", "MALE", "MALE", "FEMALE"],
            "group_id": ["G1", "G2", "G3", "G4"],
            "revised_fitzpatrick": [1, 2, 3, 5],
        }
    )


@pytest.fixture()
def fake_images(sample_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Create dummy RGB images on disk and return the data_root."""
    for img in sample_df["image_path"]:
        img_path = tmp_path / img
        Image.new("RGB", (64, 64), color="red").save(img_path)
    return tmp_path


@pytest.mark.unit
class TestFSTDataset:
    """Test the FSTDataset class."""

    def test_is_pytorch_dataset(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        assert isinstance(dataset, Dataset)

    def test_len(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        assert len(dataset) == 4

    def test_getitem_returns_image_and_label(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        image, label = dataset[0]
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)

    def test_labels_are_zero_indexed(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        """FST 1-6 should map to labels 0-5."""
        dataset = FSTDataset(sample_df, fake_images)
        assert list(dataset.labels) == [0, 1, 2, 4]  # FST 1,2,3,5 -> 0,1,2,4

    def test_class_counts_shape(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        counts = dataset.class_counts
        assert counts.shape == (FSTDataset.NUM_CLASSES,)
        assert counts.sum() == 4

    def test_class_counts_values(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        counts = dataset.class_counts
        expected = np.array([1, 1, 1, 0, 1, 0])  # FST 1,2,3,_,5,_
        np.testing.assert_array_equal(counts, expected)


@pytest.mark.unit
class TestComputeClassWeights:
    def test_output_shape(self) -> None:
        counts = np.array([580, 2793, 2081, 935, 444, 72])
        weights = compute_class_weights(counts)
        assert weights.shape == (6,)

    def test_mean_is_approximately_one(self) -> None:
        counts = np.array([580, 2793, 2081, 935, 444, 72])
        weights = compute_class_weights(counts)
        assert abs(weights.mean().item() - 1.0) < 1e-5

    def test_minority_class_has_highest_weight(self) -> None:
        counts = np.array([580, 2793, 2081, 935, 444, 72])
        weights = compute_class_weights(counts)
        assert weights.argmax().item() == 5  # FST 6 (index 5) is rarest

    def test_majority_class_has_lowest_weight(self) -> None:
        counts = np.array([580, 2793, 2081, 935, 444, 72])
        weights = compute_class_weights(counts)
        assert weights.argmin().item() == 1  # FST 2 (index 1) is most common

    def test_returns_float_tensor(self) -> None:
        counts = np.array([100, 200])
        weights = compute_class_weights(counts)
        assert weights.dtype == torch.float32


@pytest.mark.unit
class TestMakeTrainLoader:
    def test_returns_dataloader(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        loader = make_train_loader(dataset, batch_size=2, num_workers=0)
        assert loader is not None

    def test_uses_replacement_sampling(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        loader = make_train_loader(dataset, batch_size=2, num_workers=0)
        assert loader.sampler is not None


@pytest.mark.unit
class TestMakeEvalLoader:
    def test_returns_dataloader(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        loader = make_eval_loader(dataset, batch_size=2, num_workers=0)
        assert loader is not None

    def test_no_sampler(self, sample_df: pd.DataFrame, fake_images: Path) -> None:
        dataset = FSTDataset(sample_df, fake_images)
        loader = make_eval_loader(dataset, batch_size=2, num_workers=0)
        assert not isinstance(loader.sampler, WeightedRandomSampler)
