"""Unit tests for image transform pipelines."""

import torch
import pytest
from PIL import Image
from torchvision.transforms import ColorJitter

from skin_type_classifier.data.transforms import get_eval_transform, get_train_transform, IMAGE_SIZE


@pytest.mark.unit
class TestTrainTransform:
    def test_output_is_tensor(self) -> None:
        transform = get_train_transform()
        img = Image.new("RGB", (256, 256), color="blue")
        result = transform(img)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self) -> None:
        transform = get_train_transform()
        img = Image.new("RGB", (256, 256), color="blue")
        result = transform(img)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_no_hue_or_saturation_jitter(self) -> None:
        """ColorJitter in the pipeline must NOT alter hue or saturation."""
        transform = get_train_transform()
        for t in transform.transforms:
            if isinstance(t, ColorJitter):
                assert t.hue == 0 or t.hue is None, "Hue jitter is forbidden for FST classification"
                assert t.saturation == 0 or t.saturation is None, (
                    "Saturation jitter is forbidden for FST classification"
                )


@pytest.mark.unit
class TestEvalTransform:
    def test_output_is_tensor(self) -> None:
        transform = get_eval_transform()
        img = Image.new("RGB", (256, 256), color="blue")
        result = transform(img)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self) -> None:
        transform = get_eval_transform()
        img = Image.new("RGB", (256, 256), color="blue")
        result = transform(img)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_deterministic(self) -> None:
        """Eval transform should produce identical outputs for identical inputs."""
        transform = get_eval_transform()
        img = Image.new("RGB", (256, 256), color="green")
        a = transform(img)
        b = transform(img)
        assert torch.equal(a, b)
