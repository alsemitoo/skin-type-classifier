"""Image transform pipelines for FST classification.

Key domain constraint: Fitzpatrick Skin Type classification depends on skin
chromaticity (hue/saturation). Augmentations that shift hue or saturation are
forbidden because they alter the discriminative signal between skin types.
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Same idea as mean and std computed in the analysis notebook
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224
RESIZE_SIZE = 256


def get_train_transform() -> transforms.Compose:
    """Return training transform pipeline.

    Uses geometric augmentations (safe for FST) and very conservative color
    augmentation (brightness/contrast only — saturation and hue are zero).
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.10)),
        ]
    )


def get_eval_transform() -> transforms.Compose:
    """Return evaluation (validation/test) transform pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
