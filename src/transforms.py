"""Image preprocessing and train-only augmentation."""

from __future__ import annotations

from typing import Any

from PIL import Image

try:
    from torchvision import transforms
except (ImportError, RuntimeError) as error:  # pragma: no cover - depends on optional environment setup
    transforms = None
    _TORCHVISION_IMPORT_ERROR = error
else:
    _TORCHVISION_IMPORT_ERROR = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = False) -> Any:
    """Build deterministic ImageNet preprocessing, with conservative train augmentation."""
    if transforms is None:
        raise ImportError(
            "torchvision is required for image transforms. Install a torchvision build "
            "compatible with the selected PyTorch version."
        ) from _TORCHVISION_IMPORT_ERROR
    operations: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.04, 0.04),
                    scale=(0.92, 1.08),
                    fill=(0, 0, 0),
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def preprocess_pil(image: Image.Image, image_size: int = 224) -> Any:
    """Convert a PIL image into a normalized RGB tensor."""
    return build_transforms(image_size=image_size, train=False)(image.convert("RGB"))
