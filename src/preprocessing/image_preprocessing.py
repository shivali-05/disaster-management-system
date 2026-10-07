from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """
    Load image and guarantee RGB format.
    """

    image = Image.open(image_path)

    return image.convert("RGB")


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """
    Resize satellite image using bilinear interpolation.
    """

    return image.resize(
        size,
        Image.Resampling.BILINEAR,
    )


def image_to_numpy(
    image: Image.Image,
) -> np.ndarray:
    """
    Convert PIL RGB image to float32 NumPy array in range [0, 1].
    """

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    return array / 255.0


def normalize_image(
    image: np.ndarray,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Normalize RGB image.

    If dataset-specific mean/std are not yet computed,
    standard ImageNet normalization can temporarily be used
    for transfer-learning models.
    """

    if mean is None:
        mean = (0.485, 0.456, 0.406)

    if std is None:
        std = (0.229, 0.224, 0.225)

    mean_array = np.asarray(
        mean,
        dtype=np.float32,
    )

    std_array = np.asarray(
        std,
        dtype=np.float32,
    )

    return (
        image - mean_array
    ) / std_array


def preprocess_image(
    image_path: str | Path,
    size: tuple[int, int] = (512, 512),
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Complete image preprocessing pipeline.
    """

    image = load_rgb_image(image_path)

    image = resize_image(
        image,
        size,
    )

    image_array = image_to_numpy(
        image
    )

    image_array = normalize_image(
        image_array,
        mean=mean,
        std=std,
    )

    return image_array