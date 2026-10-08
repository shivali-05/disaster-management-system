from pathlib import Path

import numpy as np
from PIL import Image

from .config import VALID_LABELS


def load_mask(
    mask_path: str | Path,
) -> Image.Image:
    """
    Load segmentation mask without RGB conversion.
    """

    return Image.open(mask_path)


def resize_mask(
    mask: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """
    Resize segmentation mask.

    NEAREST interpolation is mandatory because
    label values must not be interpolated.
    """

    return mask.resize(
        size,
        Image.Resampling.NEAREST,
    )


def mask_to_numpy(
    mask: Image.Image,
) -> np.ndarray:

    array = np.asarray(
        mask,
        dtype=np.int64,
    )

    labels = set(
        np.unique(array).tolist()
    )

    invalid_labels = (
        labels - VALID_LABELS
    )

    if invalid_labels:
        raise ValueError(
            f"Invalid LoveDA labels: "
            f"{sorted(invalid_labels)}"
        )

    return array


def preprocess_mask(
    mask_path: str | Path,
    size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Complete mask preprocessing pipeline.
    """

    mask = load_mask(
        mask_path
    )

    mask = resize_mask(
        mask,
        size,
    )

    return mask_to_numpy(
        mask
    )