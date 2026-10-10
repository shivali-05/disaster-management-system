from pathlib import Path

import numpy as np

from .validation import (
    validate_image_mask_pair,
)

from .image_preprocessing import (
    preprocess_image,
)

from .mask_preprocessing import (
    preprocess_mask,
)


def preprocess_sample(
    image_path: str | Path,
    mask_path: str | Path,
    image_size: tuple[int, int] = (512, 512),
) -> dict:
    """
    End-to-end preprocessing for one LoveDA sample.
    """

    validation = validate_image_mask_pair(
        image_path,
        mask_path,
    )

    if not validation["valid"]:

        raise ValueError(
            validation["error"]
        )

    image = preprocess_image(
        image_path,
        size=image_size,
    )

    mask = preprocess_mask(
        mask_path,
        size=image_size,
    )

    return {
        "image": image,
        "mask": mask,
        "metadata": {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_width": (
                validation["image"]["width"]
            ),
            "original_height": (
                validation["image"]["height"]
            ),
            "processed_width": image_size[0],
            "processed_height": image_size[1],
            "labels": (
                validation["mask"]["label_values"]
            ),
        },
    }