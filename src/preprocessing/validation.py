from pathlib import Path

import numpy as np
from PIL import Image

from .config import VALID_LABELS


def validate_image(image_path: str | Path) -> dict:
    """
    Validate that an image can be opened and contains usable RGB data.
    """

    image_path = Path(image_path)

    result = {
        "valid": False,
        "error": None,
        "width": None,
        "height": None,
        "mode": None,
    }

    try:
        with Image.open(image_path) as image:
            image.load()

            result["width"] = image.width
            result["height"] = image.height
            result["mode"] = image.mode
            result["valid"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


def validate_mask(mask_path: str | Path) -> dict:
    """
    Validate a LoveDA segmentation mask.
    """

    mask_path = Path(mask_path)

    result = {
        "valid": False,
        "error": None,
        "width": None,
        "height": None,
        "label_values": [],
    }

    try:
        with Image.open(mask_path) as mask:
            mask.load()

            array = np.array(mask)

            if array.ndim != 2:
                result["error"] = (
                    f"Expected single-channel segmentation mask, "
                    f"received shape {array.shape}"
                )
                return result

            labels = set(np.unique(array).tolist())

            invalid_labels = labels - VALID_LABELS

            result["width"] = mask.width
            result["height"] = mask.height
            result["label_values"] = sorted(labels)

            if invalid_labels:
                result["error"] = (
                    f"Unexpected label IDs found: "
                    f"{sorted(invalid_labels)}"
                )
                return result

            result["valid"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


def validate_image_mask_pair(
    image_path: str | Path,
    mask_path: str | Path,
) -> dict:
    """
    Validate image and mask together.
    """

    image_result = validate_image(image_path)
    mask_result = validate_mask(mask_path)

    result = {
        "valid": False,
        "image": image_result,
        "mask": mask_result,
        "error": None,
    }

    if not image_result["valid"]:
        result["error"] = (
            f"Invalid image: {image_result['error']}"
        )
        return result

    if not mask_result["valid"]:
        result["error"] = (
            f"Invalid mask: {mask_result['error']}"
        )
        return result

    if (
        image_result["width"] != mask_result["width"]
        or image_result["height"] != mask_result["height"]
    ):
        result["error"] = (
            "Image and mask dimensions do not match"
        )
        return result

    result["valid"] = True

    return result