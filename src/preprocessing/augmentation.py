import random

import numpy as np
from PIL import Image


def random_horizontal_flip(
    image: Image.Image,
    mask: Image.Image,
    probability: float = 0.5,
):
    """
    Apply the same transformation to image and mask.
    """

    if random.random() < probability:

        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        mask = mask.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    return image, mask


def random_vertical_flip(
    image: Image.Image,
    mask: Image.Image,
    probability: float = 0.5,
):

    if random.random() < probability:

        image = image.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

        mask = mask.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

    return image, mask