import numpy as np


CLASS_IDS = {
    "building": 2,
    "road": 3,
    "water": 4,
    "barren": 5,
    "forest": 6,
    "agriculture": 7,
}


def calculate_area_ratio(
    segmentation_mask: np.ndarray,
    class_id: int,
) -> float:

    total_pixels = segmentation_mask.size

    class_pixels = np.sum(
        segmentation_mask == class_id
    )

    return float(
        class_pixels / total_pixels
    )


def extract_segmentation_features(
    segmentation_mask: np.ndarray,
) -> dict:
    """
    Convert segmentation output into structured
    IVP risk-assessment features.
    """

    features = {}

    for class_name, class_id in CLASS_IDS.items():

        features[
            f"{class_name}_area_ratio"
        ] = calculate_area_ratio(
            segmentation_mask,
            class_id,
        )

    features[
        "vegetation_area_ratio"
    ] = (
        features["forest_area_ratio"]
        +
        features["agriculture_area_ratio"]
    )

    features[
        "built_up_density"
    ] = (
        features["building_area_ratio"]
        +
        features["road_area_ratio"]
    )

    return features