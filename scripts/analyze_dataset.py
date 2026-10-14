from pathlib import Path
import json
from collections import Counter

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "loveda"
    / "Train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "class_distribution.json"
)


CLASS_NAMES = {
    0: "ignore",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}


def analyze_mask(mask_path: Path) -> Counter:
    """
    Count the number of pixels belonging to
    each semantic class in one mask.
    """

    with Image.open(mask_path) as mask:
        mask_array = np.array(mask)

    labels, counts = np.unique(
        mask_array,
        return_counts=True,
    )

    return Counter(
        {
            int(label): int(count)
            for label, count
            in zip(labels, counts)
        }
    )


def get_mask_files(scene_type: str):
    """
    Get all masks for Urban or Rural data.
    """

    mask_dir = (
        DATA_ROOT
        / scene_type
        / "masks_png"
    )

    return sorted(
        mask_dir.glob("*.png")
    )


def main():

    print("=" * 60)
    print("LoveDA Dataset Class Distribution Analysis")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_counts = Counter()

    scene_counts = {
        "urban": Counter(),
        "rural": Counter(),
    }

    for scene_type in [
        "Urban",
        "Rural",
    ]:

        mask_files = get_mask_files(
            scene_type
        )

        print(
            f"\nAnalyzing {scene_type}: "
            f"{len(mask_files)} masks"
        )

        for index, mask_path in enumerate(
            mask_files,
            start=1,
        ):

            counts = analyze_mask(
                mask_path
            )

            overall_counts.update(
                counts
            )

            scene_counts[
                scene_type.lower()
            ].update(
                counts
            )

            if index % 20 == 0:

                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(mask_files)}"
                )

    total_pixels = sum(
        overall_counts.values()
    )

    output = {
        "dataset": "LoveDA MVP subset",
        "total_samples": 200,
        "total_pixels": int(
            total_pixels
        ),
        "classes": {},
        "scene_distribution": {},
    }

    for class_id in range(8):

        pixel_count = overall_counts[
            class_id
        ]

        percentage = (
            pixel_count
            / total_pixels
            * 100
            if total_pixels > 0
            else 0
        )

        output["classes"][
            str(class_id)
        ] = {
            "name": CLASS_NAMES[
                class_id
            ],
            "pixel_count": int(
                pixel_count
            ),
            "percentage": round(
                percentage,
                4,
            ),
        }

    for scene_name, counts in (
        scene_counts.items()
    ):

        scene_total = sum(
            counts.values()
        )

        output[
            "scene_distribution"
        ][scene_name] = {}

        for class_id in range(8):

            pixel_count = counts[
                class_id
            ]

            percentage = (
                pixel_count
                / scene_total
                * 100
                if scene_total > 0
                else 0
            )

            output[
                "scene_distribution"
            ][scene_name][
                str(class_id)
            ] = {
                "name": CLASS_NAMES[
                    class_id
                ],
                "pixel_count": int(
                    pixel_count
                ),
                "percentage": round(
                    percentage,
                    4,
                ),
            }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(
        "Class distribution analysis completed."
    )
    print("=" * 60)

    print(
        f"Output saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nOverall class distribution:\n")

    for class_id in range(8):

        info = output[
            "classes"
        ][str(class_id)]

        print(
            f"{class_id} | "
            f"{info['name']:12} | "
            f"{info['percentage']:8.4f}%"
        )


if __name__ == "__main__":
    main()