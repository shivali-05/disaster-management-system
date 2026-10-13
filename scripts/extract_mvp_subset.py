from pathlib import Path
import random
import zipfile


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "loveda"
    / "Train.zip"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "loveda"
)

SAMPLES_PER_SCENE = 100

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def get_image_files(zip_file, scene_type):
    """
    Return valid LoveDA image file paths inside the ZIP
    for one scene type.

    Ignores:
    - __MACOSX
    - ._ macOS metadata files
    """

    prefix = f"Train/{scene_type}/images_png/"

    image_files = []

    for name in zip_file.namelist():

        if not name.startswith(prefix):
            continue

        if name.endswith("/"):
            continue

        if "__MACOSX" in name:
            continue

        filename = Path(name).name

        if filename.startswith("._"):
            continue

        if Path(name).suffix.lower() != ".png":
            continue

        image_files.append(name)

    return sorted(image_files)


def select_samples(image_files, count, rng):
    """
    Select a reproducible random subset.
    """

    if len(image_files) < count:
        raise ValueError(
            f"Requested {count} samples, "
            f"but only found {len(image_files)}."
        )

    return sorted(
        rng.sample(image_files, count)
    )


def extract_pair(zip_file, image_zip_path):
    """
    Extract one image and its matching mask.
    """

    image_path = Path(image_zip_path)

    scene_type = image_path.parts[1]
    filename = image_path.name

    mask_zip_path = (
        f"Train/{scene_type}/masks_png/{filename}"
    )

    if mask_zip_path not in zip_file.namelist():

        raise FileNotFoundError(
            f"Matching mask not found for "
            f"{image_zip_path}: {mask_zip_path}"
        )

    zip_file.extract(
        image_zip_path,
        OUTPUT_ROOT
    )

    zip_file.extract(
        mask_zip_path,
        OUTPUT_ROOT
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"Train ZIP not found: {ZIP_PATH}"
        )

    print("=" * 60)
    print("LoveDA MVP Subset Extraction")
    print("=" * 60)

    print(f"ZIP: {ZIP_PATH}")
    print(f"Output: {OUTPUT_ROOT}")
    print(
        f"Samples per scene: "
        f"{SAMPLES_PER_SCENE}"
    )
    print(
        f"Total target samples: "
        f"{SAMPLES_PER_SCENE * 2}"
    )
    print(f"Random seed: {RANDOM_SEED}")
    print()

    rng = random.Random(RANDOM_SEED)

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zip_file:

        for scene_type in [
            "Urban",
            "Rural",
        ]:

            print(
                f"Processing {scene_type}..."
            )

            image_files = get_image_files(
                zip_file,
                scene_type,
            )

            print(
                f"Available images: "
                f"{len(image_files)}"
            )

            selected_images = select_samples(
                image_files,
                SAMPLES_PER_SCENE,
                rng,
            )

            for index, image_path in enumerate(
                selected_images,
                start=1,
            ):

                extract_pair(
                    zip_file,
                    image_path,
                )

                if (
                    index % 10 == 0
                    or index == len(
                        selected_images
                    )
                ):

                    print(
                        f"Extracted "
                        f"{index}/"
                        f"{len(selected_images)}"
                    )

            print(
                f"{scene_type} extraction complete."
            )
            print()

    print("=" * 60)
    print(
        "MVP subset extraction completed."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()