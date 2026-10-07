from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_LABELS = set(range(8))
IGNORE_LABEL = 0

DEFAULT_IMAGE_SIZE = (512, 512)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}