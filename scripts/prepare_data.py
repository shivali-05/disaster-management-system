"""
LoveDA Dataset Preparation & Inspection Pipeline
"""

import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_data")

# Recognized LoveDA directory tokens
IMAGE_DIR_NAMES = {"images_png", "images", "img_png", "img"}
MASK_DIR_NAMES = {"masks_png", "masks", "labels_png", "labels"}
SPLIT_NAMES = {"train", "val", "test"}
SCENE_TYPES = {"urban", "rural"}
SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads data configuration from configs/data.yaml."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "data.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_valid_label_ids(mapping_path: Optional[Path] = None) -> Set[int]:
    """
    Loads valid segmentation label IDs from class_mapping.json.
    Falls back to official LoveDA label range (0-7) if mapping file is unavailable.
    """
    if mapping_path is None:
        mapping_path = get_project_root() / "data" / "metadata" / "class_mapping.json"

    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                seg_map = data.get("segmentation", {})
                valid_ids = {int(k) for k in seg_map.keys()}
                if valid_ids:
                    return valid_ids
        except Exception as e:
            logger.warning(f"Could not parse {mapping_path}: {e}. Using default range 0-7.")

    # Default LoveDA labels: 0=ignore, 1=background, 2=building, 3=road, 4=water, 5=barren, 6=forest, 7=agriculture
    return {0, 1, 2, 3, 4, 5, 6, 7}


def check_raw_dataset_exists(raw_root: Path) -> Tuple[bool, List[str]]:
    """
    Checks whether the raw dataset directory exists and contains actual dataset contents
    (excluding hidden files like .gitkeep, .DS_Store).
    """
    if not raw_root.exists():
        return False, []

    visible_items = [
        item for item in os.listdir(raw_root)
        if not item.startswith(".")
    ]
    return len(visible_items) > 0, visible_items


def inspect_raw_structure(raw_root: Path) -> Dict[str, Any]:
    """
    Inspects the actual directory tree of the extracted raw dataset.
    Returns discovered directory layout and image/mask file counts.
    """
    inspection = {
        "root_path": str(raw_root),
        "subdirectories": [],
        "total_png_files": 0,
        "total_tif_files": 0,
    }

    for dirpath, dirnames, filenames in os.walk(raw_root):
        rel_dir = os.path.relpath(dirpath, raw_root)
        if rel_dir != ".":
            inspection["subdirectories"].append(rel_dir)

        png_count = sum(1 for f in filenames if f.lower().endswith(".png"))
        tif_count = sum(1 for f in filenames if f.lower().endswith((".tif", ".tiff")))
        inspection["total_png_files"] += png_count
        inspection["total_tif_files"] += tif_count

    return inspection


def is_image_dir(dir_name: str) -> bool:
    """Identifies if a directory name corresponds to an imagery directory."""
    name_lower = dir_name.lower()
    return name_lower in IMAGE_DIR_NAMES or ("image" in name_lower and "mask" not in name_lower)


def is_mask_dir(dir_name: str) -> bool:
    """Identifies if a directory name corresponds to a mask/label directory."""
    name_lower = dir_name.lower()
    return name_lower in MASK_DIR_NAMES or "mask" in name_lower or "label" in name_lower


def infer_split_and_scene(path: Path, raw_root: Optional[Path] = None) -> Tuple[str, str, bool]:
    """
    Infers the official split (train/val/test) and scene_type (urban/rural)
    from path components relative to raw_root.

    Returns:
        (split, scene_type, is_confident)
        where is_confident is True only if BOTH split and scene_type were confidently identified.
    """
    if raw_root is not None:
        try:
            rel_parts = [p.lower() for p in path.relative_to(raw_root).parts]
        except ValueError:
            rel_parts = [p.lower() for p in path.parts]
    else:
        rel_parts = [p.lower() for p in path.parts]

    found_split: Optional[str] = None
    found_scene: Optional[str] = None

    for part in rel_parts:
        for s in SPLIT_NAMES:
            if s == part or s in part:
                found_split = s
                break
        for sc in SCENE_TYPES:
            if sc == part or sc in part:
                found_scene = sc
                break

    is_confident = (found_split is not None) and (found_scene is not None)
    split = found_split if found_split is not None else "unspecified"
    scene_type = found_scene if found_scene is not None else "unspecified"

    return split, scene_type, is_confident


def discover_loveda_samples(raw_root: Path, project_root: Path) -> List[Dict[str, Any]]:
    """
    Discovers image and mask pairs from the extracted LoveDA dataset structure.

    Responsibilities:
      1. Explicitly identifies image directories vs mask directories.
      2. Deterministically pairs files using (split, scene_type, stem).
      3. Detects orphan masks (mask exists without image) and orphan images (image without mask).
      4. Flags unspecified / unclassifiable metadata clearly.
    """
    discovered: List[Dict[str, Any]] = []

    # Map keys: (split, scene_type, stem) -> Path
    image_files: Dict[Tuple[str, str, str], Path] = {}
    mask_files: Dict[Tuple[str, str, str], Path] = {}
    unclassified_files: List[Tuple[Path, str]] = []

    for dirpath, _, filenames in os.walk(raw_root):
        dp = Path(dirpath)
        dir_name = dp.name

        in_img_dir = is_image_dir(dir_name)
        in_msk_dir = is_mask_dir(dir_name)

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_path = dp / fname
            split, scene_type, confident = infer_split_and_scene(file_path, raw_root)
            stem = Path(fname).stem

            if not confident:
                unclassified_files.append((
                    file_path,
                    f"Cannot confidently infer split ('{split}') or scene ('{scene_type}')"
                ))
                continue

            pair_key = (split, scene_type, stem)

            if in_msk_dir:
                mask_files[pair_key] = file_path
            elif in_img_dir:
                image_files[pair_key] = file_path
            else:
                # File is in an unrecognized directory (neither image nor mask folder)
                unclassified_files.append((
                    file_path,
                    f"File located in unclassified directory: '{dir_name}'"
                ))

    # Process all unclassified files as invalid samples
    for unclass_path, reason in unclassified_files:
        try:
            rel_path = str(unclass_path.relative_to(project_root))
        except ValueError:
            rel_path = str(unclass_path)

        split, scene_type, _ = infer_split_and_scene(unclass_path, raw_root)
        stem = unclass_path.stem
        sample_id = f"{split}_{scene_type}_{stem}"

        logger.warning(f"Unclassified dataset file detected: {rel_path} - {reason}")
        discovered.append({
            "sample_id": sample_id,
            "split": split,
            "scene_type": scene_type,
            "image_path_abs": unclass_path,
            "mask_path_abs": None,
            "image_path": rel_path,
            "mask_path": "",
            "structural_error": f"Unspecified metadata: {reason}",
        })

    # All unique pairing keys across images and masks
    all_keys = sorted(list(set(image_files.keys()) | set(mask_files.keys())))

    for pair_key in all_keys:
        split, scene_type, stem = pair_key
        img_path = image_files.get(pair_key)
        mask_path = mask_files.get(pair_key)
        sample_id = f"{split}_{scene_type}_{stem}"

        rel_img = ""
        if img_path is not None:
            try:
                rel_img = str(img_path.relative_to(project_root))
            except ValueError:
                rel_img = str(img_path)

        rel_mask = ""
        if mask_path is not None:
            try:
                rel_mask = str(mask_path.relative_to(project_root))
            except ValueError:
                rel_mask = str(mask_path)

        structural_error = ""
        # Check orphan conditions
        if img_path is None and mask_path is not None:
            # Mask exists but image is missing (orphan mask)
            structural_error = f"Orphan mask: corresponding image missing for {split}/{scene_type}/{stem}"
            logger.warning(f"Orphan mask detected: {rel_mask} (missing image)")
        elif img_path is not None and mask_path is None:
            # Image exists but mask is missing
            if split in ("train", "val"):
                structural_error = f"Missing required mask for {split} image sample ({stem})"
                logger.warning(f"Orphan image detected without required mask: {rel_img}")
            # Test split without mask is acceptable in LoveDA benchmark

        discovered.append({
            "sample_id": sample_id,
            "split": split,
            "scene_type": scene_type,
            "image_path_abs": img_path,
            "mask_path_abs": mask_path,
            "image_path": rel_img,
            "mask_path": rel_mask,
            "structural_error": structural_error,
        })

    return discovered


def validate_sample(
    sample: Dict[str, Any],
    valid_label_ids: Set[int],
) -> Dict[str, Any]:
    """
    Validates a single sample:
      - Checks structural errors (unspecified metadata, orphan masks, missing masks).
      - Checks image readability and dimensions (if image present).
      - Checks mask readability, dimensions match, and valid label values (if mask present).
    """
    img_abs: Optional[Path] = sample.get("image_path_abs")
    mask_abs: Optional[Path] = sample.get("mask_path_abs")
    split: str = sample.get("split", "unspecified")
    scene_type: str = sample.get("scene_type", "unspecified")
    structural_error: str = sample.get("structural_error", "")

    width: int = 0
    height: int = 0
    valid: bool = True
    validation_error: str = ""
    label_values: List[int] = []

    # 1. Check structural / metadata errors first
    if split == "unspecified" or scene_type == "unspecified":
        valid = False
        validation_error = structural_error or f"Unspecified metadata (split={split}, scene_type={scene_type})"
    elif structural_error:
        valid = False
        validation_error = structural_error

    # 2. Validate Image (if present)
    if img_abs is not None:
        if not img_abs.exists():
            valid = False
            validation_error = validation_error or f"Image file not found: {img_abs}"
        else:
            try:
                with Image.open(img_abs) as img:
                    width, height = img.size
            except Exception as e:
                valid = False
                validation_error = validation_error or f"Failed to open image file: {e}"
    else:
        # No image file (e.g. orphan mask)
        valid = False
        validation_error = validation_error or "Image file missing (orphan mask)"

    # 3. Validate Mask (if present)
    if mask_abs is not None:
        if not mask_abs.exists():
            if split in ("train", "val"):
                valid = False
                validation_error = validation_error or f"Mask file not found: {mask_abs}"
        else:
            try:
                with Image.open(mask_abs) as msk:
                    m_width, m_height = msk.size
                    if width > 0 and height > 0 and (m_width, m_height) != (width, height):
                        valid = False
                        validation_error = validation_error or (
                            f"Dimension mismatch: image is {width}x{height}, "
                            f"mask is {m_width}x{m_height}"
                        )
                    elif width == 0 and height == 0:
                        # Orphan mask: capture mask dimensions for diagnostic summary
                        width, height = m_width, m_height

                    mask_arr = np.array(msk)
                    unique_vals = [int(v) for v in np.unique(mask_arr)]
                    label_values = sorted(unique_vals)

                    invalid_labels = set(label_values) - valid_label_ids
                    if invalid_labels:
                        valid = False
                        validation_error = validation_error or (
                            f"Invalid label IDs found in mask: {sorted(list(invalid_labels))}. "
                            f"Allowed label IDs: {sorted(list(valid_label_ids))}"
                        )
            except Exception as e:
                valid = False
                validation_error = validation_error or f"Failed to read mask file: {e}"
    else:
        # Mask is absent
        if split in ("train", "val"):
            valid = False
            validation_error = validation_error or f"Missing required mask for {split} sample"
        elif split == "test":
            # Test split without mask is valid in LoveDA
            if valid:
                validation_error = ""

    return {
        **sample,
        "width": width,
        "height": height,
        "valid": valid,
        "validation_error": validation_error,
        "label_values": label_values,
    }


def generate_manifest(
    validated_samples: List[Dict[str, Any]],
    manifest_path: Path,
) -> Path:
    """
    Generates the LoveDA manifest CSV file.
    Selected is set to false initially (MVP subset selection occurs in Phase 2B).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "split",
        "scene_type",
        "image_path",
        "mask_path",
        "width",
        "height",
        "valid",
        "validation_error",
        "label_values",
        "selected",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in validated_samples:
            label_str = ";".join(str(v) for v in s.get("label_values", []))
            writer.writerow({
                "sample_id": s["sample_id"],
                "split": s["split"],
                "scene_type": s["scene_type"],
                "image_path": s["image_path"],
                "mask_path": s["mask_path"],
                "width": s["width"],
                "height": s["height"],
                "valid": "true" if s["valid"] else "false",
                "validation_error": s["validation_error"],
                "label_values": label_str,
                "selected": "false",
            })

    logger.info(f"Generated manifest with {len(validated_samples)} entries at {manifest_path}")
    return manifest_path


def generate_dataset_summary(
    validated_samples: List[Dict[str, Any]],
    summary_path: Path,
) -> Path:
    """
    Generates dataset summary JSON file capturing distribution statistics.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, int] = {}
    scene_types: Dict[str, int] = {}
    observed_labels: Set[int] = set()
    dimensions: Set[str] = set()
    valid_count = 0
    invalid_count = 0

    for s in validated_samples:
        if s["valid"]:
            valid_count += 1
        else:
            invalid_count += 1

        split = s["split"]
        splits[split] = splits.get(split, 0) + 1

        scene = s["scene_type"]
        scene_types[scene] = scene_types.get(scene, 0) + 1

        for lbl in s.get("label_values", []):
            observed_labels.add(lbl)

        if s["width"] > 0 and s["height"] > 0:
            dimensions.add(f"{s['width']}x{s['height']}")

    summary_data = {
        "dataset_name": "loveda",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples_discovered": len(validated_samples),
        "valid_samples_count": valid_count,
        "invalid_samples_count": invalid_count,
        "samples_by_split": splits,
        "samples_by_scene_type": scene_types,
        "observed_label_values": sorted(list(observed_labels)),
        "image_dimensions": sorted(list(dimensions)),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Generated dataset summary at {summary_path}")
    return summary_path


def prepare_loveda_pipeline() -> None:
    """
    Main execution pipeline for LoveDA preparation and inspection.
    Gracefully handles empty / un-downloaded dataset states without creating fake data.
    """
    project_root = get_project_root()
    config = load_config()

    seg_cfg = config.get("segmentation", {})
    raw_root_rel = seg_cfg.get("raw_root", "data/raw/loveda")
    manifest_rel = seg_cfg.get("manifest", "data/metadata/loveda_manifest.csv")

    raw_root = project_root / raw_root_rel
    manifest_path = project_root / manifest_rel
    summary_path = project_root / "data" / "metadata" / "dataset_summary.json"

    logger.info("=" * 65)
    logger.info("LoveDA Dataset Preparation & Inspection Pipeline (Phase 2A)")
    logger.info("=" * 65)
    logger.info(f"Target raw root directory: {raw_root}")

    # Check whether dataset directory exists and contains data
    exists, visible_items = check_raw_dataset_exists(raw_root)

    if not exists or len(visible_items) == 0:
        logger.warning(
            f"LoveDA raw dataset is not present in {raw_root_rel} (directory is empty or missing)."
        )
        logger.info(
            "To proceed with real data inspection, please follow the manual download instructions:"
        )
        logger.info("  1. Run `python scripts/download_data.py` for acquisition guidance.")
        logger.info("  2. Manually download LoveDA from official Zenodo / GitHub sources.")
        logger.info(f"  3. Extract archives directly into: {raw_root}")
        logger.info(
            "  4. Re-run `python scripts/prepare_data.py` after dataset extraction."
        )
        logger.info("Exiting cleanly without generating manifests or fake data.")
        return

    # Dataset is present: proceed with discovery and validation
    logger.info(f"Dataset root contains {len(visible_items)} visible items. Starting inspection...")
    inspection = inspect_raw_structure(raw_root)
    logger.info(f"Discovered {len(inspection['subdirectories'])} subdirectories and {inspection['total_png_files']} PNG files.")

    valid_label_ids = load_valid_label_ids()
    logger.info(f"Loaded valid label IDs: {sorted(list(valid_label_ids))}")

    discovered_samples = discover_loveda_samples(raw_root, project_root)
    if not discovered_samples:
        logger.error(
            f"No valid image/mask pairs could be identified under {raw_root}. "
            "Please verify that the extracted directory matches LoveDA structure."
        )
        return

    logger.info(f"Discovered {len(discovered_samples)} candidate samples. Running validation...")

    validated_samples: List[Dict[str, Any]] = []
    for s in discovered_samples:
        v = validate_sample(s, valid_label_ids)
        validated_samples.append(v)

    valid_count = sum(1 for v in validated_samples if v["valid"])
    logger.info(f"Validation complete: {valid_count}/{len(validated_samples)} samples are valid.")

    # Generate Manifest & Summary only from real validated data
    generate_manifest(validated_samples, manifest_path)
    generate_dataset_summary(validated_samples, summary_path)
    logger.info("LoveDA preparation pipeline completed successfully.")


if __name__ == "__main__":
    prepare_loveda_pipeline()
