"""
Dataset Download & Acquisition Guidance Script
"""

import os
import sys
from pathlib import Path
import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config() -> dict:
    config_path = get_project_root() / "configs" / "data.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def instructions_loveda(raw_loveda_dir: Path):
    print("=" * 70)
    print("LoveDA Dataset Acquisition Instructions")
    print("=" * 70)
    print("LoveDA is a multi-gigabyte high-resolution satellite imagery dataset.")
    print("Automated bulk downloads are disabled to ensure deliberate local management.")
    print("\nOfficial Sources:")
    print("  - GitHub: https://github.com/Junjue-Wang/LoveDA")
    print("  - Zenodo: https://zenodo.org/record/5706576")
    print("\nDownload & Extraction Instructions:")
    print(f"1. Download the official LoveDA archive(s) (e.g. Train.zip, Val.zip, Test.zip).")
    print(f"2. Extract the archives directly into:")
    print(f"     Target path: {raw_loveda_dir}")
    print("3. Ensure that the official split directories (e.g., Train, Val, Test) and")
    print("   scene folders (Urban, Rural) are preserved intact without modifying files.")
    print("4. After extraction, run `python scripts/prepare_data.py` to inspect and validate.")
    print("-" * 70)


def instructions_xview(raw_xview_dir: Path):
    print("=" * 70)
    print("xView Dataset Acquisition Instructions (Out of Scope for Phase 2A)")
    print("=" * 70)
    print("The xView dataset requires registration and acceptance of the EULA.")
    print("  1. Visit http://xviewdataset.org/ and register an account.")
    print("  2. Download the overhead imagery and GeoJSON labels.")
    print(f"  3. Place downloaded files in: {raw_xview_dir}")
    print("  4. Detailed xView inspection and class selection will occur in Phase 3.")
    print("-" * 70)


def inspect_and_report_loveda(raw_loveda_dir: Path):
    print("\n--- Inspecting LoveDA Raw Root ---")
    print(f"Target directory: {raw_loveda_dir}")

    # Ensure directory exists without creating internal subfolders
    raw_loveda_dir.mkdir(parents=True, exist_ok=True)

    # List top-level items excluding hidden files like .gitkeep or .DS_Store
    all_items = sorted(os.listdir(raw_loveda_dir))
    visible_items = [item for item in all_items if not item.startswith(".")]

    print(f"Top-level contents ({len(all_items)} total items, {len(visible_items)} visible):")
    if all_items:
        for item in all_items:
            item_path = raw_loveda_dir / item
            kind = "Directory" if item_path.is_dir() else "File"
            print(f"  - [{kind}] {item}")
    else:
        print("  (Directory is completely empty)")

    if not visible_items:
        print("\n[STATUS: NOT DOWNLOADED]")
        print("LoveDA raw dataset has not yet been downloaded into data/raw/loveda.")
    else:
        print(f"\n[STATUS: FILES PRESENT] Found {len(visible_items)} visible items in data/raw/loveda.")


def main():
    root = get_project_root()
    config = load_config()

    seg_raw_root = config.get("segmentation", {}).get("raw_root", "data/raw/loveda")
    det_raw_root = config.get("detection", {}).get("raw_root", "data/raw/xview")

    raw_loveda_dir = root / seg_raw_root
    raw_xview_dir = root / det_raw_root

    # Ensure raw directory roots exist (without internal assumed subfolders)
    raw_loveda_dir.mkdir(parents=True, exist_ok=True)
    raw_xview_dir.mkdir(parents=True, exist_ok=True)

    inspect_and_report_loveda(raw_loveda_dir)

    print("\n" + "=" * 70)
    print("MANUAL DATASET ACQUISITION GUIDANCE")
    print("=" * 70)
    instructions_loveda(raw_loveda_dir)
    print()
    instructions_xview(raw_xview_dir)


if __name__ == "__main__":
    main()
