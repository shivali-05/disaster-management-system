"""
Hugging Face EarthVL Dataset Downloader & Structure Initializer
Dataset Repository: Kingdrone-Junjue/EarthVLSet
Supports: EarthVL-GLOBAL and EarthVQA subsets across train, val, and test splits.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_REPO = "Kingdrone-Junjue/EarthVLSet"
SUBSETS = ["EarthVL-GLOBAL", "EarthVQA"]
SPLITS = ["train", "val", "test"]


def setup_folder_structure(base_dir: Path) -> dict:
    """Creates the target 3-fold split directory hierarchy for EarthVL dataset."""
    print(f"Setting up EarthVL folder structure in: {base_dir}")
    created_paths = {}

    for split in SPLITS:
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        created_paths[split] = []

        for subset in SUBSETS:
            subset_dir = split_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            # Create a placeholder README to preserve layout in git / filesystem
            readme_file = subset_dir / "README.md"
            if not readme_file.exists():
                readme_file.write_text(
                    f"# EarthVL {subset} [{split.upper()} SPLIT]\n"
                    f"Directory reserved for Hugging Face `{DATASET_REPO}` {subset} {split} data.\n",
                    encoding="utf-8",
                )
            created_paths[split].append(subset_dir)

    print("[SUCCESS] Folder structure initialized:")
    for split, paths in created_paths.items():
        print(f"  - {split}/")
        for p in paths:
            print(f"      ├── {p.name}")

    return created_paths


def print_dataset_info():
    """Prints paper and dataset information for Kingdrone-Junjue/EarthVLSet."""
    print("=" * 80)
    print("HUGGINGFACE DATASET METADATA: Kingdrone-Junjue/EarthVLSet")
    print("=" * 80)
    print("Title: EarthVL: A Progressive Earth Vision-Language Understanding and Generation Framework")
    print("Authors: Junjue Wang, Yanfei Zhong, Zihang Chen, Zhuo Zheng, Ailong Ma, Liangpei Zhang")
    print("Paper: arXiv:2601.02783")
    print("Repository: https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet")
    print("Modalities: Remote-Sensing Imagery, Geospatial Patches, Vision-Language QA")
    print("Total Size: ~18.1 GB")
    print("Subsets: EarthVL-GLOBAL (Global Land-Cover Segmentation) & EarthVQA (Visual QA)")
    print("-" * 80)


def download_from_huggingface(target_dir: Path, max_retries: int = 20, retry_delay: int = 15):
    """Downloads EarthVL dataset files from Hugging Face hub with resumption and verification."""
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("[ERROR] `huggingface_hub` is not installed. Install via `pip install huggingface-hub`.")
        sys.exit(1)

    print(f"\n[DOWNLOADING] Resuming/Fetching `{DATASET_REPO}` into `{target_dir}`...")
    api = HfApi()
    
    for attempt in range(1, max_retries + 1):
        print(f"\n--- Attempt {attempt}/{max_retries} ---")
        is_rate_limited = False
        try:
            downloaded_path = snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                max_workers=4,
            )
        except Exception as exc:
            print(f"[WARNING] Download error on attempt {attempt}: {exc}")
            if "429" in str(exc):
                is_rate_limited = True

        # Verify completeness against Hugging Face repository file list
        try:
            remote_files = set(api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset"))
            local_existing = set()
            for p in target_dir.rglob("*"):
                if p.is_file():
                    rel_p = p.relative_to(target_dir).as_posix()
                    local_existing.add(rel_p)

            missing_files = remote_files - local_existing
            # Ignore git/metadata system files if missing
            missing_files = {f for f in missing_files if not f.startswith(".")}

            if not missing_files:
                print(f"\n[SUCCESS] Dataset fully verified and complete! ({len(remote_files)} total files on disk)")
                return target_dir
            else:
                print(f"[INCOMPLETE] {len(local_existing)}/{len(remote_files)} files present. {len(missing_files)} files remaining.")
        except Exception as exc:
            print(f"[WARNING] Could not list repo files to verify completeness (likely 429 rate limit): {exc}")
            if "429" in str(exc):
                is_rate_limited = True

        if attempt < max_retries:
            wait_time = 60 if is_rate_limited else retry_delay
            if is_rate_limited:
                print(f"[RATE LIMITED] Hugging Face 429 quota reached. Pausing for {wait_time} seconds before resuming...")
            else:
                print(f"Retrying download in {wait_time} seconds...")
            import time
            time.sleep(wait_time)
        else:
            print(f"[ERROR] Failed to complete dataset download after {max_retries} attempts.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="EarthVL Dataset Downloader & Structure Setup")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "dataset" / "earthvl"),
        help="Path where dataset structure should be created.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Execute actual download of 18.1 GB files from Hugging Face.",
    )
    args = parser.parse_args()

    target_dir = Path(args.output_dir).resolve()
    print_dataset_info()

    # Step 1: Create train, val, test folder structure
    setup_folder_structure(target_dir)

    # Step 2: Handle download option
    if args.download:
        download_from_huggingface(target_dir)
    else:
        print("\n[NOTE] Dry-run complete. Folder structure is ready.")
        print(f"To download full 18.1 GB Hugging Face data, run:")
        print(f"  python scripts/download_earthvl_dataset.py --download")


if __name__ == "__main__":
    main()
