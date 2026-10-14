#!/usr/bin/env python3


import argparse
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.preprocessing.ivp_techniques import IVPProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="IVP Techniques & Preprocessing Pipeline Demo")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to raw dataset folder containing images (default: dataset)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_ivp_output",
        help="Directory to save generated IVP stage outputs and visual grids"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of images from dataset directory to process in demo (default: 3)"
    )
    return parser.parse_args()


def save_stage_comparison_grid(results: dict, sample_name: str, save_path: Path):
    """
    Renders a 4x4 matrix (16 panels) showing representative IVP preprocessing outputs across stages.
    """
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle(f"IVP Multi-Stage Preprocessing Overview: {sample_name}", fontsize=18, fontweight="bold", y=0.98)

    panels = [
        ("1. Original Raw RGB", results["raw_rgb"], None),
        ("2. Percentile Stretched", results["stage1_radiometric"]["percentile_stretched"], None),
        ("3. CLAHE Enhanced", results["stage1_radiometric"]["clahe_enhanced"], None),
        ("4. Gamma Shadow Boost", results["stage1_radiometric"]["gamma_boost"], None),

        ("5. Gaussian Filtered", results["stage2_noise"]["gaussian_blur"], None),
        ("6. Bilateral Filtered", results["stage2_noise"]["bilateral_filter"], None),
        ("7. ExG Index (Colormap)", results["stage3_spectral"]["exg_index_colormap"], None),
        ("8. VARI Index (Colormap)", results["stage3_spectral"]["vari_index_colormap"], None),

        ("9. Sobel Gradient Mag", results["stage4_edge"]["sobel_magnitude"], "gray"),
        ("10. Laplacian of Gaussian", results["stage4_edge"]["laplacian_of_gaussian"], "gray"),
        ("11. Canny Edge Detector", results["stage4_edge"]["canny_edges"], "gray"),
        ("12. FFT Spectrum (2D)", results["stage5_frequency"]["fft_magnitude_spectrum"], None),

        ("13. FFT High-Pass Filter", results["stage5_frequency"]["fft_high_pass_filtered"], "gray"),
        ("14. Top-Hat Bright Feature", results["stage6_morphology"]["top_hat_bright_features"], "gray"),
        ("15. Adaptive Threshold", results["stage7_threshold"]["adaptive_gaussian_threshold"], "gray"),
        ("16. Distance Transform", results["stage7_threshold"]["distance_transform_map"], None),
    ]

    for ax, (title, img, cmap) in zip(axes.flat, panels):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_ivp_demo():
    args = parse_args()
    dataset_path = ROOT_DIR / args.dataset_dir
    output_path = ROOT_DIR / args.output_dir

    print("=" * 80)
    print("      SATELLITE & REMOTE SENSING IVP PREPROCESSING PIPELINE DEMO")
    print("=" * 80)
    print(f"[*] Dataset Directory: {dataset_path}")
    print(f"[*] Output Directory : {output_path}")

    if not dataset_path.exists():
        print(f"[!] Error: Dataset directory '{dataset_path}' does not exist!")
        sys.exit(1)

    # Discover PNG / JPG images
    image_files = sorted(list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpg")))
    if not image_files:
        print(f"[!] Error: No PNG or JPG images found in '{dataset_path}'")
        sys.exit(1)

    print(f"[*] Discovered {len(image_files)} raw satellite image(s) in dataset.")
    selected_files = image_files[: args.num_samples]
    print(f"[*] Processing top {len(selected_files)} sample(s) for demonstration...\n")

    os.makedirs(output_path, exist_ok=True)
    processor = IVPProcessor()

    start_total_time = time.time()

    for idx, img_file in enumerate(selected_files, start=1):
        sample_name = img_file.stem
        print("-" * 80)
        print(f"[{idx}/{len(selected_files)}] Processing Image: {img_file.name}")
        print("-" * 80)

        t0 = time.time()
        results = processor.process_image(img_file)
        elapsed = time.time() - t0

        metrics = results["metrics"]
        print(f"  ✓ Image Dimensions       : {metrics['image_dimensions']}")
        print(f"  ✓ Mean / Std Intensity   : {metrics['mean_intensity']} / {metrics['std_intensity']}")
        print(f"  ✓ Image Entropy (bits)   : {metrics['image_entropy']}")
        print(f"  ✓ Canny Edge Density     : {metrics['canny_edge_density'] * 100:.2f}%")
        print(f"  ✓ Spatial 512x512 Tiles  : {metrics['tile_count']} tiles generated")
        print(f"  ✓ Channel Mean RGB       : {results['stage8_tiling']['channel_mean']}")
        print(f"  ✓ Channel Std RGB        : {results['stage8_tiling']['channel_std']}")
        print(f"  ✓ Processing Time        : {elapsed * 1000:.2f} ms")

        # Save individual sample folder
        sample_out_dir = output_path / f"sample_{sample_name}"
        os.makedirs(sample_out_dir, exist_ok=True)

        # Save Grid Overview
        grid_save_path = output_path / f"ivp_overview_{sample_name}.png"
        save_stage_comparison_grid(results, sample_name, grid_save_path)
        print(f"  -> Saved 16-Panel Overview Grid : {grid_save_path.relative_to(ROOT_DIR)}")

        # Save key individual stage outputs
        plt.imsave(sample_out_dir / "01_clahe.png", results["stage1_radiometric"]["clahe_enhanced"])
        plt.imsave(sample_out_dir / "02_exg_index.png", results["stage3_spectral"]["exg_index_colormap"])
        plt.imsave(sample_out_dir / "03_vari_index.png", results["stage3_spectral"]["vari_index_colormap"])
        plt.imsave(sample_out_dir / "04_canny_edges.png", results["stage4_edge"]["canny_edges"], cmap="gray")
        plt.imsave(sample_out_dir / "05_fft_spectrum.png", results["stage5_frequency"]["fft_magnitude_spectrum"])
        plt.imsave(sample_out_dir / "06_adaptive_thresh.png", results["stage7_threshold"]["adaptive_gaussian_threshold"], cmap="gray")
        print(f"  -> Saved Stage Artifacts to    : {sample_out_dir.relative_to(ROOT_DIR)}/\n")

    total_elapsed = time.time() - start_total_time

    print("=" * 80)
    print("                           SUMMARY OF EXECUTION")
    print("=" * 80)
    print(f"Successfully processed {len(selected_files)} satellite images in {total_elapsed:.2f} seconds.")
    print(f"All generated IVP artifacts and visual overview grids have been saved to:")
    print(f" -> {output_path.resolve()}\n")
    print("IVP Stages Covered in Script:")
    print(" 1. Radiometric Calibration & Dynamic Range Adjustment (CLAHE, HE, Percentile Stretch, Gamma)")
    print(" 2. Spatial Noise Reduction & Edge-Preserving Smoothing (Gaussian, Median, Bilateral)")
    print(" 3. Spectral Indexing & Color Spaces (HSV, ExG Excess Green, VARI Index, NGRDI Index)")
    print(" 4. Edge & Gradient Operators (Sobel X/Y/Mag, Laplacian of Gaussian, Multi-threshold Canny)")
    print(" 5. Frequency Domain Analysis (2D FFT Log Spectrum, High-Pass & Low-Pass Spatial Filters)")
    print(" 6. Morphological Operations (Gradient, Top-Hat Bright, Black-Hat Shadow, Opening)")
    print(" 7. Adaptive Thresholding & Binarization (Otsu, Local Adaptive Gaussian, Distance Transform)")
    print(" 8. Tiling & Standardized Normalization (512x512 Patches, RGB Channel Scaling)")
    print("=" * 80)


if __name__ == "__main__":
    run_ivp_demo()
