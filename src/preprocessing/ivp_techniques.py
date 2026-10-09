"""
IVP (Image & Video Processing) Techniques and Pipeline for Satellite / Remote Sensing Images.

This module provides a comprehensive suite of classical and modern IVP techniques
specifically designed for satellite image preprocessing, feature enhancement,
noise reduction, spectral index calculation, frequency analysis, and binarization.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np
from PIL import Image


class IVPProcessor:
    """
    Comprehensive Image & Video Processing (IVP) Pipeline.
    Encapsulates 8 major preprocessing stages for remote sensing and satellite data.
    """

    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        self.target_size = target_size

    @staticmethod
    def load_image(image_path: str | Path) -> np.ndarray:
        """Load image into RGB format as uint8 NumPy array [H, W, 3]."""
        path_str = str(image_path)
        img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB image to 8-bit grayscale."""
        if len(img_rgb.shape) == 2:
            return img_rgb
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # -------------------------------------------------------------------------
    # STAGE 1: Radiometric Calibration & Dynamic Range Enhancement
    # -------------------------------------------------------------------------
    def stage1_radiometric_calibration(self, img_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Enhances dynamic range and corrects illumination non-uniformity.
        - Percentile Dynamic Range Stretching (0.5% - 99.5%)
        - Global Histogram Equalization
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Gamma Correction
        """
        # 1. Percentile Clipping & Contrast Stretch
        p_low, p_high = np.percentile(img_rgb, (0.5, 99.5))
        img_clipped = np.clip(img_rgb, p_low, p_high)
        img_stretched = ((img_clipped - p_low) / (p_high - p_low + 1e-5) * 255.0).astype(np.uint8)

        # 2. CLAHE on Luminance (LAB color space)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        # 3. Global Histogram Equalization on Grayscale
        gray = self.to_grayscale(img_rgb)
        img_he = cv2.equalizeHist(gray)

        # 4. Gamma Correction (Shadow boost: gamma=0.7, Highlight compression: gamma=1.5)
        inv_gamma_boost = 1.0 / 0.7
        table_boost = np.array([((i / 255.0) ** inv_gamma_boost) * 255 for i in range(256)]).astype("uint8")
        img_gamma_boost = cv2.LUT(img_rgb, table_boost)

        return {
            "percentile_stretched": img_stretched,
            "clahe_enhanced": img_clahe,
            "global_hist_eq": img_he,
            "gamma_boost": img_gamma_boost
        }

    # -------------------------------------------------------------------------
    # STAGE 2: Spatial Noise Reduction & Smoothing Filters
    # -------------------------------------------------------------------------
    def stage2_noise_reduction(self, img_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Applies spatial smoothing to reduce high-frequency atmospheric & sensor noise.
        - Gaussian Filtering (5x5, sigma=1.0)
        - Median Filtering (5x5)
        - Bilateral Filtering (Edge-preserving smoothing)
        """
        gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 1.0)
        median = cv2.medianBlur(img_rgb, 5)
        bilateral = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

        return {
            "gaussian_blur": gaussian,
            "median_blur": median,
            "bilateral_filter": bilateral
        }

    # -------------------------------------------------------------------------
    # STAGE 3: Spectral Indexing & Color Space Decompositions
    # -------------------------------------------------------------------------
    def stage3_spectral_indexing(self, img_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts spectral indicators & color spaces to highlight vegetation/water/built land.
        - HSV (Hue, Saturation, Value)
        - ExG (Excess Green Index): 2G - R - B
        - VARI (Visible Atmospherically Resistant Index): (G - R) / (G + R - B)
        - NGRDI (Normalized Green-Red Difference Index): (G - R) / (G + R)
        """
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Floating point conversion normalized [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

        # Excess Green Index (ExG)
        exg = 2.0 * g - r - b
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        exg_colormap = cv2.applyColorMap(exg_norm, cv2.COLORMAP_SUMMER)

        # Visible Atmospherically Resistant Index (VARI)
        denom_vari = g + r - b + 1e-5
        vari = (g - r) / denom_vari
        vari_norm = cv2.normalize(np.clip(vari, -1, 1), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vari_colormap = cv2.applyColorMap(vari_norm, cv2.COLORMAP_VIRIDIS)

        # Normalized Green-Red Difference Index (NGRDI)
        denom_ngrdi = g + r + 1e-5
        ngrdi = (g - r) / denom_ngrdi
        ngrdi_norm = cv2.normalize(np.clip(ngrdi, -1, 1), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return {
            "hsv_color_space": hsv,
            "exg_index_colormap": cv2.cvtColor(exg_colormap, cv2.COLOR_BGR2RGB),
            "vari_index_colormap": cv2.cvtColor(vari_colormap, cv2.COLOR_BGR2RGB),
            "ngrdi_index_gray": ngrdi_norm
        }

    # -------------------------------------------------------------------------
    # STAGE 4: Edge Extraction & Gradient Operators
    # -------------------------------------------------------------------------
    def stage4_edge_and_gradient(self, img_gray: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts directional gradients and structural boundaries.
        - Sobel X, Y, Magnitude
        - Laplacian of Gaussian (LoG)
        - Canny Edge Detection
        """
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mag_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Laplacian of Gaussian (LoG)
        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        log_norm = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Multi-threshold Canny
        canny = cv2.Canny(img_gray, threshold1=50, threshold2=150)

        return {
            "sobel_magnitude": sobel_mag_norm,
            "laplacian_of_gaussian": log_norm,
            "canny_edges": canny
        }

    # -------------------------------------------------------------------------
    # STAGE 5: Frequency Domain Analysis (2D Fourier Transform)
    # -------------------------------------------------------------------------
    def stage5_frequency_domain(self, img_gray: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Performs 2D Fast Fourier Transform (FFT) for spatial frequency decomposition.
        - Magnitude Spectrum (Log scale)
        - High-Pass Filtered Image (Edge/Detail sharpening)
        - Low-Pass Filtered Image (Global background smoothing)
        """
        f = np.fft.fft2(img_gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
        mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag_colormap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

        # Create Low-Pass Filter Mask
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        r = 40
        mask_lpf = np.zeros((rows, cols), np.float32)
        cv2.circle(mask_lpf, (ccol, crow), r, 1, -1)

        # Create High-Pass Filter Mask
        mask_hpf = 1.0 - mask_lpf

        # Apply LPF
        fshift_lpf = fshift * mask_lpf
        f_lpf = np.fft.ifftshift(fshift_lpf)
        img_lpf = np.abs(np.fft.ifft2(f_lpf))
        img_lpf_norm = cv2.normalize(img_lpf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply HPF
        fshift_hpf = fshift * mask_hpf
        f_hpf = np.fft.ifftshift(fshift_hpf)
        img_hpf = np.abs(np.fft.ifft2(f_hpf))
        img_hpf_norm = cv2.normalize(img_hpf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return {
            "fft_magnitude_spectrum": cv2.cvtColor(mag_colormap, cv2.COLOR_BGR2RGB),
            "fft_low_pass_filtered": img_lpf_norm,
            "fft_high_pass_filtered": img_hpf_norm
        }

    # -------------------------------------------------------------------------
    # STAGE 6: Morphological Operations & Structural Filtering
    # -------------------------------------------------------------------------
    def stage6_morphological_operations(self, img_gray: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Applies mathematical morphology for structural feature isolation.
        - Morphological Gradient (Dilation - Erosion)
        - Top-Hat Transformation (Isolates bright targets like building roofs)
        - Black-Hat Transformation (Isolates dark targets like shadows/water bodies)
        - Opening (Erosion followed by Dilation)
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
        top_hat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
        opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

        return {
            "morphological_gradient": gradient,
            "top_hat_bright_features": top_hat,
            "black_hat_dark_features": black_hat,
            "morphological_opening": opening
        }

    # -------------------------------------------------------------------------
    # STAGE 7: Adaptive Thresholding & Binarization
    # -------------------------------------------------------------------------
    def stage7_thresholding_and_binarization(self, img_gray: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segments foreground targets from background.
        - Otsu's Automatic Global Binarization
        - Adaptive Gaussian Local Thresholding
        - Distance Transform Map
        """
        # Otsu's Binarization
        _, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive Gaussian Thresholding
        adaptive_gauss = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Distance Transform on Otsu Mask
        dist_transform = cv2.distanceTransform(otsu, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dist_colormap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_VIRIDIS)

        return {
            "otsu_threshold": otsu,
            "adaptive_gaussian_threshold": adaptive_gauss,
            "distance_transform_map": cv2.cvtColor(dist_colormap, cv2.COLOR_BGR2RGB)
        }

    # -------------------------------------------------------------------------
    # STAGE 8: Tiling & Standardized Preprocessing
    # -------------------------------------------------------------------------
    def stage8_tiling_and_normalization(
        self, img_rgb: np.ndarray, tile_size: Tuple[int, int] = (512, 512)
    ) -> Dict[str, Any]:
        """
        Tiles large satellite image into patches and calculates tensor normalization.
        """
        h, w, c = img_rgb.shape
        th, tw = tile_size
        tiles = []

        for y in range(0, h - th + 1, th):
            for x in range(0, w - tw + 1, tw):
                tile = img_rgb[y : y + th, x : x + tw]
                tiles.append({
                    "bbox": [x, y, x + tw, y + th],
                    "tile": tile
                })

        # ImageNet style normalization parameters
        float_img = img_rgb.astype(np.float32) / 255.0
        mean = float_img.mean(axis=(0, 1)).tolist()
        std = float_img.std(axis=(0, 1)).tolist()

        return {
            "total_tiles": len(tiles),
            "tile_size": tile_size,
            "tiles": tiles,
            "channel_mean": [round(m, 4) for m in mean],
            "channel_std": [round(s, 4) for s in std]
        }

    # -------------------------------------------------------------------------
    # Master Pipeline Execution
    # -------------------------------------------------------------------------
    def process_image(self, image_path: str | Path) -> Dict[str, Any]:
        """
        Executes all 8 IVP stages on the specified image path and returns
        processed images, arrays, and quantitative metrics.
        """
        img_rgb = self.load_image(image_path)
        img_gray = self.to_grayscale(img_rgb)

        s1 = self.stage1_radiometric_calibration(img_rgb)
        s2 = self.stage2_noise_reduction(img_rgb)
        s3 = self.stage3_spectral_indexing(img_rgb)
        s4 = self.stage4_edge_and_gradient(img_gray)
        s5 = self.stage5_frequency_domain(img_gray)
        s6 = self.stage6_morphological_operations(img_gray)
        s7 = self.stage7_thresholding_and_binarization(img_gray)
        s8 = self.stage8_tiling_and_normalization(img_rgb)

        # Calculate quantitative image metrics
        edge_density = float(np.mean(s4["canny_edges"] > 0))
        mean_intensity = float(np.mean(img_gray))
        std_intensity = float(np.std(img_gray))

        # Image Entropy (Information Content)
        hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        metrics = {
            "image_dimensions": img_rgb.shape,
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "image_entropy": round(entropy, 4),
            "canny_edge_density": round(edge_density, 4),
            "tile_count": s8["total_tiles"]
        }

        return {
            "raw_rgb": img_rgb,
            "raw_gray": img_gray,
            "metrics": metrics,
            "stage1_radiometric": s1,
            "stage2_noise": s2,
            "stage3_spectral": s3,
            "stage4_edge": s4,
            "stage5_frequency": s5,
            "stage6_morphology": s6,
            "stage7_threshold": s7,
            "stage8_tiling": s8
        }
