"""API routes integrating 8-stage IVP Preprocessing, ML analysis, and executive disaster mitigation report."""
from pathlib import Path
import io
import sys
import base64
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2

from fastapi import APIRouter, File, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.ivp_techniques import IVPProcessor
from src.explanation.generator import ExplanationGenerator
from src.feature_extraction.extractor import FeatureExtractor
from src.risk_engine.engine import RiskEngine

try:
    from backend.schemas.models import (
        RiskResponse,
        IVPStageOutput,
        IVPMetrics,
        Hazard,
        DisasterMitigationReport,
    )
except ModuleNotFoundError:
    from schemas.models import (
        RiskResponse,
        IVPStageOutput,
        IVPMetrics,
        Hazard,
        DisasterMitigationReport,
    )

router = APIRouter()
processor = IVPProcessor()
extractor = FeatureExtractor()
engine = RiskEngine()
explainer = ExplanationGenerator()


def numpy_to_base64(img_arr: np.ndarray, format: str = "PNG") -> str:
    """Converts numpy image array to base64 data URI string."""
    if len(img_arr.shape) == 2:
        pil_img = Image.fromarray(img_arr, mode="L")
    elif img_arr.shape[2] == 3:
        pil_img = Image.fromarray(img_arr, mode="RGB")
    elif img_arr.shape[2] == 4:
        pil_img = Image.fromarray(img_arr, mode="RGBA")
    else:
        pil_img = Image.fromarray(img_arr)

    buffered = io.BytesIO()
    pil_img.save(buffered, format=format, quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def generate_segmentation_overlay(img_rgb: np.ndarray) -> np.ndarray:
    """Creates a colorized semantic segmentation mask overlay for LoveDA classes using PyTorch."""
    try:
        from src.segmentation.model import predict_segmentation_mask
        mask, _ = predict_segmentation_mask(img_rgb)
    except Exception:
        mask = None

    if mask is not None:
        overlay = np.zeros_like(img_rgb)
        overlay[mask == 1] = [239, 68, 68]      # Building: Red
        overlay[mask == 2] = [245, 158, 11]     # Road: Yellow
        overlay[mask == 3] = [14, 165, 233]    # Water: Blue
        overlay[mask == 4] = [168, 85, 247]    # Bare land: Purple
        overlay[(mask == 5) | (mask == 6)] = [34, 197, 94]  # Vegetation: Green
        return cv2.addWeighted(img_rgb, 0.45, overlay, 0.55, 0)

    # Fallback to HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    water = (((hue >= 85) & (hue <= 125) & (saturation >= 45) & (value >= 35)) |
             ((hue >= 75) & (hue < 85) & (saturation >= 55) & (value >= 35)))
    vegetation = (hue >= 35) & (hue <= 95) & (saturation >= 45) & (value >= 35)
    bare_land = (hue >= 5) & (hue <= 34) & (saturation >= 35) & (value >= 65)
    neutral = saturation < 42
    road = neutral & (value >= 105) & (value <= 220)
    building = neutral & (value > 220)

    overlay = np.zeros_like(img_rgb)
    overlay[building] = [239, 68, 68]
    overlay[road] = [245, 158, 11]
    overlay[water] = [14, 165, 233]
    overlay[vegetation] = [34, 197, 94]
    overlay[bare_land] = [168, 85, 247]
    return cv2.addWeighted(img_rgb, 0.45, overlay, 0.55, 0)


def generate_detection_overlay(img_rgb: np.ndarray) -> np.ndarray:
    """Generates overhead object detection bounding box overlays using YOLOv8."""
    try:
        from src.detection.model import predict_objects
        dets, overlay_img = predict_objects(img_rgb)
        if dets:
            return overlay_img
    except Exception:
        pass

    # Fallback contour detection
    img_out = img_rgb.copy()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 <= area <= 4000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.3 <= aspect_ratio <= 3.0:
                cv2.rectangle(img_out, (x, y), (x + w, y + h), (6, 182, 212), 2)
                label = f"Obj #{count+1}"
                cv2.putText(img_out, label, (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                count += 1
                if count >= 25:
                    break
    return img_out


def generate_earthvl_risk_heatmap(img_rgb: np.ndarray, ivp_res: dict, risk_score: int) -> np.ndarray:
    """Generates a colorized EarthVL Multi-Hazard Risk Exposure Map."""
    h, w = img_rgb.shape[:2]
    try:
        dist_map = ivp_res["stage7_threshold"]["distance_transform_map"]
        canny_edges = ivp_res["stage4_edge"]["canny_edges"]
    except KeyError:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150)
        dist_map = cv2.distanceTransform((canny_edges == 0).astype(np.uint8), cv2.DIST_L2, 5)

    if len(dist_map.shape) == 3:
        dist_map = cv2.cvtColor(dist_map, cv2.COLOR_RGB2GRAY)
    if len(canny_edges.shape) == 3:
        canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_RGB2GRAY)

    if dist_map.shape[:2] != (h, w):
        dist_map = cv2.resize(dist_map, (w, h), interpolation=cv2.INTER_AREA)
    if canny_edges.shape[:2] != (h, w):
        canny_edges = cv2.resize(canny_edges, (w, h), interpolation=cv2.INTER_AREA)

    # Normalize metrics to [0, 255] risk map
    blur_edges = cv2.GaussianBlur(canny_edges.astype(np.float32), (15, 15), 0)
    norm_dist = cv2.normalize(dist_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    
    combined_risk = (blur_edges * 0.6) + (norm_dist * 0.4 * (risk_score / 100.0))
    combined_risk = np.clip(combined_risk, 0, 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(combined_risk, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_rgb, 0.40, heatmap_rgb, 0.60, 0)
    return blended



def build_ivp_stage_outputs(ivp_res: dict) -> list:
    """Formats all 16 key preprocessed outputs with domain explanations."""
    stages = [
        # Stage 1: Radiometric
        {
            "id": "s1_stretched",
            "title": "Percentile Dynamic Contrast Stretch",
            "stage_name": "Stage 1: Radiometric Calibration",
            "category": "Radiometric",
            "image_base64": numpy_to_base64(ivp_res["stage1_radiometric"]["percentile_stretched"]),
            "summary": "Stretches pixel intensities between 0.5th and 99.5th percentiles.",
            "technical_detail": "Removes sensor atmospheric haze and sensor clipping by mapping raw values to full [0, 255] dynamic range.",
            "disaster_relevance": "Accurately reveals shadow boundaries and hidden ground structures in satellite patch."
        },
        {
            "id": "s1_clahe",
            "title": "CLAHE Contrast Enhancement",
            "stage_name": "Stage 1: Radiometric Calibration",
            "category": "Radiometric",
            "image_base64": numpy_to_base64(ivp_res["stage1_radiometric"]["clahe_enhanced"]),
            "summary": "Contrast Limited Adaptive Histogram Equalization on L-channel.",
            "technical_detail": "Divides image into 8x8 contextual tiles with 3.0 clip limit to prevent over-amplification of noise.",
            "disaster_relevance": "Delineates subtle terrain elevation boundaries and structural roof contours."
        },
        {
            "id": "s1_gamma",
            "title": "Gamma Shadow Boost (gamma=0.7)",
            "stage_name": "Stage 1: Radiometric Calibration",
            "category": "Radiometric",
            "image_base64": numpy_to_base64(ivp_res["stage1_radiometric"]["gamma_boost"]),
            "summary": "Non-linear tone curve boost for dark shadowed regions.",
            "technical_detail": "Applies lookup table LUT transformation: I_out = 255 * (I_in / 255)^(1/0.7).",
            "disaster_relevance": "Uncovers building foundations and narrow alleyways obscured under cloud or tree shadows."
        },

        # Stage 2: Noise Reduction
        {
            "id": "s2_gaussian",
            "title": "Spatial Gaussian Noise Smoothing",
            "stage_name": "Stage 2: Noise Reduction",
            "category": "Noise Filtering",
            "image_base64": numpy_to_base64(ivp_res["stage2_noise"]["gaussian_blur"]),
            "summary": "5x5 Gaussian kernel spatial smoothing with sigma=1.0.",
            "technical_detail": "Convolves 2D Gaussian kernel to suppress high-frequency speckle noise from satellite sensor optics.",
            "disaster_relevance": "Stabilizes pixel classification and reduces false positive detection on noisy terrain."
        },
        {
            "id": "s2_bilateral",
            "title": "Edge-Preserving Bilateral Filter",
            "stage_name": "Stage 2: Noise Reduction",
            "category": "Noise Filtering",
            "image_base64": numpy_to_base64(ivp_res["stage2_noise"]["bilateral_filter"]),
            "summary": "D=9 bilateral filter preserving sharp structural edges.",
            "technical_detail": "Combines spatial Gaussian distance and radiometric intensity variance (sigmaColor=75, sigmaSpace=75).",
            "disaster_relevance": "Smooths homogeneous crop fields while keeping building walls and road edges crisp."
        },

        # Stage 3: Spectral Indexing
        {
            "id": "s3_exg",
            "title": "Excess Green Index (ExG Colormap)",
            "stage_name": "Stage 3: Spectral Indexing",
            "category": "Spectral",
            "image_base64": numpy_to_base64(ivp_res["stage3_spectral"]["exg_index_colormap"]),
            "summary": "Spectral index 2G - R - B rendered in Summer colormap.",
            "technical_detail": "Isolates vegetative green reflectance from urban concrete, asphalt, and water bodies.",
            "disaster_relevance": "Directly measures vegetation canopy loss and deforestation / degradation zones."
        },
        {
            "id": "s3_vari",
            "title": "VARI Vegetation Index (Viridis)",
            "stage_name": "Stage 3: Spectral Indexing",
            "category": "Spectral",
            "image_base64": numpy_to_base64(ivp_res["stage3_spectral"]["vari_index_colormap"]),
            "summary": "Visible Atmospherically Resistant Index (G - R)/(G + R - B).",
            "technical_detail": "Normalizes atmospheric scattering effects using visible green and red channels.",
            "disaster_relevance": "Provides robust vegetation health estimation even under light cloud cover or haze."
        },

        # Stage 4: Edge & Gradient
        {
            "id": "s4_sobel",
            "title": "Sobel Gradient Magnitude",
            "stage_name": "Stage 4: Edge Extraction",
            "category": "Edge & Gradient",
            "image_base64": numpy_to_base64(ivp_res["stage4_edge"]["sobel_magnitude"]),
            "summary": "First-derivative spatial gradient magnitude sqrt(Gx^2 + Gy^2).",
            "technical_detail": "Convolves 3x3 horizontal and vertical Sobel kernel masks to identify gradient transitions.",
            "disaster_relevance": "Highlights road networks, canals, and linear infrastructure boundaries."
        },
        {
            "id": "s4_log",
            "title": "Laplacian of Gaussian (LoG)",
            "stage_name": "Stage 4: Edge Extraction",
            "category": "Edge & Gradient",
            "image_base64": numpy_to_base64(ivp_res["stage4_edge"]["laplacian_of_gaussian"]),
            "summary": "Second-derivative Laplacian on Gaussian smoothed image.",
            "technical_detail": "Detects zero-crossings corresponding to sharp intensity boundaries and corners.",
            "disaster_relevance": "Pinpoints roof corners, building edges, and bridge structures."
        },
        {
            "id": "s4_canny",
            "title": "Multi-Threshold Canny Edge Detector",
            "stage_name": "Stage 4: Edge Extraction",
            "category": "Edge & Gradient",
            "image_base64": numpy_to_base64(ivp_res["stage4_edge"]["canny_edges"]),
            "summary": "Hysteresis thresholding Canny edge map (50 / 150 thresholds).",
            "technical_detail": "Suppresses non-maximum responses and traces connected edge contours.",
            "disaster_relevance": "Quantifies structural edge density used in urban exposure scoring."
        },

        # Stage 5: Frequency Domain
        {
            "id": "s5_fft_mag",
            "title": "2D Fast Fourier Transform Spectrum",
            "stage_name": "Stage 5: Frequency Domain",
            "category": "Frequency Domain",
            "image_base64": numpy_to_base64(ivp_res["stage5_frequency"]["fft_magnitude_spectrum"]),
            "summary": "Log-scaled 2D FFT magnitude spectrum in Jet colormap.",
            "technical_detail": "Decomposes spatial image into 2D frequency spectrum log(1 + |F(u,v)|).",
            "disaster_relevance": "Identifies periodic agricultural layouts, urban street grids, and directional patterns."
        },
        {
            "id": "s5_hpf",
            "title": "FFT High-Pass Spatial Filter",
            "stage_name": "Stage 5: Frequency Domain",
            "category": "Frequency Domain",
            "image_base64": numpy_to_base64(ivp_res["stage5_frequency"]["fft_high_pass_filtered"]),
            "summary": "High-pass spatial frequency filtering in Fourier domain.",
            "technical_detail": "Attenuates low-frequency background luminance, retaining fine structural detail.",
            "disaster_relevance": "Sharpens small infrastructure objects and fine drainage ditch lines."
        },

        # Stage 6: Morphology
        {
            "id": "s6_tophat",
            "title": "Top-Hat Transformation (Bright Features)",
            "stage_name": "Stage 6: Morphological Operations",
            "category": "Morphology",
            "image_base64": numpy_to_base64(ivp_res["stage6_morphology"]["top_hat_bright_features"]),
            "summary": "Top-hat operation (Image - Opening) with 5x5 rectangular element.",
            "technical_detail": "Isolates elements brighter than their surrounding local background.",
            "disaster_relevance": "Isolates bright building roofs, industrial structures, and metallic objects."
        },
        {
            "id": "s6_blackhat",
            "title": "Black-Hat Transformation (Dark Features)",
            "stage_name": "Stage 6: Morphological Operations",
            "category": "Morphology",
            "image_base64": numpy_to_base64(ivp_res["stage6_morphology"]["black_hat_dark_features"]),
            "summary": "Black-hat operation (Closing - Image) with 5x5 rectangular element.",
            "technical_detail": "Isolates dark features smaller than structuring element.",
            "disaster_relevance": "Highlights water ponds, shadow trenches, and deep soil depressions."
        },

        # Stage 7: Thresholding & Binarization
        {
            "id": "s7_adaptive",
            "title": "Adaptive Gaussian Binarization",
            "stage_name": "Stage 7: Adaptive Thresholding",
            "category": "Thresholding",
            "image_base64": numpy_to_base64(ivp_res["stage7_threshold"]["adaptive_gaussian_threshold"]),
            "summary": "Local Gaussian window adaptive binarization (11x11 window, C=2).",
            "technical_detail": "Calculates local threshold per pixel based on weighted mean of neighborhood minus constant C.",
            "disaster_relevance": "Segments built structures from terrain regardless of non-uniform illumination."
        },
        {
            "id": "s7_dist",
            "title": "Euclidean Distance Transform Map",
            "stage_name": "Stage 7: Adaptive Thresholding",
            "category": "Thresholding",
            "image_base64": numpy_to_base64(ivp_res["stage7_threshold"]["distance_transform_map"]),
            "summary": "Euclidean distance transform on binarized mask (Viridis).",
            "technical_detail": "Calculates distance of each foreground pixel to nearest background boundary.",
            "disaster_relevance": "Measures core structural thickness and spatial proximity vectors for flood exposure."
        },
    ]

    return [IVPStageOutput(**s) for s in stages]


def process_image_pipeline(pil_image: Image.Image, filename: str, mode: str, note: str) -> RiskResponse:
    """Master pipeline executing IVP, feature extraction, risk engine, and mitigation report."""
    # Convert PIL to temporary file or numpy array
    rgb_arr = np.array(pil_image.convert("RGB"))

    # Save temp file for IVP processor
    temp_path = ROOT / "data" / "temp_upload.png"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(temp_path, format="PNG")

    # 1. Execute IVP processor 8 stages
    ivp_results = processor.process_image(temp_path)

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    # 2. Extract features & proxies
    features = extractor.extract_features_from_image(pil_image)
    confidence = features.pop("model_confidence", 0.75)

    # 3. Calculate Risk Engine results
    risk_result = engine.calculate_risk(features, confidence=confidence)

    # 4. Generate natural language explanation & mitigation report
    explanation_str = explainer.generate_explanation(risk_result)
    mitigation_dict = explainer.generate_mitigation_report(risk_result, features)

    # 5. Format IVP metrics & stages
    m = ivp_results["metrics"]
    s8 = ivp_results["stage8_tiling"]
    ivp_metrics = IVPMetrics(
        image_dimensions=list(m["image_dimensions"]),
        mean_intensity=m["mean_intensity"],
        std_intensity=m["std_intensity"],
        image_entropy=m["image_entropy"],
        canny_edge_density=m["canny_edge_density"],
        tile_count=m["tile_count"],
        channel_mean=s8["channel_mean"],
        channel_std=s8["channel_std"],
    )

    ivp_stages = build_ivp_stage_outputs(ivp_results)

    # 6. Generate overlays
    orig_b64 = numpy_to_base64(rgb_arr)
    seg_b64 = numpy_to_base64(generate_segmentation_overlay(rgb_arr))
    det_b64 = numpy_to_base64(generate_detection_overlay(rgb_arr))
    heatmap_b64 = numpy_to_base64(generate_earthvl_risk_heatmap(rgb_arr, ivp_results, risk_result["overall_risk"]))

    # Format hazards
    hazards_list = []
    for h in risk_result["hazards"]:
        s = h["score"]
        sev = "CRITICAL" if s >= 80 else "HIGH" if s >= 60 else "MODERATE" if s >= 40 else "LOW"
        hazards_list.append(
            Hazard(
                type=h["type"],
                score=h["score"],
                confidence=h["confidence"],
                severity=sev,
                evidence=h["evidence"],
                recommendations=h["recommendations"],
            )
        )

    mitigation_report = DisasterMitigationReport(**mitigation_dict)

    return RiskResponse(
        location=filename,
        overall_risk=risk_result["overall_risk"],
        risk_level=risk_result["risk_level"],
        hazards=hazards_list,
        explanation=explanation_str,
        features={k: round(float(v), 2) for k, v in features.items()},
        ivp_metrics=ivp_metrics,
        ivp_stages=ivp_stages,
        original_image_base64=orig_b64,
        segmentation_mask_base64=seg_b64,
        detection_overlay_base64=det_b64,
        earthvl_risk_heatmap_base64=heatmap_b64,
        mitigation_report=mitigation_report,
        analysis_mode=mode,
        data_note=note,
    )


@router.post("/analyze", response_model=RiskResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Executes full 8-stage IVP preprocessing, feature extraction, ML overlays, and disaster report."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a PNG, JPEG, or TIFF satellite image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=415, detail="The file could not be read as an image.") from exc

    return process_image_pipeline(
        image,
        file.filename or "Uploaded Satellite Patch",
        "Full IVP + AI Pipeline Baseline",
        "Multi-stage IVP transformations, semantic land-cover proxies, and multi-hazard fusion calculated successfully.",
    )


@router.get("/demo", response_model=RiskResponse)
def demo_assessment():
    """Runs full pipeline on curated sample satellite image dataset/2522.png."""
    sample_file = ROOT / "dataset" / "2522.png"
    if not sample_file.exists():
        # Fallback to any image in dataset
        samples = list((ROOT / "dataset").glob("*.png"))
        if samples:
            sample_file = samples[0]

    if sample_file.exists():
        image = Image.open(sample_file).convert("RGB")
        return process_image_pipeline(
            image,
            f"Demo Case: {sample_file.name} (Peri-Urban Zone)",
            "Curated Demo Pipeline Scenario",
            "This curated scenario demonstrates all 16 IVP preprocessing stages, spectral indices, and disaster mitigation report.",
        )

    # Fallback synthetic image if no dataset folder exists
    synthetic_arr = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(synthetic_arr)
    return process_image_pipeline(
        image,
        "Synthetic Demo Scenario",
        "Synthetic Demo Pipeline",
        "Demonstration scenario rendered using baseline synthetic remote-sensing patch.",
    )


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "ivp-risk-assessment-demo", "ivp_processor_ready": True}
