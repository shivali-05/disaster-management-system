# TerraRisk: AI-Based Multi-Hazard Risk and Infrastructure Assessment using Satellite Imagery
## Comprehensive Technical Project Specification & Architecture Reference Manual

---

## 1. Executive Project Overview

**TerraRisk** is an end-to-end, production-ready remote sensing decision-support system and AI engine designed for satellite imagery analysis, multi-hazard risk assessment, infrastructure exposure mapping, and automated disaster mitigation report generation.

Existing remote sensing solutions typically operate in isolation—either performing standalone semantic segmentation, isolated object detection, or basic spectral indexing. **TerraRisk** bridges this critical research gap by unifying these capabilities into a continuous, interpretable pipeline:

$$\text{Raw Satellite Patch} \longrightarrow \text{IVP 8-Stage Matrix} \longrightarrow \text{Deep Learning Inference} \longrightarrow \text{Feature Extraction} \longrightarrow \text{Contextual Risk Fusion} \longrightarrow \text{Explanations \& Action Plans}$$

### Key Project Objectives
1. **Multi-Stage Classical IVP Preprocessing Matrix**: Apply an 8-stage computer vision matrix (16 output transformations) including CLAHE, spectral indexing (ExG, VARI, NGRDI), 2D Fourier frequency filtering, morphological operations, and distance transforms.
2. **Dual-Model Neural Perception**: Deploy deep learning models for fine-grained semantic land-cover segmentation (**EarthVL Custom ResNet-18 / LoveDA**) and overhead infrastructure detection (**YOLOv8 Nano / xView**).
3. **Contextual Multi-Hazard Fusion Engine**: Synthesize spatial indicators into normalized 0–100 risk scores for **Flood Susceptibility**, **Infrastructure Exposure**, and **Land Degradation**.
4. **Actionable Decision Support & LLM Explanations**: Generate natural-language executive summaries via local LLM (**Ollama Llama 3.2**) or rule-based fallback, paired with structured 3-tier disaster mitigation action plans (Immediate 0–48h, Mid-Term 1–6m, Long-Term 1–5y).
5. **Interactive Dashboard & Multi-Layer Overlay Visualizer**: Provide a responsive React dashboard featuring dynamic layer switching (Original RGB, LoveDA Segmentation Mask, YOLO Bounding Boxes, and EarthVL Risk Heatmap).

### Project Scope, Limitations & Task Assignment
- **Research Gap**: Existing systems typically solve one isolated task. The core contribution of this project is the combination of these into a continuous, modular pipeline: **evidence → context → risk fusion → ranking → explanation**.
- **System Limitations**: Satellite imagery may lack sufficient temporal/spatial resolution for extremely small objects. Risk outputs are configurable decision-support indicators, NOT definitive truth.
- **Future Scope**: Integration with live weather/hydrological APIs and continuous reassessment pipelines using bi-temporal change detection models.
- **Two-Person Task Assignment**:
  - **Person A (AI / Data / Risk Engine)**: Dataset Preparation, Segmentation/Detection Model Training, Feature Extraction, Configurable Risk Engine, ML Documentation.
  - **Person B (System / Product / Validation)**: FastAPI Backend, React Frontend Dashboard, Explanation Generation System, Visual Overlays, Testing, and Deployment.

### 11-Phase Academic IVP Methodology Mapping
1. **Problem Identification**: Lack of unified multi-hazard risk assessment from satellite imagery.
2. **Requirement Analysis**: Segmentation, detection, and configurable risk engine definition.
3. **Data Collection**: Sourcing LoveDA (segmentation) and xView (detection).
4. **Data Preprocessing**: Tiling, normalization, and balanced dataset splitting.
5. **Feature Extraction**: Deep learning models extracting spatial masks and bounding boxes.
6. **Proposed Algorithm**: Rule-based Contextual Risk Engine (Fusion of visual evidence).
7. **Training / Implementation**: Train DL baselines, build FastAPI backend and React frontend.
8. **Testing & Validation**: Test API endpoints, UI, model predictions, and spatial logic.
9. **Performance Evaluation**: Metrics tracking (IoU, mAP) and Risk Engine tuning.
10. **Results & Comparison**: Compare risk outputs across geographic regions and heuristic weights.
11. **Conclusion & Future Scope**: Summary of AI decision-support capabilities and future data integrations.

---

## 2. End-to-End System Architecture & Data Flow

### Architecture Diagram

```
 +-----------------------------------------------------------------------------------+
 |                                   FRONTEND                                        |
 |                             React + TS + Vite UI                                  |
 |  [Image Upload / Demo] ---> [Interactive Overlay Visualizer (Original/Seg/Det/Heat)]|
 |                            ---> [IVP 16-Stage Matrix Grid Explorer]               |
 |                            ---> [Executive Disaster Mitigation Modal]             |
 +------------------------------------------+----------------------------------------+
                                            | HTTP POST /api/analyze | GET /api/demo
                                            v
 +-----------------------------------------------------------------------------------+
 |                                   BACKEND                                         |
 |                             FastAPI (Python 3.11)                                 |
 |                                                                                   |
 |  +-----------------------------------------------------------------------------+  |
 |  | 1. IVP PREPROCESSING ENGINE (src/preprocessing/ivp_techniques.py)           |  |
 |  |    - Stage 1: Radiometric Calibration (Percentile Stretch, CLAHE, Gamma)   |  |
 |  |    - Stage 2: Noise Reduction (Gaussian, Median, Bilateral)                 |  |
 |  |    - Stage 3: Spectral Indexing (HSV, ExG, VARI, NGRDI)                    |  |
 |  |    - Stage 4: Edge Extraction (Sobel, LoG, Canny Edge Density)               |  |
 |  |    - Stage 5: Frequency Domain (2D FFT Spectrum, LPF, HPF)                  |  |
 |  |    - Stage 6: Morphology (Top-Hat Bright, Black-Hat Dark, Opening)           |  |
 |  |    - Stage 7: Binarization (Otsu, Adaptive Gaussian, Distance Transform)    |  |
 |  |    - Stage 8: Spatial Tiling (512x512 Patches & ImageNet Normalization)      |  |
 |  +-------------------------------------+---------------------------------------+  |
 |                                        | Feature Maps & Image Arrays              |
 |                                        v                                          |
 |  +-----------------------------------------------------------------------------+  |
 |  | 2. DEEP LEARNING INFERENCE LAYER (src/segmentation/ & src/detection/)      |  |
 |  |    - EarthVL Custom Segmentation (ResNet-18 Backbone + ConvTranspose2d)     |  |
 |  |    - YOLOv8 Nano Object Detector (xView Infrastructure Subset)              |  |
 |  +-------------------------------------+---------------------------------------+  |
 |                                        | Raw Masks & Bounding Boxes               |
 |                                        v                                          |
 |  +-----------------------------------------------------------------------------+  |
 |  | 3. HAZARD FEATURE EXTRACTION (src/feature_extraction/extractor.py)         |  |
 |  |    - Land-cover percentages (% Water, % Building, % Road, % Bare, % Veg)    |  |
 |  |    - Infrastructure density score & Water proximity detection flag         |  |
 |  +-------------------------------------+---------------------------------------+  |
 |                                        | Feature Metric Dictionary                |
 |                                        v                                          |
 |  +-----------------------------------------------------------------------------+  |
 |  | 4. CONTEXTUAL RISK FUSION ENGINE (src/risk_engine/engine.py)               |  |
 |  |    - Multi-Hazard Risk Scoring & Severity Rating (LOW/MODERATE/HIGH/CRITICAL)|  |
 |  +-------------------------------------+---------------------------------------+  |
 |                                        | Risk Scores & Hazard List                |
 |                                        v                                          |
 |  +-----------------------------------------------------------------------------+  |
 |  | 5. EXPLANATION & REPORT GENERATOR (src/explanation/generator.py)           |  |
 |  |    - Ollama LLM Llama3.2 Executive Summary Generation                        |  |
 |  |    - Structured Multi-Tier Disaster Mitigation Action Framework             |  |
 |  +-----------------------------------------------------------------------------+  |
 +-----------------------------------------------------------------------------------+
```

---

## 3. Image & Video Processing (IVP) 8-Stage Preprocessing Matrix

The IVP Engine ([ivp_techniques.py](file:///home/arth/Projects/DM/src/preprocessing/ivp_techniques.py)) transforms raw satellite RGB imagery into 16 distinct preprocessed feature representations across 8 mathematical stages:

### Stage 1: Radiometric Calibration & Dynamic Range Enhancement
- **Percentile Contrast Stretching**: Maps pixel values between the 0.5th and 99.5th percentiles to full $[0, 255]$ dynamic range to eliminate sensor clipping and atmospheric haze:
  $$I_{\text{stretched}} = \text{clip}\left(\frac{I - P_{0.5}}{P_{99.5} - P_{0.5} + \epsilon}, 0, 1\right) \times 255$$
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Operates on the Luminance ($L$) channel of the LAB color space using an $8 \times 8$ tile grid and a clip limit of $3.0$ to prevent noise amplification while highlighting micro-structures.
- **Global Histogram Equalization**: Equalizes grayscale intensity distribution.
- **Gamma Shadow Boost Correction**: Non-linear lookup table (LUT) transformation ($\gamma = 0.7$) to boost detail in deep shadows:
  $$I_{\text{boost}} = 255 \times \left(\frac{I}{255}\right)^{1/0.7}$$

### Stage 2: Spatial Noise Reduction & Smoothing Filters
- **Gaussian Filtering**: $5 \times 5$ kernel with spatial dispersion $\sigma = 1.0$ for high-frequency atmospheric speckle noise reduction.
- **Median Filtering**: $5 \times 5$ non-linear rank filter for salt-and-pepper sensor noise removal.
- **Bilateral Filtering**: Edge-preserving spatial smoothing with diameter $d = 9$, $\sigma_{\text{color}} = 75$, and $\sigma_{\text{space}} = 75$:
  $$I_{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) g_s(\|p - q\|) g_r(|I(p) - I(q)|)$$

### Stage 3: Spectral Indexing & Color Space Decomposition
- **HSV Decomposition**: Converts RGB to Hue, Saturation, Value space for robust illumination-invariant color segmentation.
- **ExG (Excess Green Index)**: Highlights vegetative canopy reflectance:
  $$\text{ExG} = 2G - R - B \quad \longrightarrow \quad \text{Rendered in Summer Colormap}$$
- **VARI (Visible Atmospherically Resistant Index)**: Minimizes atmospheric scattering using visible bands:
  $$\text{VARI} = \frac{G - R}{G + R - B + 10^{-5}} \quad \longrightarrow \quad \text{Rendered in Viridis Colormap}$$
- **NGRDI (Normalized Green-Red Difference Index)**:
  $$\text{NGRDI} = \frac{G - R}{G + R + 10^{-5}}$$

### Stage 4: Edge Extraction & Gradient Operators
- **Sobel Gradient Magnitude**: $3 \times 3$ directional convolution operators:
  $$G_x = \begin{bmatrix}-1 & 0 & 1\\-2 & 0 & 2\\-1 & 0 & 1\end{bmatrix} * I, \quad G_y = \begin{bmatrix}-1 & -2 & -1\\0 & 0 & 0\\1 & 2 & 1\end{bmatrix} * I, \quad M = \sqrt{G_x^2 + G_y^2}$$
- **Laplacian of Gaussian (LoG)**: Gaussian blur followed by 2nd-derivative Laplacian operator for zero-crossing corner/roof edge detection.
- **Multi-Threshold Canny Edge Detection**: Multi-stage edge detection with low threshold $50$ and high threshold $150$. Calculates `canny_edge_density` metric.

### Stage 5: Frequency Domain Analysis (2D Fast Fourier Transform)
- **2D Fast Fourier Transform (FFT)**: Converts spatial domain image $f(x,y)$ into complex frequency spectrum $F(u,v)$:
  $$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-j2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
- **Magnitude Spectrum**: Log-scaled visualization in Jet colormap: $20 \log(|F(u,v)| + 10^{-5})$.
- **Low-Pass Filter (LPF)**: Circular mask ($r = 40$ pixels) in frequency domain to retain background regional luminance.
- **High-Pass Filter (HPF)**: Complementary mask ($1 - \text{LPF}$) to isolate sharp infrastructure boundaries and linear features.

### Stage 6: Morphological Operations & Structural Filtering
Uses a $5 \times 5$ rectangular structuring element $K$:
- **Morphological Gradient**: $\text{Dilation}(I, K) - \text{Erosion}(I, K)$.
- **Top-Hat Transformation**: $I - \text{Opening}(I, K)$ to isolate bright targets (e.g., concrete building roofs).
- **Black-Hat Transformation**: $\text{Closing}(I, K) - I$ to isolate dark targets (e.g., shadows, water trenches).
- **Morphological Opening**: Erosion followed by Dilation for structural noise cleanup.

### Stage 7: Adaptive Thresholding & Binarization
- **Otsu’s Global Binarization**: Automatically calculates global threshold maximizing inter-class variance.
- **Adaptive Gaussian Thresholding**: Local threshold computed over an $11 \times 11$ neighborhood minus constant $C=2$.
- **Euclidean Distance Transform Map**: Computes distance of every foreground pixel to the nearest background boundary, rendered in Viridis colormap for structural thickness analysis.

### Stage 8: Spatial Tiling & Normalization
- **Tile Decomposition**: Splits large spatial images into $512 \times 512$ non-overlapping patches.
- **ImageNet Channel Statistics**: Computes channel-wise mean and standard deviation for model standardization:
  $$\mu_c = \frac{1}{H \times W} \sum_{x,y} I_c(x,y), \quad \sigma_c = \sqrt{\frac{1}{H \times W} \sum_{x,y} (I_c(x,y) - \mu_c)^2}$$

---

## 4. Deep Learning Models & Machine Learning Architecture

### 4.1 Semantic Segmentation Model: EarthVL Custom Net

Located in [train_earthvl_model.py](file:///home/arth/Projects/DM/scripts/train_earthvl_model.py) and [model.py](file:///home/arth/Projects/DM/src/segmentation/model.py).

#### Model Architecture Specification
- **Vision Backbone**: PyTorch ResNet-18 initialized with ImageNet weights (`ResNet18_Weights.DEFAULT`). Extracts spatial features at $\frac{1}{32}$ resolution with 512 feature channels.
- **Segmentation Decoder**: 5-stage transposed convolution upsampling stack restoring full input spatial dimensions:
  1. `ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)` + `ReLU()`
  2. `ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)` + `ReLU()`
  3. `ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)` + `ReLU()`
  4. `ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)` + `ReLU()`
  5. `ConvTranspose2d(32, num_classes=9, kernel_size=4, stride=2, padding=1)`
- **Target Classes (LoveDA Mapping)**:
  - `0`: Background / Ignore
  - `1`: Building (Red overlay: `#ef4444`)
  - `2`: Road (Yellow overlay: `#f59e0b`)
  - `3`: Water (Blue overlay: `#0ea5e9`)
  - `4`: Barren / Bare Land (Purple overlay: `#a855f7`)
  - `5`: Forest (Green overlay: `#22c55e`)
  - `6`: Agriculture (Green overlay: `#22c55e`)
- **Weights File**: Saved at `models/earthvl_custom.pt` (~55.9 MB).

#### Training Hyperparameters & Hardware Configuration
- **Optimizer**: `AdamW` ($\text{lr} = 1\times 10^{-4}$, $\text{weight\_decay} = 1\times 10^{-2}$).
- **Loss Function**: `nn.CrossEntropyLoss(ignore_index=255)`.
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast`) with `GradScaler`.
- **Hardware Profiles**:
  - `rtx4060` (Default, 8 GB VRAM Limit): `batch_size = 4`, `gradient_accumulation_steps = 2`, FP16 AMP enabled.
  - `h100` (High Performance Cluster): `batch_size = 32`, FP16 AMP enabled.

---

### 4.2 Overhead Object Detection Model: YOLOv8 Nano

Located in [model.py](file:///home/arth/Projects/DM/src/detection/model.py) and [train_detection.py](file:///home/arth/Projects/DM/scripts/train_detection.py).

#### Model Architecture Specification
- **Base Architecture**: Ultralytics YOLOv8 Nano (`yolov8n.pt`, ~6.5 MB weights).
- **Target Dataset**: xView Overhead Object Detection Dataset (subset).
- **Key Target Classes**: Building, Small Vehicle, Heavy Vehicle, Ship, Storage Tank, Bridge, Tower.
- **Inference Setup**:
  - Input resolution: $640 \times 640$ pixels.
  - Confidence threshold $\text{conf} = 0.25$.
  - Output bounding boxes rendered with cyan stroke (`#06b6d4`) and confidence labels.

---

## 5. Datasets & Data Preparation Pipeline

The project integrates three major satellite remote-sensing datasets:

### 1. EarthVL Dataset (`Kingdrone-Junjue/EarthVLSet`)
- **Source**: Hugging Face repository (`~18.1 GB`).
- **Citation**: *EarthVL: A Progressive Earth Vision-Language Understanding and Generation Framework* (arXiv:2601.02783).
- **Subsets**: `EarthVL-GLOBAL` (Land-cover segmentation patches) and `EarthVQA` (Visual Question Answering).
- **Downloader**: Scripted in [download_earthvl_dataset.py](file:///home/arth/Projects/DM/scripts/download_earthvl_dataset.py) with automatic 20-attempt retry logic and rate-limit handling.

### 2. LoveDA Dataset
- **Description**: Land-cover semantic segmentation dataset covering urban and rural scenes.
- **Classes**: 7 land-use categories (Building, Road, Water, Barren, Forest, Agriculture, Background).
- **Preparation & Inspection**: Automated manifest creation and split validation via [prepare_data.py](file:///home/arth/Projects/DM/scripts/prepare_data.py) and [extract_mvp_subset.py](file:///home/arth/Projects/DM/scripts/extract_mvp_subset.py).

### 3. xView Dataset
- **Description**: High-resolution overhead satellite object detection dataset.
- **Tiling**: Configured in [configs/data.yaml](file:///home/arth/Projects/DM/configs/data.yaml) for $1024 \times 1024$ tile sizes with $15\%$ overlap.

### 4. Case Study Methodology & Metadata Integrity
- **Purpose**: Provides curated real-world geospatial scenes (e.g., "Peri-urban Floodplain") for end-to-end evaluation.
- **Architectural Data Integrity**: Enforces a strict separation between **Externally Sourced Metadata** (objective ground-truth like acquisition timestamps, GPS coordinates, sensor GSD) and **Model-Derived Outputs** (features computed at runtime like `water_area_ratio` and risk scores). The system actively prevents hardcoding or fabricating subjective historical hazard probabilities.

---

## 6. Hazard Feature Extraction & Proxies

The feature extraction layer ([extractor.py](file:///home/arth/Projects/DM/src/feature_extraction/extractor.py)) evaluates spatial masks and object detections to compute quantitative environmental risk indicators:

### Extracted Indicators
1. **Land Cover Percentages**:
   $$\% \text{Class} = \frac{\sum \text{Pixels where Mask} == \text{ClassID}}{N_{\text{total\_pixels}}} \times 100$$
2. **Infrastructure Density Score**:
   $$\text{Infrastructure Density} = \% \text{Building} + (0.5 \times \% \text{Road})$$
3. **Water Proximity Flag**:
   $$\text{Water Proximity} = \begin{cases} \text{True} & \text{if } \% \text{Water} > 5.0\% \text{ and } \% \text{Building} > 10.0\% \\ \text{False} & \text{otherwise} \end{cases}$$
4. **HSV Color/Texture Fallback Proxies**: Executed automatically if neural weights are offline, utilizing spectral thresholding on Hue, Saturation, and Value bands.

---

## 7. Contextual Multi-Hazard Risk Fusion Engine

The Risk Engine ([engine.py](file:///home/arth/Projects/DM/src/risk_engine/engine.py)) synthesizes extracted spatial indicators into 0–100 risk scores across three core hazard categories:

### 1. Flood Susceptibility / Exposure Risk
- **Formula**:
  $$\text{Score}_{\text{flood}} = \min\left(100, S_{\text{base}} + (\text{Model Confidence} \times 10)\right)$$
  - $+40$ pts if `water_proximity_flag` is True (dense buildings near water).
  - $+30$ pts if `water_percentage` $> 15\%$.
  - $+20$ pts if `building_percentage` $> 20\%$.

### 2. Infrastructure & Urban Exposure Risk
- **Formula**:
  $$\text{Score}_{\text{infra}} = \min\left(100, S_{\text{infra\_base}}\right)$$
  - $+45$ pts if `infrastructure_density_score` $> 22$.
  - $+25$ pts if `building_percentage` $> 18\%$.
  - $+15$ pts if `road_percentage` $> 8\%$.

### 3. Land Degradation & Vegetation Loss Risk
- **Formula**:
  $$\text{Score}_{\text{degrad}} = \min\left(100, S_{\text{degrad\_base}}\right)$$
  - $+50$ pts if `bare_land_percentage` $> 30\%$.
  - $+30$ pts if `vegetation_percentage` $< 10\%$.

### Overall Risk Score & Severity Classification
The overall risk score is defined as $\text{Overall Risk} = \max(\text{Score}_{\text{flood}}, \text{Score}_{\text{infra}}, \text{Score}_{\text{degrad}})$.

| Overall Risk Score | Severity Level | UI Color Code |
| :--- | :--- | :--- |
| **80 – 100** | **CRITICAL** | Red (`#ef4444`) |
| **60 – 79** | **HIGH** | Orange (`#f97316`) |
| **40 – 59** | **MODERATE** | Amber (`#f59e0b`) |
| **0 – 39** | **LOW** | Emerald (`#10b981`) |

---

## 8. Natural Language Explanations & Mitigation Framework

Located in [generator.py](file:///home/arth/Projects/DM/src/explanation/generator.py).

### 8.1 LLM Executive Summarization
- **Primary Engine**: Queries local Ollama endpoint (`http://localhost:11434/api/generate`) running `llama3.2`.
- **Template Fallback**: Synthesizes a structured 3-sentence summary incorporating risk rating, primary hazard type, and specific spatial driver metrics.

### 8.2 Structured 3-Tier Disaster Mitigation Report
Generates an actionable, domain-specific disaster mitigation plan formatted into three distinct operational timelines:

1. **Immediate Actions (0 – 48 Hours)**: Emergency advisories, pre-positioning mobile water pumps, drainage channel inspections.
2. **Mid-Term Mitigation (1 – 6 Months)**: Retention basin construction, slope bio-engineering, SuDS implementation.
3. **Long-Term Resilience (1 – 5 Years)**: Regional watershed management, green infrastructure zoning, bi-temporal satellite SAR tracking.

---

## 9. API Specifications & Backend Routes

Built with FastAPI in [routes.py](file:///home/arth/Projects/DM/backend/api/routes.py).

### API Endpoints Overview

| Method | Endpoint | Description | Request | Response |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/analyze` | Full IVP + ML analysis on uploaded image | `multipart/form-data` (`file`) | `RiskResponse` (JSON) |
| `GET` | `/api/demo` | Curated peri-urban demo scenario execution | None | `RiskResponse` (JSON) |
| `GET` | `/api/health` | Service health check | None | `{"status": "healthy"}` |

### Key Response Schema (`RiskResponse`)
```json
{
  "location": "2522.png (Peri-Urban Zone)",
  "overall_risk": 75,
  "risk_level": "HIGH",
  "hazards": [
    {
      "type": "flood_exposure",
      "score": 75,
      "confidence": 0.85,
      "severity": "HIGH",
      "evidence": ["Dense built-up area detected near water.", "High water coverage (18.2%)."],
      "recommendations": ["Prioritize drainage assessment.", "Inspect exposed structures."]
    }
  ],
  "explanation": "EarthVL risk fusion identified a HIGH threat level (75/100)...",
  "features": {
    "water_percentage": 18.2,
    "building_percentage": 22.4,
    "road_percentage": 9.1,
    "vegetation_percentage": 14.5,
    "bare_land_percentage": 35.8,
    "infrastructure_density_score": 26.95
  },
  "ivp_metrics": {
    "image_dimensions": [512, 512, 3],
    "mean_intensity": 118.42,
    "std_intensity": 45.12,
    "image_entropy": 7.1234,
    "canny_edge_density": 0.0845,
    "tile_count": 1,
    "channel_mean": [0.462, 0.478, 0.441],
    "channel_std": [0.182, 0.175, 0.191]
  },
  "ivp_stages": [ /* 16 IVP Stage Objects with Base64 Images & Details */ ],
  "original_image_base64": "data:image/png;base64,...",
  "segmentation_mask_base64": "data:image/png;base64,...",
  "detection_overlay_base64": "data:image/png;base64,...",
  "earthvl_risk_heatmap_base64": "data:image/png;base64,...",
  "mitigation_report": {
    "disaster_type": "Riverine / Surface Flood Susceptibility",
    "risk_level": "HIGH",
    "overall_score": 75,
    "summary": "...",
    "vulnerability_factors": [...],
    "immediate_actions": [...],
    "mid_term_actions": [...],
    "long_term_actions": [...]
  }
}
```

---

## 10. Frontend Dashboard & User Interface System

Built with React 18, TypeScript, and Vite ([frontend/src](file:///home/arth/Projects/DM/frontend/src)).

### Component Architecture

```
[App.tsx]
  ├── Top Navigation Bar (Status indicator, Brand title)
  ├── Hero Upload Section (Uploader.tsx - Drag & Drop / Curated Demo Trigger)
  └── [Dashboard.tsx]
        ├── Executive Overview Header & Overall Risk Meter Dial
        ├── Hazard Severity Breakdown Cards
        ├── Environmental Feature Distribution Bars
        ├── [Visualizer.tsx] (Multi-Layer Overlay Canvas Switcher)
        ├── [IVPStageExplorer.tsx] (16-Panel Interactive Stage Matrix Grid)
        └── [ExecutiveReportModal.tsx] (3-Tier Disaster Mitigation Plan)
```

### Visualizer Layer Switcher
Allows live toggle between four spatial overlays:
1. **Original RGB**: Raw satellite image patch.
2. **Segmentation Overlay**: Colorized LoveDA land-cover mask (Buildings, Roads, Water, Barren, Vegetation).
3. **Detection Bounding Boxes**: Cyan YOLOv8 overhead object boxes.
4. **EarthVL Risk Heatmap**: OpenCV JET colormap blending distance transforms and edge density maps.

---

## 11. Verification, Execution & Testing Guide

### 11.1 Environment Setup
```bash
# Clone repository and install dependencies
cd /home/arth/Projects/DM
pip install -r requirements.txt
```

### 11.2 Run IVP Pipeline Demo
Generate all 16 preprocessed IVP stage images and a 4x4 comparison grid:
```bash
python scripts/demo_ivp_pipeline.py --dataset_dir dataset --num_samples 3
```

### 11.3 Train EarthVL Segmentation Model
```bash
# Fast pipeline dry-run verification
python scripts/train_earthvl_model.py --dry-run

# Train model on RTX 4060 GPU profile (8 GB VRAM)
python scripts/train_earthvl_model.py --epochs 5 --batch-size 4 --vram-mode rtx4060
```

### 11.4 Start System Services
```bash
# Terminal 1: Launch FastAPI Backend Server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch React Frontend Development Server
cd frontend
npm install
npm run dev
```
Open web browser at `http://localhost:5173` to interact with the full dashboard.

---
*TerraRisk Technical Manual - Comprehensive Single-File Reference*
