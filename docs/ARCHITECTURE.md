# System Architecture

## Core Philosophy
The system strictly separates visual AI inference from logical risk interpretation. The AI does not "know" the risk; it only provides structured visual evidence.

### 1. Visual Evidence
Extracted by the Deep Learning modules.
*Example: "Water body detected 82m from built-up area."*

### 2. Contextual Inference
Derived by analyzing the spatial relationships of the visual evidence.
*Example: "High building exposure near water increases flood exposure."*

### 3. Risk Interpretation
Computed by the Risk Engine using weighted heuristics.
*Example: "Flood exposure risk: HIGH (76/100)."*

### 4. Confidence
Model uncertainty passed through the pipeline.
*Example: "Model confidence: 0.84."*

## System Components

1. **Frontend (React + TypeScript)**
   - Upload image, display masks, overlays, bounding boxes, risk dashboard, and heatmaps.

2. **Backend (Python + FastAPI)**
   - Orchestrates ML inference, calculates spatial features, and serves the API.

3. **ML Pipeline (PyTorch)**
   - **Semantic Segmentation:** Evaluates land cover (LoveDA).
   - **Object Detection:** Identifies infrastructure (xView).

4. **Risk Engine**
   - Rules-based fusion scoring combining the structured JSON evidence from the ML pipeline.

5. **Explanation Layer**
   - Natural language generation interpreting the structural evidence into textual response.

## Detailed Model Training Architecture

### 1. Semantic Segmentation (LoveDA)
- **Model:** U-Net or DeepLabV3+ with a ResNet50 backbone.
- **Framework:** PyTorch & `segmentation_models_pytorch`.
- **Classes:** 
  - Background (0)
  - Building (1)
  - Road (2)
  - Water (3)
  - Barren (4)
  - Forest/Agriculture (5, 6)
- **Training Strategy:**
  - **Loss Function:** Combined Cross-Entropy + Dice Loss to handle class imbalance.
  - **Optimizer:** AdamW with an initial learning rate of `1e-4`, using CosineAnnealingLR.
  - **Data Augmentation:** Random cropping (512x512), horizontal/vertical flips, color jitter (brightness, contrast).
  - **Metrics:** mIoU, Pixel Accuracy, per-class Dice coefficient.
  - **Hardware Strategy:** Automatic mixed precision (AMP) used when a CUDA GPU is present to speed up training and reduce VRAM.

### 2. Object Detection (xView)
- **Model:** YOLOv8 (using Ultralytics library).
- **Framework:** PyTorch.
- **Classes:** Filtered subset of xView focusing on infrastructure (e.g., small buildings, bridges, towers, vehicles).
- **Training Strategy:**
  - **Pre-trained Weights:** Starting from `yolov8n.pt` or `yolov8s.pt` to ensure fast convergence.
  - **Data Augmentation:** Mosaic augmentation, HSV transformations, random scaling.
  - **Metrics:** mAP@0.5 and mAP@0.5:0.95.
  - **Inference:** Configurable confidence thresholds (e.g., `conf=0.25`, `iou=0.45`).

## Output Format

The output of the system is bipartite: a structured JSON output intended for the dashboard rendering, and a Natural Language textual response giving complete details about the input photo.

### 1. Structured JSON Output
```json
{

  "location": "Latitude/Longitude or Image Identifier",
  "overall_risk": 78,
  "risk_level": "HIGH",
  "hazards": [
    {
      "type": "flood_exposure",
      "score": 81,
      "confidence": 0.84,
      "evidence": [
        "dense buildings near water",
        "low elevation",
        "recent water expansion"
      ],
      "recommendations": [
        "prioritize drainage assessment",
        "inspect exposed structures",
        "monitor water expansion"
      ]
    }
  ]
}
```

### 2. Natural Language Textual Response (Explanation)
Generated using template-based NLP over the structured JSON evidence, ensuring the text is grounded entirely in actual ML predictions without hallucination.

**Example NLP Output:**
> **HIGH FLOOD-EXPOSURE RISK**
> 
> **Reason:** 
> Analysis of the input photo reveals a high flood-exposure risk. A dense built-up area was detected within 82 meters of a mapped water body. Additionally, building density is exceptionally high in this low-lying exposure zone.
> 
> **Confidence:** 0.84 (Model predictions are highly confident regarding the water body boundary).
> 
> **Recommended Action:**
> Prioritize drainage assessment, inspect exposed structures, and continuously monitor the water expansion in subsequent satellite passes.
