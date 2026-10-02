# Project Master Document

## 1. Problem Statement
Assessing multiple hazard risks (e.g., floods, infrastructure vulnerability, land degradation) from satellite imagery typically requires disparate models and manual interpretation, making it difficult to generate a unified, actionable risk assessment.

## 2. Motivation
To build a functional prototype that automatically converts raw remote sensing data into a comprehensive risk overview, providing decision-support backed by interpretable AI evidence.

## 3. Research Gap
Existing systems typically solve one isolated task (segmentation, object detection, or flood detection). The core contribution of this project is the combination of these into a continuous, modular pipeline: **evidence → context → risk fusion → ranking → explanation**.

## 4. Project Objectives
- Develop a semantic segmentation baseline for land/infrastructure extraction.
- Implement an object detection baseline for overhead infrastructure identification.
- Build a configurable rule-based contextual risk engine.
- Provide a visually intuitive dashboard demonstrating the full pipeline.

## 5. Scope
The prototype focuses on a limited number of defensible hazard modules (Flood Exposure, Infrastructure Risk, Land Degradation, Change Detection). 

## 6. Non-Goals
- We will NOT attempt to predict every disaster type (e.g., earthquakes, fires).
- We will NOT build a fake "live satellite" system.
- We will NOT use an LLM to hallucinate computer-vision inference.

## 7. System Architecture
Please refer to [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 8. Data Pipeline
Data will be sourced from standard datasets (LoveDA, xView) and potentially one geospatial provider (e.g., Sentinel Hub/Bhuvan). Preprocessing will involve tiling, normalization, and train/val/test splitting.

## 9. Model Pipeline
- **Segmentation**: U-Net or DeepLab family models.
- **Detection**: YOLO-family detectors.

## 10. Risk Engine
Combines segmentation evidence, detected infrastructure, proximity logic, geographic factors, and model confidence to generate a normalized 0-100 risk score.

## 11. Dataset Justification
- **LoveDA**: Proven dataset for semantic segmentation, urban/rural comparisons, and land-cover extraction.
- **xView**: Standard dataset for overhead object detection and infrastructure evidence.

## 12. Model Justification
- We use practical, reproducible baselines (U-Net/YOLO) rather than huge foundation models, to ensure the prototype runs efficiently.

## 13. Evaluation Methodology
- **Segmentation**: Train/val loss, IoU, Dice/F1, pixel accuracy.
- **Detection**: Precision, recall, mAP@0.5, mAP@0.5:0.95.
- **Risk Engine**: Manual inspection and comparison of several risk-weight configurations.

## 14. Expected Outputs
- An interpretable risk score (0-100).
- Visual overlays (heatmaps, segmentation masks, bounding boxes).
- Structured evidence and template-based natural language explanations.

## 15. Limitations
- Satellite imagery may lack sufficient temporal or spatial resolution for small objects.
- High-risk outputs are configurable indicators, NOT definitive truth, and require human verification.
- Not every hazard is visually observable from overhead imagery alone.

## 16. Future Scope
- Integration with live weather and hydrological APIs.
- Continuous reassessment pipelines with active change detection models.

## 17. Reproducibility Instructions
Refer to `scripts/` directory for automated data downloading, preprocessing, and training scripts.

## 18. Academic Justification
The project aligns with the IVP framework, transforming theoretical computer vision concepts into an applied, end-to-end decision-support prototype.

## 19. Demo Scenario
An urban/peri-urban Indian region will be processed to show buildings near a water body, automatically flagged as "HIGH" flood-exposure risk due to proximity, with visual evidence presented.

## 20. Division of Work
Please refer to [TASK_ASSIGNMENT.md](TASK_ASSIGNMENT.md) for details on the two-person division.
