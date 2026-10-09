# IVP Mapping

This document maps the project implementation to the academic IVP methodology.

| IVP Phase | Implementation Step | Artifacts / Evidence Required |
| :--- | :--- | :--- |
| **1. Problem Identification** | Define the need for multi-hazard risk assessment from satellite imagery. | Problem statement in PROJECT_MASTER.md |
| **2. Requirement Analysis** | Define scope: segmentation, detection, and configurable risk engine. | ARCHITECTURE.md, TASK_ASSIGNMENT.md |
| **3. Data Collection** | Select LoveDA (segmentation) and xView (detection). | Download scripts, Dataset documentation |
| **4. Data Preprocessing** | Tiling, normalization, train/val/test splits. | Preprocessing scripts, data samples |
| **5. Feature Extraction** | Deep learning models (U-Net, YOLO) outputting bounding boxes and masks. | Model architectures, training scripts |
| **6. Proposed Algorithm** | Contextual Risk Engine (Fusion of visual evidence + geography). | Risk Engine code, heuristic rules |
| **7. Training / Implementation**| Train baselines, build FastAPI backend and React frontend. | Checkpoints, working API and Dashboard |
| **8. Testing & Validation** | Test API, UI, model predictions, and spatial logic. | Unit tests, manual validation logs |
| **9. Performance Evaluation** | Metrics: IoU, mAP, and Risk Engine confidence tuning. | EXPERIMENTS.md, confusion matrices |
| **10. Results & Comparison** | Compare risk outputs across different heuristic weights and regions. | Visual overlays, risk score tables |
| **11. Conclusion & Future Scope**| Summarize limitations and potential integrations (e.g., live weather). | Final Report, README.md |
