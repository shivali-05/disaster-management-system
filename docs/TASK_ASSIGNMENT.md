# Two-Person Task Assignment

The project is divided into two major streams with shared integration responsibilities.

## Person A: AI / Data / Risk Engine
Responsible for the core intelligence and feature extraction.
- **Dataset Preparation:** Download, clean, and preprocess LoveDA and xView subsets.
- **Segmentation (LoveDA):** Train and evaluate the baseline (U-Net/DeepLab).
- **Detection (xView):** Train and evaluate the baseline (YOLO).
- **Feature Extraction:** Build algorithms to measure proximities and densities.
- **Risk Engine:** Design and implement the configurable scoring rules.
- **Documentation:** Write ML evaluations and experimental reports.

## Person B: System / Product / Validation
Responsible for the end-to-end application, orchestration, and user experience.
- **Backend (API):** Develop FastAPI endpoints for inference and orchestration.
- **Frontend (Dashboard):** Build the React UI for uploading images and visualizing results.
- **Explanation System:** Implement template-based natural language explanations.
- **Visualization:** Generate heatmaps and overlay graphics.
- **Testing:** Write system tests and ensure smooth UI/UX.
- **Deployment:** Handle dockerization and demonstration setup.

## Shared Responsibilities
- System Architecture and Research Design.
- Integration between the ML Engine and the Backend API.
- Final validation, report writing, and presentation.
