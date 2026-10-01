# AI-Based Multi-Hazard Risk and Infrastructure Assessment using Satellite Imagery

This project is a real, runnable prototype of a multi-hazard risk and infrastructure assessment system.
It processes satellite/remote-sensing images to extract semantic segments (e.g., buildings, water) and object detections, then applies a contextual fusion engine to output an interpretable risk score along with visual evidence and human-readable explanations.

## Overview
Existing systems generally tackle single hazards (e.g., flood detection) or single tasks (e.g., segmentation). 
This project combines these into a continuous pipeline:
**evidence → context → risk fusion → ranking → explanation**

## Architecture
- **Backend**: FastAPI
- **Frontend**: React + TypeScript
- **ML Engine**: PyTorch (LoveDA for Segmentation, xView for Detection)
- **Risk Engine**: Rule-based contextual fusion

## Demo Setup & Running

This version is ready for the mid-project demonstration. It deliberately does
not download or initialise untrained neural-network weights at startup.

1. Install the Python dependencies into the project virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend (either command works):

   ```bash
   cd backend && uvicorn main:app --reload
   # or, from the repository root:
   uvicorn backend.main:app --reload
   ```

3. In a second terminal, start the frontend:

   ```bash
   cd frontend && npm install && npm run dev
   ```

4. Open the local address displayed by the frontend (normally
   `http://localhost:5173`). Select **Open demo assessment** for the prepared
   presentation scenario, or upload an image for a textual image-derived
   assessment.

## What works now

- A polished React dashboard with textual risk score, evidence, feature bars
  and recommended next steps.
- A clearly labelled curated demo scenario for an immediate presentation.
- Uploaded-image analysis using deterministic colour/texture land-cover
  proxies (water, vegetation, bare land, built-up and road-surface cues).
- An interpretable flood-exposure, infrastructure-exposure and land-degradation
  risk engine.

The upload result is not presented as model inference or a disaster prediction.
It is a repeatable computer-vision baseline and a decision-support indicator.

## Adding a trained model later

Yes. The intended next integration is a trained LoveDA segmentation checkpoint
and an overhead-image detector checkpoint trained on an xView subset. Add the
checkpoint only after validating its licence, classes and metrics. The current
API is intentionally separated from model loading so a future model adapter can
replace `FeatureExtractor.extract_features_from_image()` without changing the
frontend or risk-engine contract. Do not use the generic `yolov8n.pt` fallback
as an xView model: it is not trained for the project’s satellite classes.

## Documentation
Please check the `docs/` folder for comprehensive documentation on architecture, datasets, models, and task assignment.
