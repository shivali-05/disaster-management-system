# Data Sources & Acquisition Strategy

This document provides documentation for all data sources incorporated into the Multi-Hazard Risk and Infrastructure Assessment system. It outlines the role, dataset structure, subsetting methodology, and data contracts for each source.

---

## 1. LoveDA (Land-Cover Dataset for Domain Adaptation)

### 1.1 Intended Segmentation Role
LoveDA serves as the primary dataset for the **Semantic Segmentation** pipeline. It provides high-resolution 0.3m ground sampling distance (GSD) satellite imagery covering both urban and rural scenes. The segmentation outputs provide essential spatial evidence of land cover, surface composition, and built-up infrastructure to inform multi-hazard risk assessment models (e.g., identifying proximity of structures to water bodies or impervious surface ratios).

### 1.2 Expected Classes
The dataset defines 7 discrete semantic classes:
- **0: Ignore / No-data** (Ignored in training loss and evaluation)
- **1: Background** (Background / non-target region)
- **2: Building** (Built environment)
- **3: Road** (Built environment / transport network)
- **4: Water** (Hydrological features / flood risk driver)
- **5: Barren** (Bare soil / land degradation risk indicator)
- **6: Forest** (Natural vegetation cover)
- **7: Agriculture** (Cultivated vegetation cover)

### 1.3 MVP Subset Strategy
To balance training efficiency and statistical representative power for the MVP without requiring full-dataset storage or multi-day training runs, a balanced subset strategy is employed while strictly preserving the official dataset split boundaries:

| Split | Urban Scenes | Rural Scenes | Total Tiles |
|---|---:|---:|---:|
| **Train** | 400 | 400 | 800 |
| **Validation** | 75 | 75 | 150 |
| **Test** | 75 | 75 | 150 |
| **Total** | **550** | **550** | **1,100** |

Subset selection is executed deterministically using a fixed random seed (`seed: 42`).

### 1.4 Manifest Location
All selected LoveDA samples, along with their image paths, mask paths, dimensions, scene types, split assignments, and validity flags, are recorded in the manifest:
- `data/metadata/loveda_manifest.csv`

---

## 2. xView (Overhead Object Detection)

### 2.1 Intended Detection Role
xView is designated as the primary data source for **Overhead Infrastructure Object Detection**. Its role is to identify discrete infrastructure components (e.g., small buildings, bridges, towers, vehicles, storage tanks) from high-resolution satellite imagery, providing quantitative object counts and density metrics to the risk engine.

### 2.2 Annotation Inspection Before Class Selection
xView contains over 60 native object classes across complex overhead scenes with high class imbalance. In accordance with project requirements and `docs/DATA_HANDLING.md`:
- Large-scale tiling and class generation must **not** occur blindly.
- An inspection phase must first parse the raw annotations, produce class frequency distributions, analyze bounding box aspect ratios and physical dimensions, and assess the co-occurrence of objects across source scenes.

### 2.3 Class Selection Status
- **No final detection classes are selected yet.**
- The detection mapping in `data/metadata/class_mapping.json` remains an empty dictionary (`"detection": {}`).
- Final selection of 3–5 defensible infrastructure classes will be made and documented in Phase 3 after statistical inspection of class distributions and evaluation of relevance to downstream hazard indicators.

---

## 3. Case Study Data

### 3.1 Purpose
Case study data provides curated real-world geospatial imagery scenes to evaluate and demonstrate the end-to-end inference pipeline:
$$\text{Raw Image} \rightarrow \text{ML Inference} \rightarrow \text{Feature Extraction} \rightarrow \text{Contextual Risk Fusion} \rightarrow \text{Natural Language Explanation}$$

These scenarios illustrate realistic decision-support workflows (such as urban flood exposure or infrastructure density near water bodies) on representative regions.

### 3.2 Required Metadata
Each case study entry must reside in a dedicated directory under `data/raw/case_study/<case_id>/` and be accompanied by a structured `metadata.json` document containing:
- `case_id`: Stable, unique identifier for the case scenario.
- `location_name`: Human-readable location description (e.g., "Peri-urban Floodplain, Krishna River Basin").
- `latitude` / `longitude`: Geographic coordinates of the scene center.
- `image_date`: Timestamp or acquisition date of the satellite capture.
- `source`: Satellite provider or sensor source (e.g., Sentinel-2, Bhuvan, Google Earth / Maxar).
- `spatial_resolution_m`: Spatial ground resolution per pixel in meters.
- `notes`: Relevant scenario context or environmental conditions.
- `limitations`: Explicitly documented data limitations or sensor constraints.

### 3.3 Distinction Between Source Metadata and Model-Derived Output
A strict conceptual and architectural distinction is enforced:
- **Externally Sourced Metadata**: Objective, verified ground-truth facts provided by the source sensor or provider (e.g., acquisition timestamp, sensor GSD, geographical coordinates, external digital elevation model data if present). The system must never fabricate or hardcode subjective risk variables (such as historical damage probabilities or arbitrary hazard likelihoods) into source metadata.
- **Model-Derived Output**: Quantitative features extracted dynamically by the computer vision models and risk engine (e.g., `water_area_ratio`, `building_water_proximity`, `flood_exposure_score`, model confidence scores). These values are computed at runtime and clearly labeled as AI decision-support estimates.
