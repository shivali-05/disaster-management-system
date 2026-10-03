# DATA_HANDLING.md

## 1. Purpose

This document defines how data is selected, acquired, validated, prepared, consumed, and transformed throughout the Multi-Hazard Risk Assessment MVP.

The project combines:

- **semantic segmentation** for land-cover and scene understanding;
- **object detection** for infrastructure and relevant objects;
- **feature extraction** for converting model outputs into structured measurements;
- **risk assessment** for combining visual and contextual indicators into an interpretable risk result.

The data pipeline is designed for an MVP. The objective is therefore **not to collect or process the largest possible dataset**, but to use a reproducible amount of data that is sufficient to validate the end-to-end pipeline.

This document is the implementation contract for data handling. Dataset-specific processing must follow the responsibilities and boundaries defined here.

---

# 2. Data Handling Principles

## 2.1 Reproducibility

Dataset selection and preparation must be reproducible.

Where a dataset subset is used:

- selection must use a fixed random seed where random sampling is involved;
- selected samples must be recorded in metadata;
- source dataset splits must be preserved where applicable;
- the pipeline must recreate the same subset without manual file selection.

Raw dataset files must never be the only record of a data decision.

---

## 2.2 Preserve Raw Data

Raw datasets remain unchanged.

```text
Raw Data
   ↓
Validation and Inspection
   ↓
Metadata / Manifest Generation
   ↓
Selection and Preparation
   ↓
Training-Ready Data
   ↓
Model Inference
   ↓
Feature Extraction
```

No preparation step should overwrite original dataset files.

---

## 2.3 MVP-First Dataset Size

The MVP does not require full-scale processing of every available dataset.

The selected data should be:

- large enough to train and validate the pipeline;
- balanced enough to expose relevant variation;
- small enough to process and retrain practically;
- documented so the subset can later be expanded.

The project should prefer **controlled subsets of established datasets** over unnecessarily large custom collections.

---

## 2.4 Separation of Responsibilities

```text
Dataset
   ↓
Preprocessing
   ↓
Segmentation / Detection
   ↓
Feature Extraction
   ↓
Risk Engine
   ↓
Explanation
```

The risk engine consumes structured features, not raw images, masks, tensors, or model-specific outputs.

---

# 3. Data Categories

The MVP uses three data categories.

| Category | Purpose | Primary Output |
|---|---|---|
| Segmentation Data | Understand land cover and scene composition | Semantic mask |
| Detection Data | Identify relevant infrastructure / objects | Bounding boxes and classes |
| Case Study Data | Demonstrate the complete end-to-end system | Risk assessment input |

---

# 4. Proposed Dataset Strategy

## 4.1 Semantic Segmentation Dataset

The proposed segmentation dataset is **LoveDA**.

It provides land-cover information required for visual risk indicators.

Relevant output categories include:

- background;
- building;
- road;
- water;
- barren land;
- forest;
- agriculture.

The no-data / ignore label must not contribute to training loss or feature calculations where the dataset defines it as invalid.

### MVP Subset Strategy

The project should preserve original dataset split boundaries and select a controlled subset within those boundaries.

Initial target:

| Split | Urban | Rural | Total |
|---|---:|---:|---:|
| Train | 400 | 400 | 800 |
| Validation | 75 | 75 | 150 |
| Test | 75 | 75 | 150 |
| **Total** | **550** | **550** | **1,100** |

These are implementation targets, not instructions to create new dataset splits.

The selected subset must be recorded in a manifest.

---

## 4.2 Object Detection Dataset

The proposed detection dataset is **xView** or the detection dataset selected during implementation.

The full dataset should not automatically be processed for the MVP.

The workflow is:

```text
Inspect annotations
   ↓
Generate class distribution report
   ↓
Evaluate object frequency and image coverage
   ↓
Select 3–5 relevant classes
   ↓
Select source scenes containing those classes
   ↓
Prepare only the required training subset
```

Final detection classes must be selected using actual dataset statistics and project relevance.

---

## 4.3 Case Study Data

Case study data demonstrates end-to-end inference.

Each case should contain:

```text
case_id
current image
source metadata
image date, where available
location/context metadata, where available
documented limitations
```

The MVP does not require a large case-study dataset.

A small number of carefully documented cases is sufficient.

Temporal comparison data is optional for the initial MVP and must not block the core pipeline.

---

# 5. Repository Data Structure

The project should add a minimal data structure without restructuring existing source modules.

```text
data/
│
├── raw/
│   ├── loveda/
│   ├── xview/
│   └── case_study/
│
├── processed/
│   ├── detection/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   │
│   └── case_study/
│
├── metadata/
│   ├── class_mapping.json
│   ├── feature_schema.json
│   ├── loveda_manifest.csv
│   ├── xview_manifest.csv
│   └── dataset_summary.json
│
└── samples/
    ├── segmentation/
    └── detection/
```

Raw and large processed datasets must be ignored by Git.

The repository stores:

- code;
- configuration;
- manifests;
- schemas;
- documentation;
- small representative samples.

It does not store large source datasets or bulk generated artifacts.

---

# 6. Data Pipeline Ownership

## 6.1 `scripts/download_data.py`

Responsible for dataset acquisition and source verification.

It may:

- download supported datasets;
- verify expected files;
- record source information;
- register a locally available dataset.

It must not:

- train models;
- perform model inference;
- tile images;
- silently create dataset splits.

---

## 6.2 `scripts/prepare_data.py`

This is the primary dataset preparation entry point.

```text
Dataset Validation
    ↓
Label Inspection
    ↓
Manifest Creation
    ↓
Subset Selection
    ↓
Dataset-Specific Preparation
    ↓
Validation of Prepared Output
```

For segmentation data, preparation primarily produces metadata and selection information.

For detection data, preparation may additionally create processed tiles and converted labels.

---

## 6.3 `src/preprocessing/dataset.py`

This is the reusable data-loading layer.

It should provide dataset loaders for:

- semantic segmentation;
- object detection.

It loads selected data and applies runtime transformations.

It should not make dataset selection decisions.

---

## 6.4 Training Scripts

```text
scripts/train_segmentation.py
scripts/train_detection.py
```

Training scripts consume:

- configuration;
- manifests;
- dataset loaders.

They should not contain hardcoded file-selection logic.

---

## 6.5 `src/feature_extraction/extractor.py`

This module converts model outputs into structured risk features.

It is the boundary between computer vision outputs and the risk engine.

---

# 7. Segmentation Data Handling

## 7.1 Validation

Before a sample is used, validation checks:

- image exists;
- mask exists;
- image dimensions;
- mask dimensions;
- image and mask dimensions match;
- image format is supported;
- mask label values are valid;
- invalid / no-data regions are identified.

The output is recorded in a manifest.

---

## 7.2 Segmentation Manifest

Minimum fields:

```text
sample_id
split
scene_type
image_path
mask_path
width
height
valid
selected
```

Example:

```text
sample_id,split,scene_type,image_path,mask_path,width,height,valid,selected
urban_001,train,urban,...,...,1024,1024,true,true
urban_002,train,urban,...,...,1024,1024,true,false
```

The manifest is the source of truth for MVP sample selection.

---

## 7.3 Runtime Preprocessing

Segmentation images may be resized and normalized during loading/training.

For segmentation masks:

- class labels remain discrete;
- resizing uses nearest-neighbour interpolation;
- image and mask interpolation must not be treated as equivalent.

Training augmentation should occur at runtime rather than storing duplicate augmented images.

Potential MVP augmentation:

- horizontal flip;
- vertical flip;
- 90-degree rotation;
- limited colour variation for images only.

Validation and test data use deterministic preprocessing only.

---

# 8. Detection Data Handling

## 8.1 Inspection Before Selection

The detection dataset must first be inspected.

The inspection stage should produce:

```text
class_id
class_name
object_count
unique_image_count
object_size_statistics
```

This stage is required before final class selection.

---

## 8.2 Class Selection

The MVP should select a limited number of classes.

Target:

```text
3–5 classes
```

Selection considers:

- object frequency;
- number of unique source scenes;
- infrastructure relevance;
- visual detectability;
- usefulness for downstream risk features.

The final class mapping must be recorded.

---

## 8.3 Source Scene Selection

After class selection, source scenes are selected based on relevant object coverage.

Initial target:

```text
700–1,000 useful training tiles
```

The exact number may change based on actual class distribution.

The pipeline should avoid repeatedly processing the entire source dataset when only a controlled subset is required.

---

## 8.4 Scene-Level Split

Train, validation, and test splits must be made at the source-scene level.

```text
Source Scene A
 ├── Tile 1
 ├── Tile 2
 └── Tile 3
```

All tiles from the same source scene belong to the same split.

Recommended initial ratio:

```text
70% Train
15% Validation
15% Test
```

This reduces spatial leakage caused by similar neighbouring tiles appearing in different splits.

---

## 8.5 Tiling

Initial configuration:

```text
tile size: 1024 × 1024
overlap: 15%
```

For each tile:

1. determine whether an object intersects the tile;
2. transform the object's bounding box into tile coordinates;
3. clip the box to the tile boundary where required;
4. calculate the visible-object ratio;
5. discard objects below the configured visibility threshold.

Initial visibility threshold:

```text
30%
```

The threshold must be configurable and documented.

---

## 8.6 Negative Tiles

A controlled number of tiles containing no selected object should be retained.

Initial target:

```text
approximately 25% negative tiles
```

This prevents the detector from learning that every input necessarily contains a selected object.

---

## 8.7 Detection Label Format

Processed detection labels should use the format required by the selected training framework.

For YOLO-compatible training:

```text
class_id x_center y_center width height
```

Coordinates must be normalized to:

```text
0–1
```

---

# 9. Data Labels and Mappings

All label mappings must be explicit.

## 9.1 Segmentation Mapping

Stored in:

```text
data/metadata/class_mapping.json
```

The mapping identifies:

- class ID;
- class name;
- whether the class is ignored;
- optional semantic group.

Example groups:

```text
built_environment
water
vegetation
land_degradation
background
```

---

## 9.2 Detection Mapping

The detection mapping contains only classes selected for the MVP.

Example:

```json
{
  "detection": {
    "0": {
      "name": "selected_class_1"
    },
    "1": {
      "name": "selected_class_2"
    }
  }
}
```

The mapping is updated after class selection.

---

# 10. Feature Data Contract

Model outputs are converted into a stable feature representation before entering the risk engine.

```text
Segmentation Model
        │
        ▼
Segmentation Features
        │
        ├───────────────┐
        │               │
Detection Model         │
        │               │
        ▼               │
Detection Features      │
        │               │
        └───────┬───────┘
                ▼
         Spatial Features
                │
                ▼
         Feature Dictionary
                │
                ▼
           Risk Engine
```

---

## 10.1 Segmentation Features

Potential MVP features:

```text
building_area_ratio
road_area_ratio
water_area_ratio
barren_area_ratio
forest_area_ratio
agriculture_area_ratio
vegetation_area_ratio
segmentation_confidence
```

Where applicable:

```text
vegetation_area_ratio
=
forest_area_ratio
+
agriculture_area_ratio
```

---

## 10.2 Detection Features

Detection features depend on final selected classes.

Common features:

```text
total_detected_objects
object_density
mean_detection_confidence
```

Class-specific counts should only exist for selected classes.

---

## 10.3 Spatial Features

Initial candidates:

```text
building_water_proximity
built_up_density
water_adjacency_score
```

The implementation must document units and assumptions for every spatial feature.

---

# 11. Feature Schema

Stored in:

```text
data/metadata/feature_schema.json
```

Example:

```json
{
  "building_area_ratio": {
    "type": "float",
    "range": [0, 1],
    "source": "segmentation"
  },
  "water_area_ratio": {
    "type": "float",
    "range": [0, 1],
    "source": "segmentation"
  },
  "object_density": {
    "type": "float",
    "minimum": 0,
    "source": "detection"
  },
  "building_water_proximity": {
    "type": "float",
    "minimum": 0,
    "source": "spatial_analysis"
  }
}
```

The risk engine depends on this logical feature contract rather than model implementation details.

---

# 12. Confidence Handling

Visual outputs are not equally reliable in every image.

The pipeline should retain confidence information where available.

Examples:

```text
segmentation_confidence
detection_confidence
```

Confidence is supporting information and must not be confused with a direct hazard measurement.

---

# 13. Case Study Data Handling

Each case study has a stable identifier.

Recommended structure:

```text
data/raw/case_study/
└── case_001/
    ├── image.*
    └── metadata.json
```

Metadata may contain:

```text
case_id
location_name
latitude
longitude
image_date
source
resolution
notes
limitations
```

Fields should only be populated when supported by source data.

The system must not invent contextual variables such as historical hazard probability or damage likelihood.

Derived model outputs must remain distinguishable from externally sourced metadata.

---

# 14. Data Validation Requirements

Every prepared dataset must pass validation before training.

## Segmentation Validation

```text
✓ Image exists
✓ Mask exists
✓ Dimensions match
✓ Labels are valid
✓ Selected samples belong to the correct split
✓ Ignore labels are handled correctly
```

## Detection Validation

```text
✓ Image exists
✓ Label file is readable
✓ Bounding boxes are within image boundaries
✓ Normalized coordinates are valid
✓ Width and height are positive
✓ Class IDs exist in the selected class mapping
✓ No source-scene leakage exists across splits
```

---

# 15. Git and Storage Policy

The repository should contain:

```text
✓ preparation scripts
✓ configuration
✓ metadata schemas
✓ manifests
✓ documentation
✓ small representative samples
```

The repository should not contain:

```text
✗ full raw datasets
✗ large processed datasets
✗ bulk generated tiles
✗ temporary training artifacts
✗ model checkpoints unless intentionally released
```

The `.gitignore` must enforce this separation.

---

# 16. Configuration

Data-related parameters should be centralised.

Recommended location:

```text
configs/data.yaml
```

The configuration should include:

```text
random seed
dataset enablement
subset sizes
image size
ignore labels
selected detection classes
tiling parameters
visibility threshold
negative tile ratio
case-study configuration
```

Data decisions should not be scattered as hardcoded constants across training scripts.

---

# 17. Implementation Phases

## Phase 1 — Data Foundation

Deliverables:

- `data/` structure;
- `.gitignore` rules;
- `configs/data.yaml`;
- dataset source documentation;
- class mapping schema;
- feature schema.

## Phase 2 — Segmentation Data

Deliverables:

- dataset acquisition;
- validation;
- manifest generation;
- class distribution inspection;
- reproducible subset selection;
- working segmentation dataset loader.

Success condition:

```text
A selected image and mask can be loaded reproducibly and transformed into training-ready tensors.
```

## Phase 3 — Detection Data Investigation

Deliverables:

- annotation inspection;
- class frequency report;
- object size report;
- documented class selection.

No tiling or large-scale processing should occur before this phase is reviewed.

## Phase 4 — Detection Data Preparation

Deliverables:

- selected source scenes;
- scene-level split;
- tiled imagery;
- transformed bounding boxes;
- processed labels;
- negative samples;
- validation report.

## Phase 5 — Feature Integration

Deliverables:

- segmentation features;
- detection features;
- spatial features;
- feature schema validation;
- structured feature output for the risk engine.

---

# 18. Initial Command Flow

```text
1. Acquire Data
   ↓
python scripts/download_data.py

2. Inspect / Prepare Data
   ↓
python scripts/prepare_data.py

3. Train Segmentation
   ↓
python scripts/train_segmentation.py

4. Train Detection
   ↓
python scripts/train_detection.py

5. Evaluate
   ↓
python scripts/evaluate.py

6. Run End-to-End Demo
   ↓
python scripts/run_demo.py
```

The exact command arguments may evolve, but data responsibilities should remain consistent.

---

# 19. Current MVP Boundaries

The following are intentionally outside the initial data scope unless required later:

- full-dataset training;
- very large custom satellite imagery collections;
- manual annotation of a large new dataset;
- large-scale temporal sequence datasets;
- external contextual data not required for the MVP;
- automatic hazard truth labels derived without a documented source.

The MVP focuses on demonstrating a complete and reproducible pipeline rather than claiming comprehensive real-world hazard prediction.

---

# 20. Definition of Done for the Data Pipeline

The initial data implementation is complete when:

```text
✓ Dataset sources are documented.

✓ Raw data is separated from processed data.

✓ Raw data is not committed to Git.

✓ Dataset validation is implemented.

✓ A reproducible segmentation subset is selected.

✓ Dataset manifests record selected samples.

✓ Segmentation labels are explicitly mapped.

✓ Detection annotations are inspected before class selection.

✓ Detection classes are documented and reproducible.

✓ Detection splits prevent source-scene leakage.

✓ Processed detection labels are validated.

✓ Model outputs can be converted into structured features.

✓ Feature definitions are documented independently of model code.

✓ The risk engine receives structured features rather than raw model objects.

✓ A small case-study input can pass through the complete pipeline.
```

---

# 21. Key Architectural Rule

The data pipeline is not a separate experimental component.

It is the contract connecting:

```text
Datasets
    ↓
Computer Vision Models
    ↓
Feature Extraction
    ↓
Risk Assessment
    ↓
Explanation and Demo
```

Any future dataset, preprocessing step, or contextual data source should be added only when its role in this flow is explicit and when it does not break reproducibility or the MVP scope.
