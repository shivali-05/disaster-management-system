You are an expert AI software architect, ML engineer, remote-sensing engineer, full-stack developer, DevOps engineer, and academic project mentor.

You are working on a college project titled:

**AI-Based Multi-Hazard Risk and Infrastructure Assessment using Satellite Imagery**

This is a two-person IVP + AI project. The goal is to build a **real, runnable prototype**, not merely a research/demo UI.

The project concept is grounded in our existing project document. It proposes a modular pipeline:

**Satellite / Remote-Sensing Imagery → Metadata → Preprocessing → Semantic Segmentation → Object Detection → Hazard Feature Extraction → Geographic/Historical Context → Risk Fusion → Risk Score & Ranking → Visualization → Explanation → Periodic Reassessment**

The project document specifically identifies the research gap as combining normally separate capabilities—segmentation, detection, contextual information, risk scoring, ranking and explanation—into one modular risk-assessment system. fileciteturn0file0L401-L408

The project should remain academically defensible and must NOT falsely claim that it can detect every possible disaster or that satellite imagery alone proves real-world risk. Risk must be presented as an **AI-derived decision-support estimate with confidence and contextual evidence**. fileciteturn0file0L573-L584

# PRIMARY OBJECTIVE

Build a working prototype that accepts a satellite/remote-sensing image or geospatial image patch and produces:

1. Semantic segmentation of important land/infrastructure classes.
2. Object/infrastructure detection.
3. Extraction of measurable risk indicators.
4. Context-aware hazard/risk estimation.
5. A configurable and interpretable risk score.
6. Ranked risk zones.
7. Visual overlays showing exactly why an area was flagged.
8. Natural-language explanation/recommendation.
9. Optional current-vs-previous imagery comparison.
10. A dashboard/API demonstrating the entire pipeline.

The system must be designed so additional hazard modules can later be added without rewriting the entire application.

Do NOT attempt to train separate deep-learning models for every conceivable disaster. That would make the project unrealistic for a college-scale implementation.

Instead, implement a **strong multi-hazard prototype around a limited number of defensible risk modules**, while architecting the system so more modules can be added later.

# IMPORTANT SCOPE DECISION

Use the following MVP scope unless a technical problem makes one component impossible:

### Core AI components

**A. Semantic segmentation**
Use **LoveDA** as the primary dataset.

Initial classes should focus on meaningful environmental/infrastructure structure such as:

- buildings
- roads
- vegetation
- water
- bare land/background classes supported by the dataset

Use a practical baseline such as:

- U-Net

or

- DeepLab-family model

The project document already identifies semantic segmentation with LoveDA as the primary baseline. fileciteturn0file0L442-L449

### B. Object detection

Use **xView** or a small compatible subset.

Use a YOLO-family detector where practical.

Do NOT attempt to reproduce the entire xView dataset or every class.

Select a small subset of infrastructure-relevant object classes that can actually contribute to the risk engine.

The project document identifies xView for overhead object detection. fileciteturn0file0L520-L527

### C. Contextual risk engine

Build this ourselves.

This is the component that makes the project different from simply running segmentation or YOLO.

The engine should combine:

- segmentation evidence
- detected infrastructure
- proximity to water
- road/building density
- vegetation/bare-land patterns
- terrain/elevation when available
- historical hazard information when available
- model confidence
- optional temporal change indicators

The project document explicitly defines risk fusion as combining hazard evidence, exposure/context, historical information and model confidence. fileciteturn0file0L430-L435

### D. Hazard modules

Implement a small number of practical hazard/risk indicators.

At minimum implement:

**1. Flood susceptibility / water-related risk**

Possible signals:

- proximity to detected water bodies
- low-lying/topographic information when available
- water expansion/change between two images
- built-up exposure near water
- historical flood information where accessible

This should be called something such as:

**Flood Susceptibility / Flood-Exposure Risk**

Do NOT claim exact flood prediction unless actual temporal/weather/hydrological inputs support that claim.

**2. Infrastructure / urban risk**

Signals may include:

- building density
- road density
- infrastructure proximity
- unusual development patterns
- change in built-up area
- structures close to water
- insufficient surrounding open/green area
- detected infrastructure concentration

**3. Land degradation / vegetation-related risk**

Signals may include:

- vegetation loss/change
- bare-land expansion
- unusual land-cover transitions
- exposure of developed areas to degraded land

**4. Change detection**

If feasible, compare current and older imagery.

Detect approximate changes in:

- buildings
- roads
- vegetation
- water
- bare land

The project document lists bi-temporal comparison as an optional but valuable component. fileciteturn0file0L447-L449

Do not pretend that a simple image difference is equivalent to professional change-detection research. Clearly label it as a prototype change indicator.

# WHAT NOT TO DO

Do NOT:

- download dozens of datasets
- train huge foundation models
- attempt to model every disaster
- build a fake "live satellite" system
- fabricate model accuracy
- fabricate historical disaster data
- fabricate test results
- claim medical/fire/earthquake/industrial predictions that the model cannot actually support
- hardcode fake AI outputs just to make the dashboard look impressive
- use an LLM as a replacement for actual computer-vision inference
- build only frontend screens without a functioning ML pipeline

The project document explicitly warns against calling imagery “live” unless the selected provider actually supplies sufficiently frequent imagery and access. fileciteturn0file0L573-L575

# DATASET STRATEGY

Keep datasets under control.

Primary datasets:

### Dataset 1 — LoveDA
Purpose:
- semantic segmentation
- urban/rural comparison
- land-cover/infrastructure extraction

### Dataset 2 — xView
Purpose:
- object detection
- infrastructure/object evidence

### Dataset 3 — One geospatial source for realistic case studies

Prefer one practical source from:

- Bhuvan / NRSC
- Sentinel Hub
- NASA Earthdata

Do not integrate all providers into the MVP.

The project document already lists these as possible remote-sensing sources. fileciteturn0file0L520-L550

### Optional supporting data

Use elevation/topographic or historical hazard information only if it can be retrieved reliably and legally.

Do not allow external-data integration to become the main project.

### BigEarthNet

Use BigEarthNet primarily as a research/reference dataset for land-cover/scene understanding.

Do NOT force it into the MVP if integration makes the project unnecessarily complicated.

### EarthVQA / EarthVL

Treat these as research/reference resources.

Do NOT spend most of the project training them.

For the MVP, natural-language explanation can use our own structured evidence → explanation layer.

The project document specifically describes EarthVQA as a later-stage reasoning resource rather than the complete risk engine. fileciteturn0file0L195-L230

# CRITICAL IMPLEMENTATION PRINCIPLE

The system must separate:

### 1. Visual evidence

Example:

"Water body detected 82 m from built-up area."

### 2. Contextual inference

Example:

"High building exposure near water increases flood exposure."

### 3. Risk interpretation

Example:

"Flood exposure risk: HIGH."

### 4. Confidence

Example:

"Model confidence: 0.84."

Never mix these together as though the neural network directly knows the final disaster risk.

# RISK ENGINE

Build an interpretable configurable scoring engine.

Example conceptual structure:

Risk Score =
weighted hazard evidence
+ exposure
+ contextual/geographic factors
+ historical factor
+ temporal-change factor
− uncertainty penalty

Normalize to:

0–100.

Example:

0–20 = Very Low
21–40 = Low
41–60 = Moderate
61–80 = High
81–100 = Critical

Do NOT arbitrarily claim these thresholds are scientifically universal.

Document them as configurable prototype thresholds.

Every final risk score should expose its component contributions.

Example:

Flood Exposure Risk: 76/100

Contributors:

Water proximity: +21
Building exposure: +22
Low elevation: +15
Historical flood indicator: +10
Model confidence: +8

This is much more defensible than simply saying:

"AI predicts flood."

# RISK OUTPUT

The system should return something structurally similar to:

```json
{
  "location": "...",
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

# VISUAL OUTPUT

The application must show:

1. Original image
2. Segmentation mask
3. Segmentation overlay
4. Object-detection boxes
5. Detected risk zones
6. Risk heatmap
7. Risk score
8. Hazard ranking
9. Evidence
10. Confidence
11. Recommendations
12. Optional before/after image comparison

A user should be able to look at the dashboard and visually understand:

**"This location was flagged because these specific things were detected."**

# DASHBOARD

Build a professional working dashboard.

Recommended stack:

### Backend

Python + FastAPI

### ML

PyTorch

### Frontend

React + TypeScript

or another stack only if it significantly simplifies implementation.

### Visualization

Use a map/image visualization library appropriate for the application.

The interface should support:

- upload image
- select analysis mode
- run analysis
- display progress/status
- show AI outputs
- display risk ranking
- inspect individual hazards
- inspect evidence
- compare images
- export analysis result

Do not overbuild authentication, billing, cloud infrastructure, multi-user SaaS functionality, etc.

This is a college research prototype.

# REPOSITORY STRUCTURE

Create something close to:

```text
multi-hazard-risk-ai/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── docker-compose.yml
│
├── docs/
│   ├── PROJECT_MASTER.md
│   ├── ARCHITECTURE.md
│   ├── DATASETS.md
│   ├── MODEL_SELECTION.md
│   ├── RISK_ENGINE.md
│   ├── API.md
│   ├── EXPERIMENTS.md
│   ├── VALIDATION.md
│   ├── LIMITATIONS.md
│   ├── IVP_MAPPING.md
│   ├── TASK_ASSIGNMENT.md
│   └── PROGRESS_LOG.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── samples/
│   └── metadata/
│
├── notebooks/
│
├── models/
│   ├── segmentation/
│   ├── detection/
│   └── change_detection/
│
├── src/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── detection/
│   ├── change_detection/
│   ├── feature_extraction/
│   ├── risk_engine/
│   ├── explanation/
│   ├── geospatial/
│   └── evaluation/
│
├── backend/
│   ├── main.py
│   ├── api/
│   ├── schemas/
│   └── services/
│
├── frontend/
│
├── tests/
│
└── scripts/
    ├── download_data.py
    ├── prepare_data.py
    ├── train_segmentation.py
    ├── train_detection.py
    ├── evaluate.py
    └── run_demo.py
```

Use a cleaner structure where appropriate, but preserve the architectural separation between components.

# DOCUMENTATION REQUIREMENT

Before doing substantial implementation, create:

## docs/PROJECT_MASTER.md

This must become the authoritative implementation document.

It must contain:

1. Problem statement
2. Motivation
3. Research gap
4. Project objectives
5. Scope
6. Non-goals
7. System architecture
8. Data pipeline
9. Model pipeline
10. Risk engine
11. Dataset justification
12. Model justification
13. Evaluation methodology
14. Expected outputs
15. Limitations
16. Future scope
17. Reproducibility instructions
18. Academic justification
19. Demo scenario
20. Division of work

# IVP REQUIREMENTS

Treat IVP as a real engineering/research component, not something added at the end.

Map the implementation to the existing methodology:

1. Problem Identification
2. Requirement Analysis
3. Data Collection / Dataset Selection
4. Data Preprocessing
5. Feature Extraction / Model Development
6. Proposed Algorithm / System Development
7. Training / Implementation
8. Testing & Validation
9. Performance Evaluation
10. Results & Comparison
11. Conclusion & Future Scope

This exact methodology exists in the current project document. fileciteturn0file0L411-L422

Create:

`docs/IVP_MAPPING.md`

Map each software/ML component to:

- IVP concept
- AI concept
- implementation
- evidence/screenshots/results required
- report section
- presentation/demo relevance

Include methodology, experimentation, algorithm design, model evaluation, system implementation, research gap, limitations and future work.

# EXPERIMENTS

The project must produce real experimental evidence.

At minimum run:

### Experiment 1
LoveDA segmentation baseline.

Report:

- train loss
- validation loss
- IoU
- Dice/F1
- pixel accuracy
- per-class metrics

### Experiment 2
xView/object detector baseline.

Report:

- precision
- recall
- mAP@0.5
- mAP@0.5:0.95

### Experiment 3
Risk engine.

Compare several risk-weight configurations and inspect their outputs.

### Experiment 4
Urban vs rural validation where supported.

### Experiment 5
Optional temporal/change detection.

The project's predefined metrics include segmentation mIoU/IoU/Dice/F1/pixel accuracy, detection mAP/precision/recall, classification macro-F1 and risk prioritization metrics. fileciteturn0file0L451-L457

Never fabricate these values.

If training is incomplete, clearly record:

`NOT YET RUN`

instead of inventing numbers.

# TWO-PERSON TASK DIVISION

There are exactly two people.

Create a balanced division of work.

Do NOT divide it as:

Person A = AI
Person B = frontend

That produces unequal technical ownership.

Instead divide the project into two major streams with shared integration.

## PERSON A — AI / DATA / RISK ENGINE

Own:

- dataset preparation
- preprocessing
- LoveDA segmentation
- xView detection
- feature extraction
- geospatial features
- risk engine
- change detection
- model evaluation
- experimentation
- ML documentation

## PERSON B — SYSTEM / PRODUCT / VALIDATION

Own:

- backend
- API
- dashboard
- visualization
- inference orchestration
- explanation system
- result storage
- case-study workflow
- system testing
- UX
- deployment
- documentation/presentation assets

## SHARED

Both people must contribute to:

- architecture
- literature review
- model selection
- validation
- integration
- final testing
- report
- presentation
- demo

Create a detailed task table containing:

- task ID
- person
- task
- deliverable
- dependency
- estimated difficulty
- completion status

The project already uses a two-person research/development workflow and identifies parallel responsibilities for segmentation/dataset work, detection, risk design, visualization, integration and validation. fileciteturn0file0L458-L519

# DEVELOPMENT ORDER

Do not build everything simultaneously.

Implement in this order:

## Phase 0 — Planning

Create:

- PROJECT_MASTER.md
- architecture
- task division
- dataset plan
- environment setup
- experiment plan

## Phase 1 — Data

Implement:

- dataset download/verification
- preprocessing
- tiling
- normalization
- train/validation/test splitting
- sample visualization

Ensure geographic/spatial leakage is minimized.

## Phase 2 — Segmentation

Train/test LoveDA segmentation baseline.

Save:

- checkpoint
- metrics
- predictions
- sample visualizations

## Phase 3 — Detection

Train/test xView-compatible detector.

Save:

- weights
- metrics
- predictions
- visualizations

## Phase 4 — Feature extraction

Convert model outputs into measurable risk indicators.

Examples:

- building density
- road density
- water proximity
- built-up area percentage
- vegetation percentage
- bare land percentage
- detected object counts
- change percentages
- confidence

## Phase 5 — Risk engine

Implement configurable risk rules.

Write unit tests.

## Phase 6 — Realistic case study

Use one accessible Indian region/case study.

Do not claim comprehensive nationwide coverage.

## Phase 7 — Backend

Expose APIs.

Example:

```text
POST /analyze
POST /segment
POST /detect
POST /risk
POST /compare
GET  /health
GET  /model-info
```

## Phase 8 — Dashboard

Build the frontend.

## Phase 9 — Explanation

Generate explanations from structured evidence.

Prefer deterministic/template-based explanations first.

An LLM can optionally improve language later, but the explanation must always be grounded in actual computed evidence.

## Phase 10 — Testing

Test:

- image validation
- model loading
- API
- risk calculation
- visualization
- invalid input
- low-confidence predictions
- missing metadata
- different image sizes

## Phase 11 — Final demo

Create an end-to-end demonstration:

**Input image → AI analysis → visual outputs → hazard scores → ranking → recommendation**

# CONTINUOUS / UPDATED ASSESSMENT

Implement the architecture for periodic reassessment, but don't fake real-time satellite imagery.

The system should support:

```text
Image A
     ↓
Analysis A
     ↓
Image B
     ↓
Analysis B
     ↓
Change Detection
     ↓
Risk Update
```

Display:

- previous risk
- current risk
- change
- reason for change

Call this:

**Periodic Reassessment**

rather than:

**Real-Time Disaster Prediction**

unless an actual provider/API makes the stronger claim technically valid.

# DATASET AUTOMATION

Create scripts so a fresh developer can reproduce the setup.

Example:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_segmentation.py
python scripts/train_detection.py
python scripts/evaluate.py
python scripts/run_demo.py
```

Clearly document which datasets require:

- manual download
- credentials
- API keys
- registration
- accepting terms
- special preprocessing

Never place secrets in Git.

# HARDWARE AWARENESS

Detect available hardware.

If CUDA GPU exists:

- use CUDA
- use mixed precision where safe
- configure batch size automatically

If no GPU:

- allow reduced CPU demonstration mode.

Do not assume a specific GPU.

# ENGINEERING QUALITY

Write:

- type-safe code where applicable
- modular classes/functions
- configuration files
- logging
- error handling
- reproducible seeds
- unit tests
- model/version metadata
- clear README instructions

Avoid massive monolithic Python files.

# MODEL CHECKPOINT MANAGEMENT

Never commit huge model files to Git unless genuinely small.

Use:

```text
models/
checkpoints/
```

with appropriate `.gitignore`.

Document how checkpoints are obtained or generated.

# SECURITY

Audit all dependencies.

Do NOT:

- execute arbitrary downloaded shell scripts
- curl | bash
- run unknown binaries
- blindly install random GitHub repositories
- hardcode credentials
- commit API keys
- disable SSL verification
- trust unverified model weights

Pin or constrain important Python dependencies where practical.

# ACADEMIC REPRODUCIBILITY

For every experiment record:

- dataset version/source
- selected subset
- preprocessing
- train/validation/test split
- hyperparameters
- model architecture
- hardware
- training epochs
- random seed
- metrics
- sample outputs
- failure cases

Create:

`docs/EXPERIMENTS.md`

with tables.

# VALIDATION

The validation strategy must explicitly address:

- false positives
- false negatives
- urban/rural differences
- low-resolution imagery
- clouds/quality issues
- small objects
- spatial leakage
- domain shift
- model confidence

The existing project documentation specifically identifies urban/rural domain shift, small infrastructure objects, spatial leakage, uncertainty and the need for confidence information as limitations. fileciteturn0file0L61-L71

# LIMITATIONS

Create a serious limitations section.

Include statements such as:

- satellite imagery may not have sufficient temporal resolution
- not every hazard is visually observable
- geographic context may be incomplete
- historical data may be unavailable or inconsistent
- predictions are not ground truth
- risk scoring is configurable rather than universally authoritative
- high-risk outputs require human/official verification
- model performance may vary by geography
- imagery resolution affects small-object detection
- cloud/atmospheric conditions can affect imagery

Do not hide limitations for the sake of a better-looking demo.

# DEMO DESIGN

Create one compelling end-to-end scenario.

Example:

### Scenario

A selected urban/peri-urban region containing:

- road network
- dense buildings
- vegetation
- water body

System performs:

1. segmentation
2. object detection
3. water/building proximity analysis
4. infrastructure exposure calculation
5. risk scoring
6. map visualization
7. explanation

Output:

**Overall Risk: HIGH**

Then show:

**Top risk contributors**

and visualize the exact regions.

Then compare an older image with a newer image.

This gives the project a strong "before → AI → decision" flow.

# UI INFORMATION ARCHITECTURE

Dashboard should have:

### Overview

- selected location
- overall risk
- confidence
- top hazards

### Visual Analysis

- source imagery
- segmentation
- detections
- heatmap

### Hazard Breakdown

- flood exposure
- infrastructure risk
- vegetation/land degradation
- change detection

### Evidence

Show measurable indicators.

### Recommendations

Show practical mitigation/inspection suggestions.

### Comparison

Previous vs current.

### Model information

- model
- dataset
- confidence
- version

# EXPLANATION SYSTEM

Use structured templates.

For example:

```text
HIGH FLOOD-EXPOSURE RISK

Reason:
• Dense built-up area was detected near a mapped water body.
• Water coverage increased compared with the previous image.
• A high proportion of infrastructure lies within the defined exposure zone.

Confidence:
0.82

Recommended action:
Prioritize drainage and water-channel inspection and manually verify exposed structures.
```

The explanation must be generated from actual pipeline outputs.

Never let a language model invent reasons.

# RESEARCH DIFFERENTIATION

The project must explicitly explain:

Existing systems often solve one task:

- segmentation
- object detection
- flood detection
- damage detection
- land-cover classification

Our project combines these into:

**evidence → context → risk fusion → ranking → explanation**

This is the project's central contribution.

The literature review in the supplied project document already establishes that these individual capabilities exist but are generally presented separately. fileciteturn0file0L401-L408

# CODEX EXECUTION RULES

Act as the implementation agent.

Do not merely generate suggestions.

First inspect the repository.

Then:

1. Create `docs/PROJECT_MASTER.md`.
2. Create `docs/TASK_ASSIGNMENT.md`.
3. Create `docs/ARCHITECTURE.md`.
4. Create the repository/module structure.
5. Create environment/config files.
6. Implement data pipeline.
7. Implement segmentation baseline.
8. Implement detection baseline.
9. Implement feature extraction.
10. Implement risk engine.
11. Implement backend/API.
12. Implement frontend.
13. Implement explanation layer.
14. Implement testing.
15. Run the system.
16. Fix errors.
17. Generate sample outputs.
18. Update documentation with real results.
19. Create final reproducibility instructions.

Do not pretend a component is complete if it was not successfully executed.

# AUTONOMOUS DECISION MAKING

You are authorized to make reasonable engineering decisions without asking unnecessary questions.

When there are multiple possible technologies:

1. Prefer simpler.
2. Prefer open-source.
3. Prefer reproducible.
4. Prefer locally runnable.
5. Prefer academically defensible.
6. Prefer fewer dependencies.
7. Prefer a working baseline over an ambitious unfinished implementation.

# WHEN A DATASET IS TOO LARGE

Do not stop development.

Create a reproducible subset.

Document:

```text
Dataset:
LoveDA

Subset:
N images / tiles

Reason:
Training and development constraints

Selection:
deterministic/random seed

Purpose:
baseline demonstration
```

The project document's current next-step plan explicitly recommends beginning with a small reproducible LoveDA or xView subset before expanding. fileciteturn0file0L585-L593

# WHEN TRAINING IS TOO EXPENSIVE

Use:

- reduced image size
- small subset
- pretrained backbone
- fewer epochs
- mixed precision
- cached preprocessing

But still produce real predictions and evaluation.

Document the trade-off.

# FINAL DELIVERABLES

At the end, the repository must contain:

### Software

- working ML pipeline
- working risk engine
- working backend
- working dashboard
- reproducible scripts
- tests

### AI artifacts

- model checkpoints/configuration
- training/evaluation metrics
- segmentation predictions
- object detections
- risk outputs
- sample visualizations

### IVP artifacts

- problem statement
- objectives
- literature review
- methodology
- architecture
- algorithm
- experimentation
- validation
- results
- limitations
- conclusion
- future scope
- work log
- task division

### Documentation

- README
- master project document
- architecture document
- dataset document
- model-selection document
- risk-engine document
- API document
- validation document
- IVP mapping
- experiment report
- progress log

# FINAL README MUST EXPLAIN

1. What problem this solves.
2. Why existing single-hazard approaches are insufficient.
3. What makes this system different.
4. Architecture.
5. Datasets.
6. Models.
7. Setup.
8. Training.
9. Running inference.
10. Running dashboard.
11. Running tests.
12. Example output.
13. Metrics.
14. Limitations.
15. Future work.

# FINAL CHECKLIST

Before declaring the project complete, verify:

[ ] Repository runs from a clean environment
[ ] Data pipeline works
[ ] Segmentation model runs
[ ] Detection model runs
[ ] Risk engine produces real scores
[ ] Risk scores expose evidence
[ ] Dashboard works
[ ] API works
[ ] Visual overlays work
[ ] Confidence is shown
[ ] At least one Indian case study is implemented
[ ] No fabricated metrics
[ ] No fabricated data
[ ] No fake live-satellite claim
[ ] Tests pass
[ ] Documentation matches actual implementation
[ ] IVP methodology is mapped
[ ] Two-person work division is documented
[ ] Final demo can be executed from the README

# MOST IMPORTANT PRIORITY

Do not build an enormous "AI that predicts every disaster."

Build a **small but genuine multi-hazard risk-assessment system** whose strongest contribution is the fusion layer:

**Satellite imagery → AI visual evidence → geographic/contextual evidence → interpretable risk score → ranked hazards → visual explanation → periodic reassessment**

The project must be impressive because it is **actually integrated and demonstrable**, not because it contains dozens of unfinished models.

At every stage, prefer a working implementation over theoretical complexity.