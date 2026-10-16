from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class IVPStageOutput(BaseModel):
    id: str
    title: str
    stage_name: str
    category: str  # Radiometric, Spectral, Noise, Edge, Frequency, Morphology, Threshold
    image_base64: str
    summary: str
    technical_detail: str
    disaster_relevance: str


class IVPMetrics(BaseModel):
    image_dimensions: List[int]
    mean_intensity: float
    std_intensity: float
    image_entropy: float
    canny_edge_density: float
    tile_count: int
    channel_mean: List[float]
    channel_std: List[float]


class Hazard(BaseModel):
    type: str
    score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    severity: str  # CRITICAL, HIGH, MODERATE, LOW
    evidence: List[str]
    recommendations: List[str]


class MitigationTier(BaseModel):
    timeframe: str
    title: str
    actions: List[str]


class DisasterMitigationReport(BaseModel):
    disaster_type: str
    risk_level: str
    overall_score: int
    summary: str
    vulnerability_factors: List[str]
    immediate_actions: List[str]  # 0-48 hrs
    mid_term_actions: List[str]   # 1-6 months
    long_term_actions: List[str]  # 1-5 years


class RiskResponse(BaseModel):
    location: str
    overall_risk: int = Field(ge=0, le=100)
    risk_level: str
    hazards: List[Hazard]
    explanation: str
    features: Dict[str, float]
    ivp_metrics: IVPMetrics
    ivp_stages: List[IVPStageOutput]
    original_image_base64: Optional[str] = None
    segmentation_mask_base64: Optional[str] = None
    detection_overlay_base64: Optional[str] = None
    earthvl_risk_heatmap_base64: Optional[str] = None
    mitigation_report: DisasterMitigationReport
    analysis_mode: str
    data_note: str

