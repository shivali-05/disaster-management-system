export interface IVPStageOutput {
  id: string;
  title: string;
  stage_name: string;
  category: string;
  image_base64: string;
  summary: string;
  technical_detail: string;
  disaster_relevance: string;
}

export interface IVPMetrics {
  image_dimensions: number[];
  mean_intensity: number;
  std_intensity: number;
  image_entropy: number;
  canny_edge_density: number;
  tile_count: number;
  channel_mean: number[];
  channel_std: number[];
}

export interface Hazard {
  type: string;
  score: number;
  confidence: number;
  severity: string;
  evidence: string[];
  recommendations: string[];
}

export interface DisasterMitigationReport {
  disaster_type: string;
  risk_level: string;
  overall_score: number;
  summary: string;
  vulnerability_factors: string[];
  immediate_actions: string[];
  mid_term_actions: string[];
  long_term_actions: string[];
}

export interface RiskResponse {
  location: string;
  overall_risk: number;
  risk_level: string;
  hazards: Hazard[];
  explanation: string;
  features: Record<string, number>;
  ivp_metrics: IVPMetrics;
  ivp_stages: IVPStageOutput[];
  original_image_base64?: string;
  segmentation_mask_base64?: string;
  detection_overlay_base64?: string;
  earthvl_risk_heatmap_base64?: string;
  mitigation_report: DisasterMitigationReport;
  analysis_mode: string;
  data_note: string;
}
