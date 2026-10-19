import React, { useState } from 'react';
import type { RiskResponse } from '../types';
import IVPStageExplorer from './IVPStageExplorer';
import Visualizer from './Visualizer';
import ExecutiveReportModal from './ExecutiveReportModal';

interface DashboardProps {
  data: RiskResponse;
  onReset: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ data, onReset }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'ivp' | 'vision' | 'hazards' | 'report'>('overview');
  const [showReportModal, setShowReportModal] = useState<boolean>(false);

  const riskColor = data.risk_level === 'CRITICAL'
    ? 'rose'
    : data.risk_level === 'HIGH'
    ? 'amber'
    : data.risk_level === 'MODERATE'
    ? 'yellow'
    : 'emerald';

  const formatTitle = (str: string) => {
    return str.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div className="dashboard-wrapper">
      {/* Dashboard Top Header Bar */}
      <div className="dashboard-topbar glass-card">
        <div className="topbar-left">
          <button className="back-btn" onClick={onReset}>
            ← New Upload
          </button>
          <div className="location-info">
            <span className="location-label">ASSESSMENT TARGET</span>
            <h2 className="location-title">{data.location}</h2>
          </div>
          <span className="mode-badge">{data.analysis_mode}</span>
        </div>

        <div className="topbar-right">
          <div className={`score-badge ${riskColor}`}>
            <span className="score-num">{data.overall_risk}</span>
            <span className="score-max">/ 100</span>
            <strong className="risk-rank">{data.risk_level} THREAT</strong>
          </div>

          <button className="primary-btn report-launch-btn" onClick={() => setShowReportModal(true)}>
            📋 Open Printable Executive Report
          </button>
        </div>
      </div>

      {/* Navigation Tabs Bar */}
      <nav className="dashboard-nav-tabs glass-card">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📊 Executive Overview
        </button>

        <button
          className={`tab-btn ${activeTab === 'ivp' ? 'active' : ''}`}
          onClick={() => setActiveTab('ivp')}
        >
          🔬 IVP Preprocessing Breakdown ({data.ivp_stages.length} Steps)
        </button>

        <button
          className={`tab-btn ${activeTab === 'vision' ? 'active' : ''}`}
          onClick={() => setActiveTab('vision')}
        >
          🗺️ AI Vision & Layers
        </button>

        <button
          className={`tab-btn ${activeTab === 'hazards' ? 'active' : ''}`}
          onClick={() => setActiveTab('hazards')}
        >
          ⚠️ Multi-Hazard Analysis ({data.hazards.length})
        </button>

        <button
          className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`}
          onClick={() => setActiveTab('report')}
        >
          📋 Disaster Mitigation Plan
        </button>
      </nav>

      {/* Tab 1: Executive Overview */}
      {activeTab === 'overview' && (
        <div className="tab-content overview-tab">
          <div className="overview-grid">
            {/* Executive Summary Card */}
            <div className="overview-card summary-card glass-card">
              <span className="section-badge">EXECUTIVE SUMMARY</span>
              <h3>Multi-Hazard AI Assessment Summary</h3>
              <p className="summary-p">{data.explanation}</p>
              <div className="data-note-box">
                ℹ <strong>Pipeline Note:</strong> {data.data_note}
              </div>
            </div>

            {/* Overall Risk Score Dial Card */}
            <div className="overview-card gauge-card glass-card">
              <span className="section-badge">RISK ENGINE RATING</span>
              <div className="gauge-box">
                <div className={`gauge-ring ${riskColor}`}>
                  <span className="big-score">{data.overall_risk}</span>
                  <small>Score / 100</small>
                </div>
                <div className="gauge-status">
                  <span className={`status-pill ${riskColor}`}>{data.risk_level}</span>
                  <p>Derived via IVP feature fusion, proximity calculations, and land-cover proxies.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Land Cover & Spectral Features Stats */}
          <div className="features-overview-section glass-card">
            <div className="section-header">
              <div>
                <span className="section-badge">LAND COVER DISTRIBUTION</span>
                <h3>Estimated Surface Proxies</h3>
              </div>
            </div>

            <div className="feature-bars-grid">
              <div className="feature-bar-item">
                <div className="bar-info">
                  <span>Water Coverage</span>
                  <strong>{data.features.water_percentage ?? 0}%</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill water" style={{ width: `${Math.min(data.features.water_percentage ?? 0, 100)}%` }}></div>
                </div>
              </div>

              <div className="feature-bar-item">
                <div className="bar-info">
                  <span>Built-up Structure</span>
                  <strong>{data.features.building_percentage ?? 0}%</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill building" style={{ width: `${Math.min(data.features.building_percentage ?? 0, 100)}%` }}></div>
                </div>
              </div>

              <div className="feature-bar-item">
                <div className="bar-info">
                  <span>Road Infrastructure</span>
                  <strong>{data.features.road_percentage ?? 0}%</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill road" style={{ width: `${Math.min(data.features.road_percentage ?? 0, 100)}%` }}></div>
                </div>
              </div>

              <div className="feature-bar-item">
                <div className="bar-info">
                  <span>Vegetation Canopy</span>
                  <strong>{data.features.vegetation_percentage ?? 0}%</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill vegetation" style={{ width: `${Math.min(data.features.vegetation_percentage ?? 0, 100)}%` }}></div>
                </div>
              </div>

              <div className="feature-bar-item">
                <div className="bar-info">
                  <span>Exposed Bare Land</span>
                  <strong>{data.features.bare_land_percentage ?? 0}%</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill bare" style={{ width: `${Math.min(data.features.bare_land_percentage ?? 0, 100)}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: IVP Preprocessing Breakdown */}
      {activeTab === 'ivp' && (
        <div className="tab-content ivp-tab">
          <IVPStageExplorer
            stages={data.ivp_stages}
            metrics={data.ivp_metrics}
            originalImage={data.original_image_base64}
          />
        </div>
      )}

      {/* Tab 3: AI Model Vision & Layers */}
      {activeTab === 'vision' && (
        <div className="tab-content vision-tab">
          <Visualizer
            originalImg={data.original_image_base64}
            segMask={data.segmentation_mask_base64}
            detOverlay={data.detection_overlay_base64}
            earthvlHeatmap={data.earthvl_risk_heatmap_base64}
            ivpStages={data.ivp_stages}
          />
        </div>
      )}

      {/* Tab 4: Multi-Hazard Analysis */}
      {activeTab === 'hazards' && (
        <div className="tab-content hazards-tab">
          <div className="hazards-grid">
            {data.hazards.length > 0 ? (
              data.hazards.map(h => (
                <div key={h.type} className={`hazard-card glass-card severity-${h.severity.toLowerCase()}`}>
                  <div className="hazard-card-header">
                    <div>
                      <span className="hazard-type-pill">{h.type.replace('_', ' ').toUpperCase()}</span>
                      <h3>{formatTitle(h.type)}</h3>
                    </div>
                    <div className="hazard-score-badge">
                      <strong>{h.score}</strong>
                      <small>/ 100</small>
                    </div>
                  </div>

                  <div className="hazard-confidence-strip">
                    <span>Model Confidence: <strong>{Math.round(h.confidence * 100)}%</strong></span>
                    <span className={`severity-badge ${h.severity.toLowerCase()}`}>{h.severity}</span>
                  </div>

                  <div className="hazard-card-section">
                    <h4>Evidence Signals & Visual Observations</h4>
                    <ul>
                      {h.evidence.map((item, i) => (
                        <li key={i}>🔍 {item}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="hazard-card-section recommendations">
                    <h4>Actionable Technical Recommendation</h4>
                    <p>💡 {h.recommendations[0] || 'Conduct ground inspection of identified tile.'}</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="empty-hazards-box glass-card">
                <h3>No Critical Hazard Anomalies Detected</h3>
                <p>Derived land-cover proxies stay below critical threat thresholds. Continuous screening recommended.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 5: Disaster Mitigation Plan */}
      {activeTab === 'report' && (
        <div className="tab-content report-tab glass-card">
          <div className="report-tab-header">
            <div>
              <span className="section-badge">STRATEGIC DISASTER PLAN</span>
              <h2>{data.mitigation_report.disaster_type} Mitigation Report</h2>
              <p>{data.mitigation_report.summary}</p>
            </div>
            <button className="primary-btn" onClick={() => setShowReportModal(false)}>
              🖨 Launch Printable Version
            </button>
          </div>

          <div className="mitigation-tiers-preview">
            <div className="preview-tier immediate">
              <span className="tier-badge">⚡ 0 - 48 Hours</span>
              <h4>Immediate Tactical Actions</h4>
              <ul>
                {data.mitigation_report.immediate_actions.map((act, i) => <li key={i}>{act}</li>)}
              </ul>
            </div>

            <div className="preview-tier midterm">
              <span className="tier-badge">🏗️ 1 - 6 Months</span>
              <h4>Mid-Term Engineering & Infrastructure</h4>
              <ul>
                {data.mitigation_report.mid_term_actions.map((act, i) => <li key={i}>{act}</li>)}
              </ul>
            </div>

            <div className="preview-tier longterm">
              <span className="tier-badge">🌿 1 - 5 Years</span>
              <h4>Long-Term Policy & Restoration</h4>
              <ul>
                {data.mitigation_report.long_term_actions.map((act, i) => <li key={i}>{act}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Report Modal */}
      {showReportModal && (
        <ExecutiveReportModal
          location={data.location}
          report={data.mitigation_report}
          hazards={data.hazards}
          features={data.features}
          metrics={data.ivp_metrics}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
};

export default Dashboard;
