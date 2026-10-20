import React from 'react';
import type { DisasterMitigationReport, Hazard, IVPMetrics } from '../types';

interface ExecutiveReportModalProps {
  location: string;
  report: DisasterMitigationReport;
  hazards: Hazard[];
  features: Record<string, number>;
  metrics: IVPMetrics;
  onClose: () => void;
}

export const ExecutiveReportModal: React.FC<ExecutiveReportModalProps> = ({
  location,
  report,
  hazards,
  features,
  metrics,
  onClose,
}) => {
  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="modal-backdrop print-modal-backdrop" onClick={onClose}>
      <div className="report-modal-container glass-card printable-area" onClick={e => e.stopPropagation()}>
        {/* Top bar (Hidden on Print) */}
        <div className="report-modal-header no-print">
          <div>
            <span className="badge-official">OFFICIAL DECISION SUPPORT</span>
            <h2>Disaster Risk & Mitigation Executive Report</h2>
          </div>
          <div className="header-actions">
            <button className="primary-btn print-btn" onClick={handlePrint}>
              🖨 Print / Export PDF
            </button>
            <button className="icon-btn close-btn" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Official Letterhead Header for Print/View */}
        <div className="report-doc-header">
          <div className="doc-brand">
            <span className="brand-icon">◈</span>
            <div>
              <h1 className="doc-title">TerraRisk Multi-Hazard Intelligence</h1>
              <span className="doc-sub">Remote Sensing IVP & Neural Assessment Engine</span>
            </div>
          </div>
          <div className="doc-meta">
            <div><strong>Location / Sample:</strong> {location}</div>
            <div><strong>Generated Date:</strong> {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</div>
            <div><strong>Spatial Entropy:</strong> {metrics.image_entropy} bits ({metrics.image_dimensions[1]}x{metrics.image_dimensions[0]})</div>
            <div><strong>Report ID:</strong> TR-RS-{Math.floor(100000 + Math.random() * 900000)}</div>
          </div>
        </div>

        {/* Executive Summary Banner */}
        <div className={`report-summary-banner severity-${report.risk_level.toLowerCase()}`}>
          <div className="score-dial">
            <div className="dial-value">{report.overall_score}</div>
            <div className="dial-max">/ 100</div>
          </div>
          <div className="banner-details">
            <div className="banner-top">
              <span className={`risk-pill ${report.risk_level.toLowerCase()}`}>
                {report.risk_level} RISK THREAT
              </span>
              <span className="disaster-tag">{report.disaster_type}</span>
            </div>
            <p className="summary-text">{report.summary}</p>
          </div>
        </div>

        {/* Vulnerability & Key Metrics Section */}
        <div className="report-section">
          <h3 className="section-heading">1. Spatial Vulnerability & Land-Cover Profile</h3>
          
          <div className="vuln-grid">
            <div className="vuln-factors-card">
              <h4>Identified Vulnerability Drivers</h4>
              <ul>
                {report.vulnerability_factors.map((factor, i) => (
                  <li key={i}>⚠️ {factor}</li>
                ))}
              </ul>
            </div>

            <div className="landcover-table-card">
              <h4>Land-Cover Coverage Breakdown</h4>
              <table className="report-table">
                <thead>
                  <tr>
                    <th>Class Proxy</th>
                    <th>Coverage (%)</th>
                    <th>Risk Signal</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Building Built-up</td>
                    <td><strong>{features.building_percentage ?? 0}%</strong></td>
                    <td>{features.building_percentage > 20 ? 'High Exposure' : 'Normal'}</td>
                  </tr>
                  <tr>
                    <td>Water Surfaces</td>
                    <td><strong>{features.water_percentage ?? 0}%</strong></td>
                    <td>{features.water_percentage > 10 ? 'Flood Buffer / Exposure' : 'Standard'}</td>
                  </tr>
                  <tr>
                    <td>Road Infrastructure</td>
                    <td><strong>{features.road_percentage ?? 0}%</strong></td>
                    <td>Access Arterial</td>
                  </tr>
                  <tr>
                    <td>Vegetation Canopy</td>
                    <td><strong>{features.vegetation_percentage ?? 0}%</strong></td>
                    <td>{features.vegetation_percentage < 15 ? 'Low Buffer' : 'Good Canopy'}</td>
                  </tr>
                  <tr>
                    <td>Exposed Bare Land</td>
                    <td><strong>{features.bare_land_percentage ?? 0}%</strong></td>
                    <td>{features.bare_land_percentage > 20 ? 'Erosion Susceptible' : 'Stable Ground'}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Hazards & Evidence Section */}
        <div className="report-section">
          <h3 className="section-heading">2. Multi-Hazard Threat Evidence Matrix</h3>
          
          <div className="hazards-report-list">
            {hazards.length > 0 ? (
              hazards.map(h => (
                <div key={h.type} className="hazard-report-item">
                  <div className="h-header">
                    <h4>{h.type.replace('_', ' ').toUpperCase()}</h4>
                    <span className="h-score">Severity: {h.score}/100</span>
                  </div>
                  <div className="h-evidence">
                    <strong>Observed Visual & Spatial Signals:</strong>
                    <ul>
                      {h.evidence.map((ev, i) => <li key={i}>• {ev}</li>)}
                    </ul>
                  </div>
                </div>
              ))
            ) : (
              <p className="clean-state">No threshold-triggered hazard anomalies detected in this area patch.</p>
            )}
          </div>
        </div>

        {/* Multi-Tier Mitigation Action Plan */}
        <div className="report-section">
          <h3 className="section-heading">3. Comprehensive Mitigation & Action Strategy</h3>
          
          <div className="mitigation-tiers-grid">
            {/* Tier 1: Immediate */}
            <div className="tier-card immediate">
              <div className="tier-header">
                <span className="tier-time">⚡ 0 - 48 HOURS</span>
                <h4>Immediate Tactical Emergency Actions</h4>
              </div>
              <ul className="action-list">
                {report.immediate_actions.map((act, i) => (
                  <li key={i}>{act}</li>
                ))}
              </ul>
            </div>

            {/* Tier 2: Mid-Term */}
            <div className="tier-card midterm">
              <div className="tier-header">
                <span className="tier-time">🏗️ 1 - 6 MONTHS</span>
                <h4>Mid-Term Infrastructure & Engineering</h4>
              </div>
              <ul className="action-list">
                {report.mid_term_actions.map((act, i) => (
                  <li key={i}>{act}</li>
                ))}
              </ul>
            </div>

            {/* Tier 3: Long-Term */}
            <div className="tier-card longterm">
              <div className="tier-header">
                <span className="tier-time">🌿 1 - 5 YEARS</span>
                <h4>Long-Term Policy & Ecological Restoration</h4>
              </div>
              <ul className="action-list">
                {report.long_term_actions.map((act, i) => (
                  <li key={i}>{act}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Report Footer / Signature Area */}
        <div className="report-doc-footer">
          <div className="disclaimer">
            <p><strong>Notice:</strong> This report is derived from remote sensing image preprocessing, spectral index calculation, and automated decision-support risk engines. Ground-truth validation and local municipal oversight must accompany any engineering implementation.</p>
          </div>
          <div className="sign-block">
            <div className="sign-line"></div>
            <span>Authorized Remote Sensing Risk Assessment Signature</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExecutiveReportModal;
