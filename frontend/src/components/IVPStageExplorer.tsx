import React, { useState } from 'react';
import type { IVPStageOutput, IVPMetrics } from '../types';

interface IVPStageExplorerProps {
  stages: IVPStageOutput[];
  metrics: IVPMetrics;
  originalImage?: string;
}

export const IVPStageExplorer: React.FC<IVPStageExplorerProps> = ({ stages, metrics, originalImage }) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [activeModalStage, setActiveModalStage] = useState<IVPStageOutput | null>(null);
  const [showCompare, setShowCompare] = useState<boolean>(false);

  const categories = ['All', 'Radiometric', 'Noise Filtering', 'Spectral', 'Edge & Gradient', 'Frequency Domain', 'Morphology', 'Thresholding'];

  const filteredStages = selectedCategory === 'All'
    ? stages
    : stages.filter(s => s.category.toLowerCase().includes(selectedCategory.toLowerCase()));

  return (
    <div className="ivp-explorer">
      {/* Header & Metrics Overview Bar */}
      <div className="ivp-header">
        <div>
          <span className="section-badge">PIPELINE DEEP-DIVE</span>
          <h2 className="section-title">Multi-Stage IVP Preprocessing Matrix</h2>
          <p className="section-subtitle">
            All 16 deterministic image & video processing stage outputs derived dynamically from your satellite patch.
          </p>
        </div>
        
        <div className="ivp-metrics-strip">
          <div className="metric-chip">
            <span className="metric-label">Entropy</span>
            <strong className="metric-val">{metrics.image_entropy} bits</strong>
          </div>
          <div className="metric-chip">
            <span className="metric-label">Edge Density</span>
            <strong className="metric-val">{(metrics.canny_edge_density * 100).toFixed(1)}%</strong>
          </div>
          <div className="metric-chip">
            <span className="metric-label">Dimensions</span>
            <strong className="metric-val">{metrics.image_dimensions[1]}x{metrics.image_dimensions[0]}</strong>
          </div>
          <div className="metric-chip">
            <span className="metric-label">Tiles (512²)</span>
            <strong className="metric-val">{metrics.tile_count} Patches</strong>
          </div>
        </div>
      </div>

      {/* Category Filter Pills */}
      <div className="category-filters">
        {categories.map(cat => (
          <button
            key={cat}
            className={`filter-btn ${selectedCategory === cat ? 'active' : ''}`}
            onClick={() => setSelectedCategory(cat)}
          >
            {cat}
            <span className="count-pill">
              {cat === 'All' ? stages.length : stages.filter(s => s.category.toLowerCase().includes(cat.toLowerCase())).length}
            </span>
          </button>
        ))}
      </div>

      {/* 16 Stage Image Cards Grid */}
      <div className="stages-grid">
        {filteredStages.map((stage, idx) => (
          <div key={stage.id} className="stage-card">
            <div className="stage-card-header">
              <span className="stage-num">#{idx + 1}</span>
              <span className="category-tag">{stage.category}</span>
            </div>

            <div className="stage-image-wrapper" onClick={() => setActiveModalStage(stage)}>
              <img src={stage.image_base64} alt={stage.title} className="stage-img" />
              <div className="img-overlay-hover">
                <span>🔍 Click to Expand & Compare</span>
              </div>
            </div>

            <div className="stage-card-body">
              <h4 className="stage-title">{stage.title}</h4>
              <p className="stage-summary">{stage.summary}</p>
              
              <div className="ai-info-box">
                <div className="ai-info-header">
                  <span className="ai-sparkle">✦</span>
                  <strong>AI Domain Analysis</strong>
                </div>
                <p className="ai-info-text">{stage.disaster_relevance}</p>
              </div>

              <div className="stage-card-footer">
                <small className="tech-detail">⚙ {stage.technical_detail}</small>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Interactive Lightbox / Comparison Modal */}
      {activeModalStage && (
        <div className="modal-backdrop" onClick={() => setActiveModalStage(null)}>
          <div className="modal-content glass-card" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setActiveModalStage(null)}>✕</button>

            <div className="modal-header">
              <div>
                <span className="category-tag">{activeModalStage.category}</span>
                <h3>{activeModalStage.title}</h3>
                <p>{activeModalStage.stage_name}</p>
              </div>

              {originalImage && (
                <button
                  className={`compare-toggle-btn ${showCompare ? 'active' : ''}`}
                  onClick={() => setShowCompare(!showCompare)}
                >
                  {showCompare ? 'Show Single View' : '⇄ Side-by-Side Compare'}
                </button>
              )}
            </div>

            <div className={`modal-view-container ${showCompare ? 'split-view' : ''}`}>
              {showCompare && originalImage && (
                <div className="image-panel">
                  <span className="panel-label">Original Raw RGB Satellite Image</span>
                  <img src={originalImage} alt="Raw Input" className="modal-img" />
                </div>
              )}

              <div className="image-panel">
                <span className="panel-label">IVP Preprocessed Output: {activeModalStage.title}</span>
                <img src={activeModalStage.image_base64} alt={activeModalStage.title} className="modal-img" />
              </div>
            </div>

            <div className="modal-info-grid">
              <div className="info-block">
                <h4>Stage Purpose & Processing</h4>
                <p>{activeModalStage.summary}</p>
                <code>{activeModalStage.technical_detail}</code>
              </div>
              
              <div className="info-block highlight">
                <h4>AI Remote Sensing & Disaster Insights</h4>
                <p>{activeModalStage.disaster_relevance}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IVPStageExplorer;
