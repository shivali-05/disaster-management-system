import React, { useState } from 'react';

interface VisualizerProps {
  originalImg?: string;
  segMask?: string;
  detOverlay?: string;
  earthvlHeatmap?: string;
  ivpStages?: Array<{ id: string; title: string; image_base64: string }>;
}

export const Visualizer: React.FC<VisualizerProps> = ({
  originalImg,
  segMask,
  detOverlay,
  earthvlHeatmap,
  ivpStages = [],
}) => {
  const [activeLayer, setActiveLayer] = useState<'original' | 'segmentation' | 'detection' | 'heatmap' | 'ivp'>('heatmap');
  const [selectedIvpId, setSelectedIvpId] = useState<string>(ivpStages[0]?.id || '');
  const [opacity, setOpacity] = useState<number>(0.75);

  const selectedIvpStage = ivpStages.find(s => s.id === selectedIvpId) || ivpStages[0];

  return (
    <div className="visualizer-container glass-card">
      <div className="visualizer-header">
        <div>
          <span className="section-badge">INTERACTIVE MAP ENGINE</span>
          <h3>Multi-Spectral Layer & Evidence Visualizer</h3>
        </div>

        <div className="layer-selector">
          <button
            className={`layer-btn ${activeLayer === 'heatmap' ? 'active' : ''}`}
            onClick={() => setActiveLayer('heatmap')}
          >
            🔥 EarthVL Risk Heatmap
          </button>

          <button
            className={`layer-btn ${activeLayer === 'original' ? 'active' : ''}`}
            onClick={() => setActiveLayer('original')}
          >
            Raw Satellite RGB
          </button>

          <button
            className={`layer-btn ${activeLayer === 'segmentation' ? 'active' : ''}`}
            onClick={() => setActiveLayer('segmentation')}
          >
            LoveDA Semantic Mask
          </button>

          <button
            className={`layer-btn ${activeLayer === 'detection' ? 'active' : ''}`}
            onClick={() => setActiveLayer('detection')}
          >
            YOLO Object Overlay
          </button>

          {ivpStages.length > 0 && (
            <button
              className={`layer-btn ${activeLayer === 'ivp' ? 'active' : ''}`}
              onClick={() => setActiveLayer('ivp')}
            >
              IVP Filter Stage
            </button>
          )}
        </div>
      </div>

      {activeLayer === 'ivp' && ivpStages.length > 0 && (
        <div className="ivp-selector-bar">
          <label>Select Preprocessed Stage Output:</label>
          <select value={selectedIvpId} onChange={e => setSelectedIvpId(e.target.value)}>
            {ivpStages.map(s => (
              <option key={s.id} value={s.id}>{s.title}</option>
            ))}
          </select>
        </div>
      )}

      {/* Layer Opacity Slider */}
      {(activeLayer === 'segmentation' || activeLayer === 'detection' || activeLayer === 'heatmap') && (
        <div className="opacity-control-strip">
          <span>Overlay Opacity: <strong>{Math.round(opacity * 100)}%</strong></span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={opacity}
            onChange={e => setOpacity(parseFloat(e.target.value))}
          />
        </div>
      )}

      {/* Image Display Area */}
      <div className="image-view-box">
        {originalImg && (
          <img
            src={originalImg}
            alt="Original Satellite"
            className="base-image-layer"
          />
        )}

        {activeLayer === 'heatmap' && earthvlHeatmap && (
          <img
            src={earthvlHeatmap}
            alt="EarthVL Risk Heatmap"
            className="overlay-image-layer"
            style={{ opacity }}
          />
        )}

        {activeLayer === 'segmentation' && segMask && (
          <img
            src={segMask}
            alt="Segmentation Overlay"
            className="overlay-image-layer"
            style={{ opacity }}
          />
        )}

        {activeLayer === 'detection' && detOverlay && (
          <img
            src={detOverlay}
            alt="Detection Overlay"
            className="overlay-image-layer"
            style={{ opacity }}
          />
        )}

        {activeLayer === 'ivp' && selectedIvpStage && (
          <img
            src={selectedIvpStage.image_base64}
            alt={selectedIvpStage.title}
            className="overlay-image-layer"
            style={{ opacity: 1 }}
          />
        )}
      </div>

      {/* Legend & Key */}
      <div className="visualizer-legend">
        <span className="legend-title">Land Cover Class Key:</span>
        <div className="legend-items">
          <div className="legend-item"><i className="color-dot building"></i> Building</div>
          <div className="legend-item"><i className="color-dot road"></i> Road Surface</div>
          <div className="legend-item"><i className="color-dot water"></i> Water Body</div>
          <div className="legend-item"><i className="color-dot vegetation"></i> Vegetation / Forest</div>
          <div className="legend-item"><i className="color-dot bare"></i> Exposed Bare Land</div>
        </div>
      </div>
    </div>
  );
};

export default Visualizer;
