import React, { useState, useRef } from 'react';

interface UploaderProps {
  onAnalyze: (file: File) => void;
  onDemo: () => void;
  isLoading: boolean;
}

export const Uploader: React.FC<UploaderProps> = ({ onAnalyze, onDemo, isLoading }) => {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onAnalyze(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onAnalyze(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="launch-section">
      <div className="uploader-card glass-card">
        <div className="card-header">
          <span className="eyebrow">IMAGE & GEOSPATIAL INPUT</span>
          <h2>Upload Satellite / Remote Sensing Image</h2>
          <p>
            Drop any high-resolution RGB satellite tile (PNG, JPG, TIFF) to execute all 8 IVP preprocessing stages, spectral indexing, land-cover segmentation, and risk scoring.
          </p>
        </div>

        <div
          className={`dropzone ${dragOver ? 'drag-active' : ''} ${isLoading ? 'loading' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !isLoading && fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            style={{ display: 'none' }}
          />

          {isLoading ? (
            <div className="loading-state">
              <div className="pulse-spinner"></div>
              <h3>Running 8-Stage IVP & Neural Assessment Pipeline</h3>
              <p>Computing CLAHE, ExG, VARI, 2D FFT, Canny edge matrix, and disaster report...</p>
              <div className="loading-progress-bar">
                <div className="progress-bar-fill"></div>
              </div>
            </div>
          ) : (
            <div className="dropzone-prompt">
              <div className="upload-icon-circle">
                <span>🛰️</span>
              </div>
              <h3>Drag & drop satellite image patch here</h3>
              <p>or <span className="highlight-text">browse files</span> from your computer</p>
              <span className="format-note">Supports PNG, JPEG, TIFF (Max 25 MB)</span>
            </div>
          )}
        </div>

        <div className="demo-launcher-strip">
          <div className="demo-text">
            <strong>Want a pre-loaded presentation scenario?</strong>
            <span>Explore the Peri-Urban Water-Edge demo case (Sample #2522) instantly.</span>
          </div>

          <button
            className="secondary-btn demo-btn"
            disabled={isLoading}
            onClick={onDemo}
          >
            🚀 Open Demo Assessment
          </button>
        </div>
      </div>
    </div>
  );
};

export default Uploader;
