import { useState } from 'react';
import Dashboard from './components/Dashboard';
import Uploader from './components/Uploader';
import type { RiskResponse } from './types';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api';

export function App() {
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<RiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const requestAnalysis = async (url: string, options?: RequestInit) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(url, options);
      const body = await response.json();

      if (!response.ok) {
        throw new Error(body.detail || 'The assessment service could not complete the processing request.');
      }

      setResult(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to reach the satellite assessment backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    void requestAnalysis(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });
  };

  const handleDemo = () => {
    void requestAnalysis(`${API_URL}/demo`);
  };

  return (
    <main className="app-shell">
      {/* Top Navbar Header */}
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <div className="brand-titles">
            <span className="brand-name">TerraRisk</span>
            <span className="brand-sub">Multi-Hazard IVP Platform</span>
          </div>
        </div>

        <div className="topbar-status">
          <span className="status-dot online"></span>
          <span className="status-text">IVP 8-Stage Engine Online</span>
        </div>
      </header>

      {/* Main App Content Area */}
      {!result ? (
        <div className="hero-container">
          <section className="hero-section">
            <span className="eyebrow">AI-BASED SATELLITE & REMOTE SENSING ASSESSMENT</span>
            <h1>
              Multi-Hazard Risk Fusion<br />
              <em>& IVP Preprocessing Matrix</em>
            </h1>
            <p className="hero-copy">
              A comprehensive decision-support platform integrating 8-stage image processing (CLAHE, ExG, VARI, 2D FFT, Canny edge detection), neural semantic segmentation, overhead object detection, and actionable disaster mitigation reports.
            </p>
          </section>

          {error && <div className="error-banner">⚠️ {error}</div>}

          <Uploader
            onAnalyze={handleAnalyze}
            onDemo={handleDemo}
            isLoading={loading}
          />
        </div>
      ) : (
        <Dashboard
          data={result}
          onReset={() => setResult(null)}
        />
      )}

      {/* Footer */}
      <footer className="footer-strip">
        <p>TerraRisk Remote Sensing Decision-Support System · Powered by 8-Stage IVP Engine & Deep Learning</p>
      </footer>
    </main>
  );
}

export default App;
