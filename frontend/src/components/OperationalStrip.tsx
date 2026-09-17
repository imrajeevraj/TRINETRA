import React, { useState } from 'react';
import type { SystemStats } from '../types';
import { ChevronUp, UploadCloud, GitBranch, TrendingUp, Share2, Cpu, ShieldCheck, BrainCircuit, Flame } from 'lucide-react';
import { AuditUploadModal } from './AuditUploadModal';
import { IncidentSequenceModal } from './IncidentSequenceModal';
import { ForecastComparisonModal } from './ForecastComparisonModal';
import { HandoverMeshModal } from './HandoverMeshModal';
import { EdgeMeshModal } from './EdgeMeshModal';
import { ModelLifecycleModal } from './ModelLifecycleModal';
import { AiQualityDashboardModal } from './AiQualityDashboardModal';
import { MultimodalSensorModal } from './MultimodalSensorModal';

interface OperationalStripProps {
  stats: SystemStats;
  dataOrigin: 'LIVE' | 'DEMO' | 'IMPORTED';
  onToggleDataOrigin: () => void;
  onRunDemo: (scenario: 'seed-intrusion' | 'seed-watchlist') => void;
}

export const OperationalStrip: React.FC<OperationalStripProps> = ({
  stats,
  dataOrigin,
  onToggleDataOrigin,
  onRunDemo,
}) => {
  const [showSimControls, setShowSimControls] = useState(false);
  const [showAuditModal, setShowAuditModal] = useState(false);
  const [showIncidentModal, setShowIncidentModal] = useState(false);
  const [showForecastModal, setShowForecastModal] = useState(false);
  const [showHandoverModal, setShowHandoverModal] = useState(false);
  const [showEdgeMeshModal, setShowEdgeMeshModal] = useState(false);
  const [showLifecycleModal, setShowLifecycleModal] = useState(false);
  const [showAiQualityModal, setShowAiQualityModal] = useState(false);
  const [showMultimodalModal, setShowMultimodalModal] = useState(false);
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  return (
    <>
    <footer className="operational-strip" role="contentinfo">
      <div className="strip-metrics">
        <div className="strip-metric">
          <span className="strip-metric-label">Cameras</span>
          <span className={`strip-metric-value ${stats.cameras_online === stats.total_cameras ? 'healthy' : 'warning'}`}>
            {stats.cameras_online}/{stats.total_cameras}
          </span>
        </div>

        <span className="strip-separator" />

        <div className="strip-metric">
          <span className="strip-metric-label">Persons</span>
          <span className="strip-metric-value">{String(stats.person_detections).padStart(2, '0')}</span>
        </div>

        <div className="strip-metric">
          <span className="strip-metric-label">Vehicles</span>
          <span className="strip-metric-value">{String(stats.vehicle_detections).padStart(2, '0')}</span>
        </div>

        <span className="strip-separator" />

        <div className="strip-metric">
          <span className="strip-metric-label">Active Alerts</span>
          <span className={`strip-metric-value ${stats.active_alerts > 0 ? 'warning' : ''}`}>
            {stats.active_alerts}
          </span>
        </div>

        <div className="strip-metric">
          <span className="strip-metric-label">Critical</span>
          <span className={`strip-metric-value ${stats.critical_alerts > 0 ? 'critical' : ''}`}>
            {stats.critical_alerts}
          </span>
        </div>

        <span className="strip-separator" />

        <div className="strip-metric">
          <span className="strip-metric-label">AI Engine</span>
          <span className="strip-metric-value" style={{ color: 'var(--accent)' }}>YOLO11</span>
        </div>

        <span className="strip-separator" />

        <div className="strip-metric">
          <span className="strip-metric-label">PTZ Slew-to-Cue</span>
          <span className="strip-metric-value healthy" title="PTZ Cue Engine: Auto-Tracking & Slew-to-Cue Active">
            STABLE (SIM)
          </span>
        </div>
      </div>

      <div className="strip-right">
        <div className="strip-metric">
          <span className="strip-metric-label">Data Mode</span>
          <span className={`strip-metric-value ${dataOrigin === 'DEMO' ? 'warning' : 'healthy'}`}>
            {dataOrigin}
          </span>
        </div>

        <div className="demo-controls flex items-center gap-2">
          <button 
            className="btn btn-sm flex items-center gap-2 text-amber-300 border-amber-500/30 bg-amber-950/20"
            onClick={() => setShowMultimodalModal(true)}
            title="Multimodal Sensor Intelligence & Native Thermal AI (Phase XIV)"
          >
            <Flame size={14} />
            Multimodal AI
          </button>
          <button 
            className="btn btn-sm flex items-center gap-2 text-cyan-300 border-cyan-500/30 bg-cyan-950/20"
            onClick={() => setShowAiQualityModal(true)}
            title="Continuous Edge Learning & AI Quality Intelligence (Phase XIII)"
          >
            <BrainCircuit size={14} />
            AI Quality
          </button>
          <button 
            className="btn btn-sm flex items-center gap-2 text-emerald-400 border-emerald-500/30"
            onClick={() => setShowLifecycleModal(true)}
            title="Edge AI Model Lifecycle & Distributed Governance"
          >
            <ShieldCheck size={14} />
            Model Lifecycle
          </button>
          <button 
            className="btn btn-sm flex items-center gap-2 text-cyan-400 border-cyan-500/30"
            onClick={() => setShowEdgeMeshModal(true)}
            title="Distributed Edge Camera Mesh & Peer Handover"
          >
            <Cpu size={14} />
            Edge Mesh
          </button>
          <button 
            className="btn btn-sm flex items-center gap-2"
            onClick={() => setShowHandoverModal(true)}
            title="Autonomous PTZ Handover Mesh"
          >
            <Share2 size={14} />
            Handover Mesh
          </button>
          <button 
            className="btn btn-sm flex items-center gap-2"
            onClick={() => setShowForecastModal(true)}
            title="Predictive Threat Intelligence & Forecast Audit"
          >
            <TrendingUp size={14} />
            Forecast Audit
          </button>
          <button 
            className="btn btn-sm flex items-center gap-2"
            onClick={() => setShowIncidentModal(true)}
            title="Unified Incident Sequence View"
          >
            <GitBranch size={14} />
            Incident Sequence
          </button>
          <button 
            className="btn btn-sm flex items-center gap-2"
            onClick={() => setShowAuditModal(true)}
            title="Upload Offline Video"
          >
            <UploadCloud size={14} />
            Offline Audit
          </button>
        </div>

        {/* Simulation Controls */}
        <div className="sim-controls">
          <button
            className="btn btn-sm"
            onClick={() => setShowSimControls(prev => !prev)}
            title="Simulation controls"
          >
            <ChevronUp size={11} style={{ transform: showSimControls ? 'rotate(180deg)' : undefined, transition: 'transform 0.2s' }} />
            Sim
          </button>
          {showSimControls && (
            <div className="sim-dropdown">
              <button className="btn btn-sm" onClick={() => { onRunDemo('seed-intrusion'); setShowSimControls(false); }}>
                Demo Intrusion
              </button>
              <button className="btn btn-sm" onClick={() => { onRunDemo('seed-watchlist'); setShowSimControls(false); }}>
                Demo ANPR
              </button>
              <button className="btn btn-sm" onClick={() => { onToggleDataOrigin(); setShowSimControls(false); }}>
                Cycle Data Source (Currently {dataOrigin})
              </button>
            </div>
          )}
        </div>
      </div>
    </footer>
    {showAuditModal && (
      <AuditUploadModal 
        onClose={() => setShowAuditModal(false)}
        apiBase={apiBase}
      />
    )}
    {showIncidentModal && (
      <IncidentSequenceModal 
        onClose={() => setShowIncidentModal(false)}
        apiBase={apiBase}
      />
    )}
    {showForecastModal && (
      <ForecastComparisonModal 
        isOpen={showForecastModal}
        onClose={() => setShowForecastModal(false)}
      />
    )}
    {showHandoverModal && (
      <HandoverMeshModal 
        isOpen={showHandoverModal}
        onClose={() => setShowHandoverModal(false)}
      />
    )}
    {showEdgeMeshModal && (
      <EdgeMeshModal 
        isOpen={showEdgeMeshModal}
        onClose={() => setShowEdgeMeshModal(false)}
      />
    )}
    {showLifecycleModal && (
      <ModelLifecycleModal 
        isOpen={showLifecycleModal}
        onClose={() => setShowLifecycleModal(false)}
      />
    )}
    {showAiQualityModal && (
      <AiQualityDashboardModal 
        isOpen={showAiQualityModal}
        onClose={() => setShowAiQualityModal(false)}
      />
    )}
    {showMultimodalModal && (
      <MultimodalSensorModal 
        isOpen={showMultimodalModal}
        onClose={() => setShowMultimodalModal(false)}
      />
    )}
    </>
  );
};
