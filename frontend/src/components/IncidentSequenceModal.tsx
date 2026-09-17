import React, { useState, useMemo } from 'react';
import { X, ShieldAlert, ArrowRight, Camera, Clock, Layers } from 'lucide-react';

interface TimelineStep {
  timestamp: number;
  camera_id: string;
  event_type: string;
  description: string;
  track_id?: string;
  global_id?: string;
  sensor_type: string;
  risk_delta: number;
}

interface IncidentDetail {
  incident_id: string;
  created_at: number;
  updated_at: number;
  status: string;
  severity: string;
  risk_score: number;
  summary: string;
  primary_camera: string;
  cameras_involved: string[];
  global_entities: string[];
  sensors_involved: string[];
  risk_reasons: string[];
}

interface IncidentSequenceModalProps {
  incidentId?: string;
  onClose: () => void;
  apiBase?: string;
}

const getMockDetail = (incidentId: string): IncidentDetail => ({
  incident_id: incidentId,
  created_at: Date.now() / 1000 - 120,
  updated_at: Date.now() / 1000,
  status: "ESCALATED",
  severity: "HIGH",
  risk_score: 75,
  summary: "Multi-camera intrusion corridor: target traversed from CAM-001 North sentry to CAM-002 Logistics gate.",
  primary_camera: "CAM-001",
  cameras_involved: ["CAM-001", "CAM-002", "CAM-003"],
  global_entities: ["GLOBAL-PERSON-00001"],
  sensors_involved: ["GROUND", "PTZ", "ROI"],
  risk_reasons: [
    "Target entered restricted boundary (+40)",
    "Automated PTZ slew-to-cue lock achieved",
    "Correlated across adjacent camera within 14.5s travel window"
  ]
});

const getMockTimeline = (): TimelineStep[] => [
  {
    timestamp: Date.now() / 1000 - 90,
    camera_id: "CAM-001",
    event_type: "PERSON_DETECTED",
    description: "Pedestrian observed moving south along boundary fence",
    track_id: "CAM-001:P-024",
    global_id: "GLOBAL-PERSON-00001",
    sensor_type: "GROUND",
    risk_delta: 20
  },
  {
    timestamp: Date.now() / 1000 - 75,
    camera_id: "CAM-001",
    event_type: "PTZ_STABLE_ROI",
    description: "PTZ slew stabilized (Pan=12.4°, Zoom=2.2x); Level 2 high-res crop confirmed",
    track_id: "CAM-001:P-024",
    global_id: "GLOBAL-PERSON-00001",
    sensor_type: "PTZ",
    risk_delta: 15
  },
  {
    timestamp: Date.now() / 1000 - 50,
    camera_id: "CAM-002",
    event_type: "CAMERA_TRANSITION",
    description: "Probable entity appearance on adjacent gate camera (15.2s elapsed, feasible window 5-30s)",
    track_id: "CAM-002:P-089",
    global_id: "GLOBAL-PERSON-00001",
    sensor_type: "GROUND",
    risk_delta: 25
  },
  {
    timestamp: Date.now() / 1000 - 15,
    camera_id: "CAM-002",
    event_type: "VIRTUAL_FENCE_CROSSING",
    description: "Target crossed primary perimeter security line into inner secure perimeter",
    track_id: "CAM-002:P-089",
    global_id: "GLOBAL-PERSON-00001",
    sensor_type: "GROUND",
    risk_delta: 40
  }
];

export const IncidentSequenceModal: React.FC<IncidentSequenceModalProps> = ({
  incidentId = "INC-2026-000001",
  onClose,
  apiBase: _apiBase = "http://localhost:8000"
}) => {
  const incident = useMemo(() => getMockDetail(incidentId), [incidentId]);
  const timeline = useMemo(() => getMockTimeline(), []);
  const [activeTab, setActiveTab] = useState<'sequence' | 'evidence' | 'timeline'>('sequence');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl text-slate-100 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/60">
          <div className="flex items-center gap-3">
            <ShieldAlert className="text-amber-400" size={24} />
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white tracking-wide">{incident?.incident_id}</h2>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">
                  {incident?.severity}
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  {incident?.status}
                </span>
              </div>
              <p className="text-xs text-slate-400">Risk Score: {incident?.risk_score}/100 • {incident?.cameras_involved.length} Cameras Correlated</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
          >
            <X size={20} />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-4 px-6 border-b border-slate-800 bg-slate-950/40 text-sm">
          <button
            className={`py-3 font-medium border-b-2 transition flex items-center gap-2 ${activeTab === 'sequence' ? 'border-amber-400 text-amber-400' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            onClick={() => setActiveTab('sequence')}
          >
            <Camera size={16} />
            Camera Sequence View
          </button>
          <button
            className={`py-3 font-medium border-b-2 transition flex items-center gap-2 ${activeTab === 'timeline' ? 'border-amber-400 text-amber-400' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            onClick={() => setActiveTab('timeline')}
          >
            <Clock size={16} />
            Forensic Timeline
          </button>
          <button
            className={`py-3 font-medium border-b-2 transition flex items-center gap-2 ${activeTab === 'evidence' ? 'border-amber-400 text-amber-400' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            onClick={() => setActiveTab('evidence')}
          >
            <Layers size={16} />
            Unified Evidence Graph
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto flex-1">
          {activeTab === 'sequence' && (
            <div className="space-y-6">
              <div className="p-4 rounded-lg bg-slate-950/50 border border-slate-800 text-sm">
                <p className="text-slate-300 font-medium">Cross-Camera Track Lineage:</p>
                <p className="text-xs text-slate-400 mt-1">{incident?.summary}</p>
              </div>

              {/* Step-by-Step Camera Corridor Flow */}
              <div className="flex items-center gap-3 overflow-x-auto pb-4 pt-2">
                {incident?.cameras_involved.map((cam, idx) => (
                  <React.Fragment key={cam}>
                    <div className="flex-1 min-w-[200px] p-4 rounded-xl bg-slate-800/80 border border-slate-700 shadow-md">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-mono font-bold text-amber-400">{cam}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">HOP {idx + 1}</span>
                      </div>
                      <p className="text-xs font-semibold text-slate-200 mb-1">
                        {idx === 0 ? "Initial Intrusion Sentry" : idx === 1 ? "Adjacent Gate Transfer" : "Inner Perimeter Sentry"}
                      </p>
                      <p className="text-[11px] text-slate-400">
                        {idx === 0 ? "Track CAM-001:P-024" : idx === 1 ? "Track CAM-002:P-089" : "Track CAM-003:P-114"}
                      </p>
                      <div className="mt-3 pt-2 border-t border-slate-700/60 flex items-center justify-between text-[10px] text-emerald-400">
                        <span>Likely Match (92%)</span>
                        <span>Δt ~14.8s</span>
                      </div>
                    </div>
                    {idx < (incident?.cameras_involved.length - 1) && (
                      <ArrowRight className="text-slate-500 shrink-0" size={20} />
                    )}
                  </React.Fragment>
                ))}
              </div>

              {/* Risk Reasons */}
              <div className="rounded-lg bg-rose-950/20 border border-rose-800/30 p-4">
                <h4 className="text-xs font-bold text-rose-300 uppercase tracking-wider mb-2">Dynamic Risk Escalation Audit</h4>
                <ul className="text-xs text-rose-200/80 space-y-1 list-disc list-inside">
                  {incident?.risk_reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'timeline' && (
            <div className="space-y-4">
              <div className="relative pl-6 border-l border-slate-700 space-y-6">
                {timeline.map((step, idx) => (
                  <div key={idx} className="relative">
                    <div className="absolute -left-[31px] top-1 w-3 h-3 rounded-full bg-amber-400 ring-4 ring-slate-900" />
                    <div className="p-3.5 rounded-lg bg-slate-800/60 border border-slate-700/80">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="font-mono font-bold text-amber-300">{step.camera_id}</span>
                        <span className="text-slate-400">{new Date(step.timestamp * 1000).toLocaleTimeString()}</span>
                      </div>
                      <p className="text-xs font-semibold text-slate-100">{step.event_type}</p>
                      <p className="text-xs text-slate-300 mt-1">{step.description}</p>
                      {step.global_id && (
                        <p className="text-[11px] text-slate-400 mt-1 font-mono">Entity: {step.global_id}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'evidence' && (
            <div className="space-y-4 text-xs text-slate-300">
              <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800">
                <h4 className="font-semibold text-white mb-2">Cryptographic Evidence Artifacts (SHA-256 Verified):</h4>
                <div className="space-y-2 font-mono text-[11px]">
                  <div className="p-2.5 rounded bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <span>CAM-001_frame_0492.jpg</span>
                    <span className="text-emerald-400">SHA: 7DBF36...296277B0 (MATCH)</span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <span>CAM-001_ptz_stable_crop.jpg</span>
                    <span className="text-emerald-400">SHA: 522963...30367D23 (MATCH)</span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <span>CAM-002_fence_breach.jpg</span>
                    <span className="text-emerald-400">SHA: 72464C...E7F6E1DA (MATCH)</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between px-6 py-3.5 border-t border-slate-800 bg-slate-950/80">
          <span className="text-xs text-slate-400">Operator Review Required</span>
          <div className="flex gap-2">
            <button 
              onClick={onClose}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
            >
              Close
            </button>
            <button 
              onClick={() => { alert("Incident Confirmed by Operator"); onClose(); }}
              className="px-3.5 py-1.5 text-xs font-medium rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm transition"
            >
              Confirm Incident
            </button>
            <button 
              onClick={() => { alert("Incident Marked False Alarm"); onClose(); }}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-rose-600/80 hover:bg-rose-600 text-white transition"
            >
              Reject / False Alarm
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
