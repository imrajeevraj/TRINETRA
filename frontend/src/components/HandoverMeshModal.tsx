import React, { useState } from "react";
import { 
  X, 
  Share2, 
  Video, 
  Clock, 
  CheckCircle2, 
  ArrowRight, 
  Pause, 
  Play, 
  Mountain
} from "lucide-react";

interface HopData {
  hop_index: number;
  source_camera: string;
  target_camera: string;
  state: "CONFIRMED" | "ACQUIRED" | "PREDICTED" | "CANCELLED";
  prediction_confidence: number;
  association_confidence?: number;
  handover_confidence: "CONFIRMED" | "PROBABLE" | "UNCONFIRMED";
  expected_eta_sec: number;
  actual_arrival_sec?: number;
  lead_time_sec?: number;
  reason: string;
}

interface HandoverChainData {
  chain_id: string;
  entity_id: string;
  incident_id?: string;
  chain_status: "ACTIVE" | "COMPLETED" | "PAUSED" | "CANCELLED";
  hops: HopData[];
  current_camera: string;
  next_camera: string;
  terrain_mode: "DISABLED" | "SIMULATED";
  terrain_slope_deg?: number;
  terrain_speed_factor?: number;
}

interface HandoverMeshModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HandoverMeshModal: React.FC<HandoverMeshModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [chain, setChain] = useState<HandoverChainData>({
    chain_id: "CHAIN-2026-000001",
    entity_id: "GLOBAL-PERSON-00001",
    incident_id: "INC-2026-000001",
    chain_status: "ACTIVE",
    current_camera: "CAM-003",
    next_camera: "CAM-004",
    terrain_mode: "SIMULATED",
    terrain_slope_deg: 7.2,
    terrain_speed_factor: 0.85,
    hops: [
      {
        hop_index: 1,
        source_camera: "CAM-001",
        target_camera: "CAM-002",
        state: "CONFIRMED",
        prediction_confidence: 0.82,
        association_confidence: 0.85,
        handover_confidence: "CONFIRMED",
        expected_eta_sec: 14.0,
        actual_arrival_sec: 13.8,
        lead_time_sec: 11.2,
        reason: "Target entered CAM-002; acquired Optics with 11.2s lead time"
      },
      {
        hop_index: 2,
        source_camera: "CAM-002",
        target_camera: "CAM-003",
        state: "CONFIRMED",
        prediction_confidence: 0.78,
        association_confidence: 0.80,
        handover_confidence: "CONFIRMED",
        expected_eta_sec: 20.0,
        actual_arrival_sec: 19.4,
        lead_time_sec: 9.8,
        reason: "Handover confirmed at Logistics Gate corridor"
      },
      {
        hop_index: 3,
        source_camera: "CAM-003",
        target_camera: "CAM-004",
        state: "PREDICTED",
        prediction_confidence: 0.75,
        handover_confidence: "PROBABLE",
        expected_eta_sec: 28.0,
        lead_time_sec: 11.0,
        reason: "Target traversing South Sentry toward Airspace Tower ridge"
      }
    ]
  });

  const [isPaused, setIsPaused] = useState(false);

  if (!isOpen) return null;

  const togglePause = () => {
    setIsPaused(!isPaused);
    setChain(prev => ({
      ...prev,
      chain_status: isPaused ? "ACTIVE" : "PAUSED"
    }));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-5xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Share2 className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white tracking-wide flex items-center gap-2">
                Autonomous PTZ Handover Mesh
                <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                  Phase X Corridor
                </span>
              </h2>
              <p className="text-xs text-slate-400 font-mono">
                Coordinated Target Handoff • Multi-Camera Arbitration • Terrain Correction
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Corridor Meta Strip */}
        <div className="grid grid-cols-4 gap-4 px-6 py-3 bg-slate-950/40 border-b border-slate-800/80">
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 uppercase font-mono">Target Entity</span>
            <div className="text-sm font-bold font-mono text-white mt-0.5">
              {chain.entity_id}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">{chain.incident_id}</span>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 uppercase font-mono">Active Sequence</span>
            <div className="text-sm font-bold font-mono text-cyan-400 mt-0.5 flex items-center gap-1.5">
              {chain.current_camera} <ArrowRight className="w-3.5 h-3.5 text-slate-500" /> {chain.next_camera}
            </div>
            <span className="text-[10px] text-emerald-400 font-mono">Hop 3 of 4 in progress</span>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 uppercase font-mono">Terrain Telemetry</span>
            <div className="text-sm font-bold font-mono text-amber-400 mt-0.5 flex items-center gap-1.5">
              <Mountain className="w-3.5 h-3.5" /> {chain.terrain_slope_deg}° Incline
            </div>
            <span className="text-[10px] text-slate-500 font-mono">Factor: {chain.terrain_speed_factor}x ({chain.terrain_mode})</span>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-400 uppercase font-mono">Handover Status</span>
              <div className="text-sm font-bold font-mono text-emerald-400 mt-0.5">
                {chain.chain_status}
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                onClick={togglePause}
                className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                title={isPaused ? "Resume Handover" : "Pause Handover"}
              >
                {isPaused ? <Play className="w-4 h-4 text-emerald-400" /> : <Pause className="w-4 h-4 text-amber-400" />}
              </button>
            </div>
          </div>
        </div>

        {/* Visual 4-Camera Chain Corridor */}
        <div className="px-6 py-4 bg-slate-950/20 border-b border-slate-800">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 font-semibold mb-3 block">
            4-Camera Corridor Handover Chain
          </span>
          <div className="flex items-center justify-between gap-2">
            {/* CAM-001 */}
            <div className="flex-1 bg-slate-900/90 border border-emerald-500/40 rounded-lg p-3 relative overflow-hidden">
              <div className="text-xs font-mono font-bold text-white flex items-center justify-between">
                <span>CAM-001</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">ORIGIN</span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-1">North Sentry</div>
              <div className="text-[10px] text-emerald-400 font-mono mt-2 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Target Handed Off
              </div>
            </div>

            <ArrowRight className="w-5 h-5 text-emerald-500 shrink-0" />

            {/* CAM-002 */}
            <div className="flex-1 bg-slate-900/90 border border-emerald-500/40 rounded-lg p-3 relative overflow-hidden">
              <div className="text-xs font-mono font-bold text-white flex items-center justify-between">
                <span>CAM-002</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">HOP 1</span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-1">Logistics Gate</div>
              <div className="text-[10px] text-emerald-400 font-mono mt-2 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Handover Confirmed
              </div>
            </div>

            <ArrowRight className="w-5 h-5 text-emerald-500 shrink-0" />

            {/* CAM-003 */}
            <div className="flex-1 bg-slate-900/90 border border-cyan-500/50 rounded-lg p-3 relative overflow-hidden ring-1 ring-cyan-500/30">
              <div className="text-xs font-mono font-bold text-white flex items-center justify-between">
                <span>CAM-003</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400">ACTIVE</span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-1">South Sentry</div>
              <div className="text-[10px] text-cyan-400 font-mono mt-2 flex items-center gap-1">
                <Video className="w-3 h-3 animate-pulse" /> LOCKED • Tracking
              </div>
            </div>

            <ArrowRight className="w-5 h-5 text-purple-500 shrink-0 animate-pulse" />

            {/* CAM-004 */}
            <div className="flex-1 bg-slate-900/90 border border-purple-500/40 rounded-lg p-3 relative overflow-hidden">
              <div className="text-xs font-mono font-bold text-white flex items-center justify-between">
                <span>CAM-004</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400">PRE-CUED</span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-1">Airspace Tower</div>
              <div className="text-[10px] text-purple-400 font-mono mt-2 flex items-center gap-1">
                <Clock className="w-3 h-3" /> PTZ READY (Lead: 11s)
              </div>
            </div>
          </div>
        </div>

        {/* Hop Evidence Feed */}
        <div className="p-6 overflow-y-auto space-y-3 flex-1">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 font-semibold mb-2 block">
            Hop Audit Log & Verification Trace
          </span>
          {chain.hops.map((h) => (
            <div
              key={h.hop_index}
              className="bg-slate-950/60 border border-slate-800 rounded-lg p-4 flex items-center justify-between text-xs font-mono"
            >
              <div className="flex items-center gap-3">
                <span className="px-2 py-1 rounded bg-slate-800 text-white font-bold">
                  Hop {h.hop_index}
                </span>
                <div>
                  <div className="text-slate-200 font-semibold">
                    {h.source_camera} ➔ {h.target_camera}
                  </div>
                  <div className="text-slate-500 text-[11px] mt-0.5">
                    {h.reason}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-6">
                <div>
                  <span className="text-slate-500 block text-[10px]">Pred / Assoc</span>
                  <span className="text-slate-300">
                    {(h.prediction_confidence * 100).toFixed(0)}% / {h.association_confidence ? `${(h.association_confidence * 100).toFixed(0)}%` : 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">Lead Time</span>
                  <span className="text-cyan-400 font-bold">
                    {h.lead_time_sec ? `+${h.lead_time_sec}s` : 'N/A'}
                  </span>
                </div>
                <div>
                  {h.state === "CONFIRMED" && (
                    <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      CONFIRMED
                    </span>
                  )}
                  {h.state === "PREDICTED" && (
                    <span className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                      PREDICTED
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between text-xs text-slate-500 font-mono">
          <span>Max Handover Depth: 4 • Loop Protection: ACTIVE • Arbitration: PRIORITY_WEIGHTED</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
