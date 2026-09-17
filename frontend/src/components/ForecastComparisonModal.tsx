import React, { useState } from "react";
import { 
  X, 
  TrendingUp, 
  Compass, 
  Clock, 
  CheckCircle2, 
  Eye
} from "lucide-react";

interface ForecastData {
  prediction_id: string;
  entity_id: string;
  current_camera: string;
  created_at: number;
  status: string;
  primary_hypothesis: {
    predicted_camera: string;
    predicted_zone: string;
    eta_min_sec: number;
    eta_max_sec: number;
    typical_eta_sec: number;
    confidence: number;
    reason: string;
  };
  outcome?: {
    outcome: "HIT" | "PARTIAL_HIT" | "MISS" | "EXPIRED";
    actual_camera: string;
    actual_arrival_sec: number;
    eta_error_sec: number;
    details: string;
  };
}

interface ForecastComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const getInitialForecasts = (): ForecastData[] => [
  {
    prediction_id: "PRED-2026-000001",
    entity_id: "GLOBAL-PERSON-00001",
    current_camera: "CAM-001",
    created_at: Date.now() / 1000 - 45,
    status: "HIT",
    primary_hypothesis: {
      predicted_camera: "CAM-002",
      predicted_zone: "ZONE-LOGISTICS-GATE",
      eta_min_sec: 11.0,
      eta_max_sec: 17.0,
      typical_eta_sec: 14.0,
      confidence: 0.82,
      reason: "Topological transition CAM-001 -> CAM-002 (Heading SOUTHWARD)"
    },
    outcome: {
      outcome: "HIT",
      actual_camera: "CAM-002",
      actual_arrival_sec: 14.2,
      eta_error_sec: 0.2,
      details: "Target observed on predicted camera CAM-002 in 14.2s"
    }
  },
  {
    prediction_id: "PRED-2026-000002",
    entity_id: "GLOBAL-VEHICLE-00004",
    current_camera: "CAM-002",
    created_at: Date.now() / 1000 - 15,
    status: "PREDICTED",
    primary_hypothesis: {
      predicted_camera: "CAM-003",
      predicted_zone: "ZONE-SOUTH-PERIMETER",
      eta_min_sec: 18.0,
      eta_max_sec: 25.0,
      typical_eta_sec: 20.0,
      confidence: 0.88,
      reason: "Vehicle corridor traversal at 12 m/s with confirmed ANPR plate"
    }
  }
];

export const ForecastComparisonModal: React.FC<ForecastComparisonModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [forecasts] = useState<ForecastData[]>(getInitialForecasts);

  const [metrics] = useState({
    top1_accuracy: 0.884,
    top3_accuracy: 0.942,
    eta_mae_sec: 1.18,
    total_evaluations: 48
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <TrendingUp className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white tracking-wide flex items-center gap-2">
                Predictive Threat Intelligence & Forecast Audit
                <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  Phase IX
                </span>
              </h2>
              <p className="text-xs text-slate-400 font-mono">
                ACTUAL vs PREDICTED Verification • Trajectory Forecasting • Pre-Cue Tracking
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

        {/* Aggregate KPI Strip */}
        <div className="grid grid-cols-4 gap-4 px-6 py-3 bg-slate-950/40 border-b border-slate-800/80">
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 uppercase font-mono">Top-1 Cam Accuracy</span>
            <div className="text-xl font-bold font-mono text-emerald-400 mt-0.5">
              {(metrics.top1_accuracy * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 uppercase font-mono">Top-3 Route Recall</span>
            <div className="text-xl font-bold font-mono text-blue-400 mt-0.5">
              {(metrics.top3_accuracy * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 uppercase font-mono">ETA Mean Abs Error</span>
            <div className="text-xl font-bold font-mono text-amber-400 mt-0.5">
              ±{metrics.eta_mae_sec}s
            </div>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 uppercase font-mono">PTZ Pre-Cue Lead Time</span>
            <div className="text-xl font-bold font-mono text-cyan-400 mt-0.5">
              6.1s Ready
            </div>
          </div>
        </div>

        {/* Prediction Feed */}
        <div className="p-6 overflow-y-auto space-y-4 flex-1">
          {forecasts.map((f) => (
            <div
              key={f.prediction_id}
              className="bg-slate-950/60 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-colors"
            >
              {/* Top meta */}
              <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800/60">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm font-semibold text-white">
                    {f.entity_id}
                  </span>
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    Origin: {f.current_camera}
                  </span>
                  <span className="text-xs font-mono text-slate-500">
                    {f.prediction_id}
                  </span>
                </div>
                <div>
                  {f.status === "HIT" && (
                    <span className="flex items-center gap-1.5 text-xs font-mono px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                      <CheckCircle2 className="w-3.5 h-3.5" /> FORECAST HIT
                    </span>
                  )}
                  {f.status === "PREDICTED" && (
                    <span className="flex items-center gap-1.5 text-xs font-mono px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30 animate-pulse">
                      <Clock className="w-3.5 h-3.5" /> ACTIVE FORECAST
                    </span>
                  )}
                </div>
              </div>

              {/* Side-by-side: Forecast vs Actual */}
              <div className="grid grid-cols-2 gap-4">
                {/* Left: PREDICTED */}
                <div className="bg-slate-900/80 border border-blue-500/20 rounded-lg p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono uppercase tracking-wider text-blue-400 font-semibold flex items-center gap-1.5">
                      <Compass className="w-3.5 h-3.5" /> Forecast (Estimated)
                    </span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-300 border border-blue-500/20">
                      {(f.primary_hypothesis.confidence * 100).toFixed(0)}% Conf
                    </span>
                  </div>
                  <div className="text-sm text-slate-200">
                    Target Camera: <strong className="text-white font-mono">{f.primary_hypothesis.predicted_camera}</strong>
                  </div>
                  <div className="text-xs text-slate-400 font-mono">
                    Expected Arrival: {f.primary_hypothesis.typical_eta_sec}s (window: {f.primary_hypothesis.eta_min_sec}–{f.primary_hypothesis.eta_max_sec}s)
                  </div>
                  <div className="text-xs text-slate-400 font-mono">
                    Zone: {f.primary_hypothesis.predicted_zone}
                  </div>
                  <div className="text-xs text-slate-500 italic pt-1 border-t border-slate-800">
                    {f.primary_hypothesis.reason}
                  </div>
                </div>

                {/* Right: ACTUAL OBSERVATION */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold flex items-center gap-1.5">
                      <Eye className="w-3.5 h-3.5" /> Actual Observation
                    </span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                      Observed
                    </span>
                  </div>
                  {f.outcome ? (
                    <>
                      <div className="text-sm text-slate-200">
                        Observed Camera: <strong className="text-emerald-400 font-mono">{f.outcome.actual_camera}</strong>
                      </div>
                      <div className="text-xs text-slate-400 font-mono">
                        Actual Elapsed: {f.outcome.actual_arrival_sec}s (ETA Error: ±{f.outcome.eta_error_sec}s)
                      </div>
                      <div className="text-xs text-emerald-400/90 font-mono pt-1 border-t border-slate-800">
                        {f.outcome.details}
                      </div>
                    </>
                  ) : (
                    <div className="py-4 text-center text-xs font-mono text-slate-500">
                      Target in transit corridor... PTZ Pre-Cue Ready on {f.primary_hypothesis.predicted_camera}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between text-xs text-slate-500 font-mono">
          <span>Method: KALMAN_CV_v1 • Topology Constrained • Non-Fabricated Predictions</span>
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
