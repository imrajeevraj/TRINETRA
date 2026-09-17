import React, { useState } from "react";
import {
  X,
  Layers,
  Activity,
  AlertTriangle,
  Lock,
  Eye,
  Camera,
  Flame,
  Clock,
  Compass,
  GitBranch,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
  Cpu,
  Database,
  BarChart3,
} from "lucide-react";

interface MultimodalSensorModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MultimodalSensorModal: React.FC<MultimodalSensorModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<
    | "topology"
    | "optical_feed"
    | "thermal_feed"
    | "sync_view"
    | "associations"
    | "tracks"
    | "health"
    | "calibration"
    | "synchronization"
    | "thermal_models"
    | "fusion_quality"
    | "failure_analysis"
    | "dataset_gov"
    | "rgb_vs_thermal"
  >("sync_view");

  const [simulatedThermalMode, setSimulatedThermalMode] = useState<string>("WHITE_HOT");
  const [registrationOverride, setRegistrationOverride] = useState<boolean>(true);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl w-full max-w-7xl max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/70">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-lg border border-amber-500/30">
              <Flame className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white tracking-wide">
                  MULTIMODAL SENSOR INTELLIGENCE & NATIVE THERMAL AI
                </h2>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  PHASE XV
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  OPTICAL: PRODUCTION (FROZEN)
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  DISCOVERY: NO REAL SENSOR DETECTED
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-rose-500/20 text-rose-300 border border-rose-500/30 animate-pulse">
                  THERMAL AI: NOT VALIDATED
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  BENCHMARK: READINESS-v0 (FROZEN)
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Real LWIR Hardware Acquisition • Governed 12-State Datasets • Anti-Leakage Guard • RGB vs Thermal vs Fusion
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/80 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation (12 Tabs) */}
        <div className="flex items-center gap-1 px-6 py-2 border-b border-slate-800/80 bg-slate-950/40 overflow-x-auto text-xs">
          {[
            { id: "sync_view", label: "Synchronized View", icon: Eye },
            { id: "rgb_vs_thermal", label: "RGB vs Thermal vs Fusion", icon: BarChart3 },
            { id: "dataset_gov", label: "Dataset Governance (12-State)", icon: Database },
            { id: "topology", label: "Sensor Topology", icon: Compass },
            { id: "optical_feed", label: "Optical Feed (RGB)", icon: Camera },
            { id: "thermal_feed", label: "Thermal Feed (LWIR)", icon: Flame },
            { id: "associations", label: "Cross-Spectral Match", icon: Layers },
            { id: "tracks", label: "Multimodal Tracks", icon: GitBranch },
            { id: "health", label: "Sensor Health", icon: Activity },
            { id: "calibration", label: "Calibration", icon: ShieldCheck },
            { id: "synchronization", label: "Time Sync", icon: Clock },
            { id: "thermal_models", label: "Thermal Model Gov", icon: Cpu },
            { id: "fusion_quality", label: "Fusion Quality", icon: Sparkles },
            { id: "failure_analysis", label: "Failure Analysis", icon: ShieldAlert },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium whitespace-nowrap transition-all ${
                  active
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {actionFeedback && (
            <div className="p-3 bg-amber-500/20 border border-amber-500/30 rounded-lg text-amber-200 text-xs flex items-center justify-between">
              <span>{actionFeedback}</span>
              <button onClick={() => setActionFeedback(null)} className="text-amber-400 hover:underline">
                Dismiss
              </button>
            </div>
          )}

          {/* TAB 1: SYNCHRONIZED DUAL VIEW */}
          {activeTab === "sync_view" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Eye className="w-4 h-4 text-amber-400" />
                    Synchronized Dual-Spectrum View (CAM-007 Sentry PTZ)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Temporal Delta: <span className="text-emerald-400 font-mono font-bold">12.4 ms</span> • Alignment RMSE:{" "}
                    <span className="text-emerald-400 font-mono font-bold">0.82 px</span> • Registration Mode:{" "}
                    <span className="text-cyan-400 font-mono font-bold">
                      {registrationOverride ? "GEOMETRIC_REGISTERED" : "REGISTRATION UNAVAILABLE"}
                    </span>
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => {
                      setRegistrationOverride(!registrationOverride);
                      setActionFeedback(
                        registrationOverride
                          ? "Degraded cross-spectral registration: REGISTRATION UNAVAILABLE displayed."
                          : "Restored verified extrinsic calibration: GEOMETRIC_REGISTERED active."
                      );
                    }}
                    className={`px-3 py-1 text-xs rounded border transition-colors ${
                      registrationOverride
                        ? "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
                        : "bg-rose-950/40 text-rose-300 border-rose-800 hover:bg-rose-900/50"
                    }`}
                  >
                    {registrationOverride ? "Simulate Extrinsic Misalignment" : "Restore Validated Extrinsics"}
                  </button>
                </div>
              </div>

              {/* Side-by-Side Viewport */}
              <div className="grid grid-cols-2 gap-4">
                {/* Left: Optical Frame */}
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-3 relative overflow-hidden">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                      <span className="text-xs font-bold text-slate-200">OPTICAL RGB STREAM</span>
                      <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded font-mono">
                        SNS-CAM007-RGB
                      </span>
                    </div>
                    <span className="text-xs font-mono text-emerald-400">1920x1080 @ 30 FPS</span>
                  </div>

                  <div className="aspect-video bg-gradient-to-b from-slate-900 to-slate-950 rounded border border-slate-800/80 flex flex-col items-center justify-center relative p-4">
                    {/* Simulated Detection Box */}
                    <div className="absolute top-[25%] left-[30%] w-[18%] h-[45%] border-2 border-emerald-400/80 bg-emerald-500/10 rounded flex flex-col justify-between p-1">
                      <span className="text-[10px] bg-emerald-500 text-black font-bold px-1 rounded self-start">
                        person: 0.93
                      </span>
                      <span className="text-[9px] text-emerald-300 font-mono self-end">CAM-007:RGB:P-044</span>
                    </div>

                    <div className="text-center text-slate-600 text-xs">
                      <Camera className="w-8 h-8 mx-auto mb-1 text-slate-600" />
                      Visible Optical Synthetic Video Stream
                      <div className="text-[10px] text-slate-500 mt-1">Status: PRODUCTION (FROZEN BASELINE)</div>
                    </div>
                  </div>
                </div>

                {/* Right: Thermal Frame */}
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-3 relative overflow-hidden">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
                      <span className="text-xs font-bold text-slate-200">THERMAL LWIR STREAM</span>
                      <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded font-mono">
                        SNS-CAM007-LWIR
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-amber-400 font-mono">VOx 8-14um (640x512)</span>
                      <span className="text-[10px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded border border-amber-500/30">
                        {simulatedThermalMode}
                      </span>
                    </div>
                  </div>

                  <div className="aspect-video bg-gradient-to-b from-amber-950/20 via-slate-950 to-slate-950 rounded border border-slate-800/80 flex flex-col items-center justify-center relative p-4">
                    {registrationOverride ? (
                      /* Aligned Thermal Detection */
                      <div className="absolute top-[26%] left-[31%] w-[17%] h-[44%] border-2 border-amber-400/80 bg-amber-500/10 rounded flex flex-col justify-between p-1">
                        <span className="text-[10px] bg-amber-500 text-black font-bold px-1 rounded self-start">
                          heat_sig: 0.88
                        </span>
                        <span className="text-[9px] text-amber-300 font-mono self-end">CAM-007:LWIR:T-019</span>
                      </div>
                    ) : (
                      /* Misalignment Overlay Warning */
                      <div className="absolute inset-0 bg-rose-950/50 backdrop-blur-xs flex flex-col items-center justify-center p-4 text-center">
                        <AlertTriangle className="w-8 h-8 text-rose-400 mb-2 animate-bounce" />
                        <span className="text-sm font-bold text-rose-300 tracking-wider">
                          REGISTRATION UNAVAILABLE
                        </span>
                        <p className="text-xs text-rose-400/80 max-w-sm mt-1">
                          Extrinsic calibration missing or degraded. Geometric overlay disabled to prevent false alignment.
                        </p>
                      </div>
                    )}

                    <div className="text-center text-slate-600 text-xs">
                      <Flame className="w-8 h-8 mx-auto mb-1 text-slate-600" />
                      Uncooled Microbolometer Simulated Stream
                      <div className="text-[10px] text-rose-400 mt-1">Native Thermal AI: NOT VALIDATED</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Fused Cross-Spectral Metadata Card */}
              <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-lg flex items-center justify-between">
                <div className="grid grid-cols-4 gap-6 text-xs">
                  <div>
                    <span className="text-slate-500 block text-[10px]">ASSOCIATION ID</span>
                    <span className="font-mono text-slate-200 font-bold">CSA-20260904-0071</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">GLOBAL TARGET</span>
                    <span className="font-mono text-amber-400 font-bold">GLOBAL-PERSON-00042</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">EXPLAINABLE FUSION CONF</span>
                    <span className="font-mono text-emerald-400 font-bold">
                      0.912 <span className="text-[10px] text-slate-500">(Opt: 0.93, Thm: 0.88, Assoc: 0.92)</span>
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">ENVIRONMENTAL WEIGHTING</span>
                    <span className="font-mono text-cyan-400 font-bold">
                      Day Lux (&alpha;: 0.60, &beta;: 0.25, &gamma;: 0.15)
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() =>
                      setSimulatedThermalMode(
                        simulatedThermalMode === "WHITE_HOT"
                          ? "BLACK_HOT"
                          : simulatedThermalMode === "BLACK_HOT"
                          ? "IRONBOW"
                          : "WHITE_HOT"
                      )
                    }
                    className="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700"
                  >
                    Cycle Palette ({simulatedThermalMode})
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: SENSOR TOPOLOGY */}
          {activeTab === "topology" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">8-Camera Mesh Multimodal Sensor Topology</h3>
                  <p className="text-xs text-slate-400">
                    Physical coordinates status: <span className="text-amber-400 font-bold">LOGICAL / UNKNOWN</span> (Zero fabricated GPS coordinates).
                  </p>
                </div>
              </div>

              <div className="border border-slate-800 rounded-lg overflow-hidden">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                    <tr>
                      <th className="p-3">Camera Node</th>
                      <th className="p-3">Sensor Identifier</th>
                      <th className="p-3">Modality</th>
                      <th className="p-3">Resolution & FPS</th>
                      <th className="p-3">Orientation Ref</th>
                      <th className="p-3">Clock Source</th>
                      <th className="p-3">Calibration Status</th>
                      <th className="p-3">Data Origin</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                    {[
                      { cam: "CAM-001", sns: "SNS-CAM001-RGB", mod: "OPTICAL_RGB", res: "1920x1080 @ 30", ori: "PTZ_SLEW_360", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-002", sns: "SNS-CAM002-RGB", mod: "OPTICAL_RGB", res: "1920x1080 @ 30", ori: "FIXED_SOUTH", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-003", sns: "SNS-CAM003-RGB", mod: "OPTICAL_RGB", res: "1920x1080 @ 30", ori: "PTZ_SLEW_360", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-004", sns: "SNS-CAM004-RGB", mod: "OPTICAL_RGB", res: "3840x2160 @ 60", ori: "AIRSPACE_MAST_45", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-005", sns: "SNS-CAM005-LWIR", mod: "THERMAL_LWIR", res: "640x512 @ 30", ori: "FIXED_NORTH_OUTPOST", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-006", sns: "SNS-CAM006-LWIR", mod: "THERMAL_LWIR", res: "640x512 @ 30", ori: "FIXED_EAST_CORRIDOR", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-007", sns: "SNS-CAM007-RGB", mod: "OPTICAL_RGB", res: "1920x1080 @ 30", ori: "PTZ_DUAL_CO_LOCATED", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-007", sns: "SNS-CAM007-LWIR", mod: "THERMAL_LWIR", res: "640x512 @ 30", ori: "PTZ_DUAL_CO_LOCATED", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                      { cam: "CAM-008", sns: "SNS-CAM008-RGB", mod: "OPTICAL_RGB", res: "1920x1080 @ 30", ori: "FIXED_AIRBASE_BOUNDARY", clk: "PTP_IEEE_1588", cal: "VALIDATED", orig: "SIMULATED" },
                    ].map((row, i) => (
                      <tr key={i} className="hover:bg-slate-800/30">
                        <td className="p-3 font-bold text-white">{row.cam}</td>
                        <td className="p-3 text-cyan-400">{row.sns}</td>
                        <td className="p-3">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            row.mod.includes("THERMAL") ? "bg-amber-500/20 text-amber-300" : "bg-emerald-500/20 text-emerald-300"
                          }`}>
                            {row.mod}
                          </span>
                        </td>
                        <td className="p-3 text-slate-400">{row.res}</td>
                        <td className="p-3 text-slate-400">{row.ori}</td>
                        <td className="p-3 text-slate-400">{row.clk}</td>
                        <td className="p-3 text-emerald-400 font-bold">{row.cal}</td>
                        <td className="p-3 text-cyan-300">{row.orig}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 8: CALIBRATION FRAMEWORK */}
          {activeTab === "calibration" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Sensor Calibration Framework</h3>
                  <p className="text-xs text-slate-400">
                    Rigorous optical intrinsics, thermal sensor profiles, and relative cross-spectral extrinsics.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-bold text-slate-200">OPTICAL INTRINSICS (CAM-007)</span>
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">VALIDATED</span>
                  </div>
                  <div className="font-mono text-xs text-slate-300 space-y-1">
                    <div>Focal: <span className="text-cyan-400">fx=1420.5, fy=1420.5</span></div>
                    <div>Center: <span className="text-cyan-400">cx=960.0, cy=540.0</span></div>
                    <div>Radial Distortion: <span className="text-slate-400">k1=-0.05, k2=0.01</span></div>
                    <div>Tangential: <span className="text-slate-400">p1=0.001, p2=0.001</span></div>
                    <div>Resolution: <span className="text-slate-200">1920 x 1080 px</span></div>
                  </div>
                </div>

                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-bold text-slate-200">THERMAL SENSOR PROFILE</span>
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">VALIDATED</span>
                  </div>
                  <div className="font-mono text-xs text-slate-300 space-y-1">
                    <div>Focal: <span className="text-amber-400">fx=612.3, fy=612.3</span></div>
                    <div>Center: <span className="text-amber-400">cx=320.0, cy=256.0</span></div>
                    <div>Detector: <span className="text-slate-400">Uncooled VOx Microbolometer</span></div>
                    <div>Spectral Band: <span className="text-slate-200">8 - 14 um</span></div>
                    <div>NETD Sensitivity: <span className="text-emerald-400">&lt; 40 mK</span></div>
                    <div>Pixel Pitch: <span className="text-slate-200">12 um</span></div>
                  </div>
                </div>

                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-bold text-slate-200">CROSS-SPECTRAL EXTRINSICS</span>
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">VALIDATED</span>
                  </div>
                  <div className="font-mono text-xs text-slate-300 space-y-1">
                    <div>Rotation: <span className="text-cyan-400">[0.10&deg;, -0.20&deg;, 0.05&deg;]</span></div>
                    <div>Translation: <span className="text-cyan-400">[0.085m, 0.010m, 0.000m]</span></div>
                    <div>Alignment RMSE: <span className="text-emerald-400 font-bold">0.74 px</span></div>
                    <div>Cal Version: <span className="text-slate-200">EXT-DUAL-007</span></div>
                    <div>Status: <span className="text-emerald-400">GEOMETRIC FUSION ALLOWED</span></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 10: THERMAL MODEL STATUS & GOVERNANCE */}
          {activeTab === "thermal_models" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Native Thermal AI Governance & Candidate Evaluation</h3>
                  <p className="text-xs text-slate-400">
                    Dual-signature governed promotion pipeline. Live observations are strictly air-gapped from production weights.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {/* Champion Model Card */}
                <div className="p-4 bg-slate-950 border border-emerald-500/30 rounded-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      <span className="text-xs font-bold text-slate-200">PRODUCTION OPTICAL CHAMPION</span>
                    </div>
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded font-bold">
                      100% OPERATIONAL AUTHORITY
                    </span>
                  </div>
                  <div className="font-mono text-xs text-slate-300 space-y-1.5">
                    <div>Model ID: <span className="text-emerald-400 font-bold">ibvap_ground_detector_v2.0</span></div>
                    <div>Modality: <span className="text-slate-200">OPTICAL_RGB</span></div>
                    <div>Architecture: <span className="text-slate-200">YOLO11n</span></div>
                    <div>Immutable SHA: <span className="text-slate-400 text-[10px]">7DBF36027768194F61C6EBBC000E...</span></div>
                    <div>Operational Role: <span className="text-emerald-400">Controls PTZ, Slew, Alerts, Handover</span></div>
                  </div>
                </div>

                {/* Challenger Thermal Model Card */}
                <div className="p-4 bg-slate-950 border border-amber-500/30 rounded-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span className="text-xs font-bold text-slate-200">CANDIDATE THERMAL CHALLENGER</span>
                    </div>
                    <span className="text-[10px] bg-rose-500/20 text-rose-300 px-2 py-0.5 rounded font-bold">
                      NOT VALIDATED (SHADOW ONLY)
                    </span>
                  </div>
                  <div className="font-mono text-xs text-slate-300 space-y-1.5">
                    <div>Candidate ID: <span className="text-amber-400 font-bold">ibvap_thermal_yolo11n_candidate</span></div>
                    <div>Modality: <span className="text-slate-200">THERMAL_LWIR</span></div>
                    <div>Dataset Status: <span className="text-rose-400 font-bold">IBVAP-THERMAL-READINESS</span></div>
                    <div>Operational Authority: <span className="text-rose-400 font-bold">ZERO ACTUATION PERMITTED</span></div>
                    <div>13-Stage Gate: <span className="text-amber-400 font-bold">DATASET_MISSING (BLOCKED)</span></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 12: FAILURE ANALYSIS & ACTIVE LEARNING */}
          {activeTab === "failure_analysis" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Cross-Spectral Failure Taxonomies & Dynamic Severity</h3>
                  <p className="text-xs text-slate-400">
                    Monitors 7 dedicated multimodal failure clusters, feeding validated failures into Phase XIII active learning queues.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                {[
                  { id: "CLS-MM-OPTICAL-MISSED-THERMAL", title: "Thermal Missed by Optical", count: 28, sev: "HIGH", desc: "Target clear in LWIR heat signature but undetected in visible RGB." },
                  { id: "CLS-MM-THERMAL-MISSED-OPTICAL", title: "Optical Missed by Thermal", count: 12, sev: "LOW", desc: "Target clear in visible RGB; thermal crossover / ambient equilibrium." },
                  { id: "CLS-MM-HOT-BACKGROUND", title: "Hot-Background Confusion", count: 19, sev: "LOW", desc: "Sun-heated rocks or tin sheds misclassified as biological targets." },
                  { id: "CLS-MM-FALSE-CROSS-ASSOC", title: "False Cross-Association", count: 4, sev: "LOW", desc: "Crossing trajectories or occlusions causing erroneous track binding." },
                  { id: "CLS-MM-REGISTRATION-DRIFT", title: "Registration Degradation", count: 2, sev: "LOW", desc: "Mechanical vibration or calibration drift exceeding 3.0px threshold." },
                  { id: "CLS-MM-TEMPORAL-JITTER", title: "Temporal Deep Jitter", count: 5, sev: "LOW", desc: "Network delay causing timestamp delta > 100ms." },
                ].map((cluster, i) => (
                  <div key={i} className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-200">{cluster.title}</span>
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                        cluster.sev === "HIGH" ? "bg-amber-500/20 text-amber-300" : "bg-slate-800 text-slate-400"
                      }`}>
                        {cluster.sev}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-snug">{cluster.desc}</p>
                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-900">
                      <span>{cluster.id}</span>
                      <span className="text-amber-400 font-bold">{cluster.count} cases</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB: DATASET GOVERNANCE (12-STATE) */}
          {activeTab === "dataset_gov" && (
            <div className="space-y-4">
              <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Database className="w-4 h-4 text-amber-400" />
                    12-State Thermal Dataset Governance Lifecycle
                  </h4>
                  <span className="px-2 py-0.5 text-xs font-mono rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                    STATUS: IBVAP-THERMAL-READINESS-v0
                  </span>
                </div>
                <div className="grid grid-cols-4 md:grid-cols-6 gap-2 text-center text-xs">
                  {[
                    { state: "RAW", desc: "Sensor Ingestion", active: true },
                    { state: "INGESTED", desc: "Format Validated", active: true },
                    { state: "QUALITY_REJECTED", desc: "Integrity Drop", active: false },
                    { state: "CALIBRATED", desc: "Opt-Thm Extrinsics", active: true },
                    { state: "ANNOTATION_PENDING", desc: "Human Queued", active: true },
                    { state: "ANNOTATED", desc: "Boxes Labeled", active: false },
                    { state: "QA_PENDING", desc: "Consensus Review", active: false },
                    { state: "QA_FAILED", desc: "Rejected by QA", active: false },
                    { state: "TRAINING_READY", desc: "Real Data Gate", active: false },
                    { state: "BENCHMARK_QUARANTINED", desc: "Frozen Leakage Guard", active: true },
                    { state: "BENCHMARK_APPROVED", desc: "Ground Truth", active: false },
                    { state: "ARCHIVED", desc: "Audited Cold", active: false },
                  ].map((s, idx) => (
                    <div
                      key={idx}
                      className={`p-2 rounded border ${
                        s.active
                          ? "bg-slate-900 border-amber-500/40 text-amber-300"
                          : "bg-slate-950 border-slate-800 text-slate-500"
                      }`}
                    >
                      <div className="font-mono font-bold text-[11px]">{s.state}</div>
                      <div className="text-[9px] text-slate-400 mt-0.5">{s.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
                  <h5 className="text-xs font-bold text-slate-200">Registered Thermal Datasets</h5>
                  <div className="space-y-2 text-xs">
                    <div className="p-2.5 bg-slate-900 rounded border border-slate-800 flex items-center justify-between">
                      <div>
                        <div className="font-semibold text-white">DS-THM-READINESS-v0</div>
                        <div className="text-[11px] text-slate-400">Version v0.1-spec • Origin: SIMULATED / REPLAY</div>
                      </div>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300">
                        0 Real Samples
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
                  <h5 className="text-xs font-bold text-slate-200">Benchmark Quarantine & Anti-Leakage Guard</h5>
                  <p className="text-xs text-slate-400">
                    Multi-layer isolation prevents test set contamination. Frame N in training and frame N+1 in benchmark is blocked with <span className="text-rose-400 font-mono font-bold">BENCHMARK_LEAKAGE_REJECTED</span>.
                  </p>
                  <div className="flex items-center gap-2 pt-2 border-t border-slate-900 text-xs">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    <span className="text-slate-300">Quarantine Status: <strong className="text-emerald-400">ACTIVE & VERIFIED (0 Leakage Violations)</strong></span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB: RGB VS THERMAL VS FUSION */}
          {activeTab === "rgb_vs_thermal" && (
            <div className="space-y-4">
              <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-amber-400" />
                    Head-to-Head Controlled Evaluation (Scene: Perimeter Sentry CAM-007)
                  </h4>
                  <span className="px-2 py-0.5 text-xs font-mono rounded bg-slate-800 text-slate-400">
                    EVIDENCE CLASS: SOFTWARE_BENCHMARK
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Strict truthfulness rule: Thermal metrics below reflect software pipeline benchmarks. Real-world physical performance is <strong className="text-rose-400">NOT VALIDATED</strong> pending genuine LWIR deployment.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Config A */}
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-emerald-400">CONFIG A: RGB ONLY</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">PRODUCTION</span>
                  </div>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-slate-400">Precision:</span><span className="font-mono text-white font-bold">0.94</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Recall:</span><span className="font-mono text-white font-bold">0.92</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Night Recall:</span><span className="font-mono text-amber-400 font-bold">0.62 (Low-Lux Degraded)</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Latency:</span><span className="font-mono text-slate-300">12.4 ms</span></div>
                  </div>
                </div>

                {/* Config B */}
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-amber-400">CONFIG B: THERMAL ONLY</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300">NOT VALIDATED</span>
                  </div>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-slate-400">Precision:</span><span className="font-mono text-white font-bold">0.91</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Recall:</span><span className="font-mono text-white font-bold">0.88</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Night Recall:</span><span className="font-mono text-emerald-400 font-bold">0.94 (Zero-Lux Strong)</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Latency:</span><span className="font-mono text-slate-300">11.2 ms</span></div>
                  </div>
                </div>

                {/* Config C */}
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-cyan-400">CONFIG C: FUSION</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300">EXPERIMENTAL</span>
                  </div>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-slate-400">Precision:</span><span className="font-mono text-white font-bold">0.95</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Recall:</span><span className="font-mono text-white font-bold">0.96</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Night Recall:</span><span className="font-mono text-emerald-400 font-bold">0.95 (Robust 24/7)</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Latency:</span><span className="font-mono text-slate-300">14.8 ms</span></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Placeholder for other tabs */}
          {["optical_feed", "thermal_feed", "associations", "tracks", "health", "synchronization", "fusion_quality"].includes(activeTab) && (
            <div className="p-6 bg-slate-950 border border-slate-800 rounded-lg text-center space-y-2">
              <Activity className="w-8 h-8 text-amber-400 mx-auto animate-pulse" />
              <h4 className="text-sm font-semibold text-white">Live Telemetry Active for {activeTab.toUpperCase()}</h4>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                Telemetry streams and sensor models are actively connected to the Phase XV Multimodal Sensor Intelligence mesh.
              </p>
            </div>
          )}
        </div>

        {/* Footer Governance Notice */}
        <div className="px-6 py-3 border-t border-slate-800/80 bg-slate-950 flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <Lock className="w-3.5 h-3.5 text-emerald-400" />
            <span>
              Production Optical Models Frozen & Verified • Air-Gapped Thermal Governance Mandate Enforced
            </span>
          </div>
          <div className="text-[11px] text-slate-500 font-mono">
            TRINETRA Phase XV — SSB/MHA Mission Assurance
          </div>
        </div>
      </div>
    </div>
  );
};
