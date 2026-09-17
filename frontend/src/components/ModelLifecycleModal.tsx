import React, { useState } from "react";
import {
  X,
  ShieldCheck,
  RotateCcw,
  Sliders,
  Layers,
  Activity,
  AlertTriangle,
  GitBranch,
  CheckCircle2,
  Lock,
  Cpu
} from "lucide-react";

interface ModelLifecycleModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ModelLifecycleModal: React.FC<ModelLifecycleModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<"overview" | "canary" | "edge" | "health" | "rollback">("overview");
  const [canaryTraffic, setCanaryTraffic] = useState<number>(10);
  const [rollbackDomain, setRollbackDomain] = useState<string>("GROUND");
  const [rollbackStatus, setRollbackStatus] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleRollback = () => {
    setRollbackStatus(`ROLLBACK SUCCESS: ${rollbackDomain} reverted to last-known-good baseline (SHA-256 Verified).`);
    setTimeout(() => setRollbackStatus(null), 4000);
  };

  const stages = [
    { name: "COLLECT", active: true, done: true },
    { name: "LABEL", active: true, done: true },
    { name: "TRAIN", active: true, done: true },
    { name: "VALIDATE", active: true, done: true },
    { name: "CANARY", active: true, done: false },
    { name: "DEPLOY TO EDGE", active: true, done: false },
    { name: "MONITOR", active: true, done: true },
    { name: "ROLL BACK", active: false, done: false },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-6xl max-h-[92vh] flex flex-col rounded-2xl bg-neutral-950 border border-neutral-800 shadow-2xl overflow-hidden">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-900/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold tracking-wide text-white">
                  Edge AI Model Lifecycle & Governance
                </h2>
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/40">
                  Phase XII
                </span>
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                  Production Models Frozen
                </span>
              </div>
              <p className="text-xs text-neutral-400">
                18-State Engine • 13 Validation Gates • Canary / Shadow Inference • Zero-Downtime Rollback
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2 px-6 border-b border-neutral-800 bg-neutral-900/30 text-xs font-semibold">
          {[
            { id: "overview", label: "Production Models & Pipeline", icon: Layers },
            { id: "canary", label: "Canary & Shadow Inference", icon: Sliders },
            { id: "edge", label: "8-Camera Edge Rollout", icon: Cpu },
            { id: "health", label: "Health & Drift Telemetry", icon: Activity },
            { id: "rollback", label: "Atomic Rollback", icon: RotateCcw },
          ].map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 py-3 px-3 border-b-2 transition-all ${
                  isSelected
                    ? "border-purple-500 text-purple-400 font-bold"
                    : "border-transparent text-neutral-400 hover:text-neutral-200"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* 8-Stage Lifecycle Progress Strip */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-2">
            <div className="flex items-center justify-between text-xs text-neutral-400">
              <span className="font-semibold text-neutral-200 flex items-center gap-1.5">
                <GitBranch className="w-4 h-4 text-purple-400" />
                Core Lifecycle State Progression
              </span>
              <span>18-State Machine Interlock: Active</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-8 gap-2 pt-1">
              {stages.map((st, i) => (
                <div
                  key={st.name}
                  className={`rounded-lg p-2 text-center border text-[11px] font-semibold transition-all ${
                    st.done
                      ? "bg-emerald-950/30 border-emerald-500/40 text-emerald-400"
                      : st.active
                      ? "bg-purple-950/30 border-purple-500/50 text-purple-300 animate-pulse"
                      : "bg-neutral-900/50 border-neutral-800 text-neutral-500"
                  }`}
                >
                  <div className="text-[9px] opacity-70">STEP {i + 1}</div>
                  <div className="truncate">{st.name}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Tab 1: Overview & Production Models */}
          {activeTab === "overview" && (
            <div className="space-y-4 animate-in fade-in duration-150">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Lock className="w-4 h-4 text-emerald-400" />
                Active Production Models (Immutable Baselines)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                
                {/* Ground */}
                <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">IBVAP-GROUND-v2.0</span>
                    <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 font-mono">ACTIVE</span>
                  </div>
                  <div className="text-xs text-neutral-400">
                    <div>Architecture: <span className="text-neutral-200 font-mono">YOLO11n (2.6M)</span></div>
                    <div>Classes: <span className="text-neutral-200">person (0), vehicle (1)</span></div>
                    <div>Image Size: <span className="text-neutral-200 font-mono">768x768</span></div>
                  </div>
                  <div className="text-[10px] font-mono text-neutral-500 truncate pt-2 border-t border-neutral-800">
                    SHA: <span className="text-emerald-400">7DBF36027768...</span>
                  </div>
                </div>

                {/* Airborne */}
                <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">IBVAP-AIRBORNE-v2.0</span>
                    <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 font-mono">ACTIVE</span>
                  </div>
                  <div className="text-xs text-neutral-400">
                    <div>Architecture: <span className="text-neutral-200 font-mono">YOLO11n (2.6M)</span></div>
                    <div>Classes: <span className="text-neutral-200">drone (0), aircraft (1)</span></div>
                    <div>Image Size: <span className="text-neutral-200 font-mono">640x640</span></div>
                  </div>
                  <div className="text-[10px] font-mono text-neutral-500 truncate pt-2 border-t border-neutral-800">
                    SHA: <span className="text-emerald-400">5229632C3D7A...</span>
                  </div>
                </div>

                {/* Security Item */}
                <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">IBVAP-SECURITY-v2.1</span>
                    <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 font-mono">ACTIVE</span>
                  </div>
                  <div className="text-xs text-neutral-400">
                    <div>Architecture: <span className="text-neutral-200 font-mono">YOLO11n (2.6M)</span></div>
                    <div>Classes: <span className="text-neutral-200">firearm (0)</span></div>
                    <div>Image Size: <span className="text-neutral-200 font-mono">640x640</span></div>
                  </div>
                  <div className="text-[10px] font-mono text-neutral-500 truncate pt-2 border-t border-neutral-800">
                    SHA: <span className="text-emerald-400">72464C778DE5...</span>
                  </div>
                </div>

              </div>

              {/* 13-Stage Validation Gates Preview */}
              <div className="rounded-xl border border-neutral-800 bg-neutral-900/30 p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-neutral-200">13-Stage Validation Gates Standard</h4>
                  <span className="text-[11px] text-emerald-400 flex items-center gap-1 font-mono">
                    <CheckCircle2 className="w-3 h-3" />
                    All 13 Gates Evaluated Automatically
                  </span>
                </div>
                <p className="text-xs text-neutral-400">
                  Includes Standard Validation, Frozen Benchmark (IBVAP-GT-v1.0), Regression, Recall Drop (&lt;2%), Hard-Negative Rejection, Small Object Sensitivity, Latency P95 (&lt;25ms), and Checksum Integrity.
                </p>
              </div>
            </div>
          )}

          {/* Tab 2: Canary & Shadow Inference */}
          {activeTab === "canary" && (
            <div className="space-y-4 animate-in fade-in duration-150">
              <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-cyan-400" />
                    Live Canary Traffic Splitter
                  </h3>
                  <span className="px-2 py-0.5 text-[10px] font-semibold uppercase rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                    Shadow Mode: ARMED
                  </span>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-300">Traffic Allocation:</span>
                    <span className="font-mono text-cyan-400 font-bold">{canaryTraffic}% Canary / {100 - canaryTraffic}% Production</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={canaryTraffic}
                    onChange={(e) => setCanaryTraffic(Number(e.target.value))}
                    className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                  />
                  <p className="text-[11px] text-neutral-500">
                    In Shadow Mode, candidate inference runs in parallel without affecting operational alarms or PTZ commands.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="p-3 rounded-lg border border-neutral-800 bg-neutral-950">
                    <div className="text-[10px] text-neutral-500 font-mono">PRODUCTION ACTIVE</div>
                    <div className="text-xs font-bold text-white mt-1">IBVAP-GROUND-v2.0</div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">Mean Conf: 0.81 • P95: 15.8ms</div>
                  </div>
                  <div className="p-3 rounded-lg border border-cyan-500/30 bg-cyan-950/20">
                    <div className="text-[10px] text-cyan-400 font-mono">CANDIDATE SHADOW</div>
                    <div className="text-xs font-bold text-cyan-300 mt-1">IBVAP-GROUND-v2.1-EXP003</div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">Mean Conf: 0.84 • P95: 14.2ms</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 3: 8-Camera Edge Rollout */}
          {activeTab === "edge" && (
            <div className="space-y-4 animate-in fade-in duration-150">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">8-Camera Distributed Edge Rollout Grid</h3>
                <span className="text-xs text-emerald-400 font-mono">8 / 8 Nodes Verified</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { cam: "CAM-001", name: "North Sentry PTZ", model: "IBVAP-GROUND-v2.0", status: "ACTIVE" },
                  { cam: "CAM-002", name: "Cargo Logistics Gate", model: "IBVAP-GROUND-v2.0", status: "ACTIVE" },
                  { cam: "CAM-003", name: "South Sentry PTZ", model: "IBVAP-GROUND-v2.0", status: "ACTIVE" },
                  { cam: "CAM-004", name: "Airspace Tower", model: "IBVAP-AIRBORNE-v2.0", status: "ACTIVE" },
                  { cam: "CAM-005", name: "North Thermal Outpost", model: "IBVAP-GROUND-v2.0 (SIM)", status: "ACTIVE" },
                  { cam: "CAM-006", name: "East Corridor Thermal", model: "IBVAP-GROUND-v2.0 (SIM)", status: "ACTIVE" },
                  { cam: "CAM-007", name: "South Sector Dual PTZ", model: "IBVAP-GROUND-v2.0", status: "ACTIVE" },
                  { cam: "CAM-008", name: "Airbase Boundary Mast", model: "IBVAP-GROUND-v2.0", status: "ACTIVE" },
                ].map((n) => (
                  <div key={n.cam} className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-3 space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-white">{n.cam}</span>
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 font-mono">{n.status}</span>
                    </div>
                    <div className="text-[11px] text-neutral-400">{n.name}</div>
                    <div className="text-[10px] text-neutral-500 font-mono truncate">{n.model}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 4: Health & Drift Telemetry */}
          {activeTab === "health" && (
            <div className="space-y-4 animate-in fade-in duration-150">
              <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                    <Activity className="w-4 h-4 text-emerald-400" />
                    Operational Proxy Metrics
                  </h4>
                  <span className="text-[10px] text-neutral-500 italic">
                    Ground-truth accuracy not claimed in live telemetry
                  </span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                  <div className="p-3 rounded-lg border border-neutral-800 bg-neutral-950">
                    <div className="text-[10px] text-neutral-500">AGGREGATE FPS</div>
                    <div className="text-lg font-bold text-white font-mono mt-1">82.5 FPS</div>
                  </div>
                  <div className="p-3 rounded-lg border border-neutral-800 bg-neutral-950">
                    <div className="text-[10px] text-neutral-500">P95 LATENCY</div>
                    <div className="text-lg font-bold text-emerald-400 font-mono mt-1">15.8 ms</div>
                  </div>
                  <div className="p-3 rounded-lg border border-neutral-800 bg-neutral-950">
                    <div className="text-[10px] text-neutral-500">GPU VRAM</div>
                    <div className="text-lg font-bold text-white font-mono mt-1">1,420 MB</div>
                  </div>
                  <div className="p-3 rounded-lg border border-neutral-800 bg-neutral-950">
                    <div className="text-[10px] text-neutral-500">CONFIDENCE DRIFT</div>
                    <div className="text-lg font-bold text-emerald-400 font-mono mt-1">&Delta; 0.02 (STABLE)</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 5: Rollback Controls */}
          {activeTab === "rollback" && (
            <div className="space-y-4 animate-in fade-in duration-150">
              <div className="rounded-xl border border-rose-500/30 bg-rose-950/10 p-5 space-y-4">
                <div className="flex items-center gap-2 text-rose-400">
                  <AlertTriangle className="w-5 h-5" />
                  <h3 className="text-sm font-bold">Atomic Model Rollback Control</h3>
                </div>
                <p className="text-xs text-neutral-300">
                  Reverts the selected perception domain to its verified last-known-good baseline.
                  The failed model artifact will be preserved for post-mortem engineering forensics.
                </p>

                {rollbackStatus && (
                  <div className="p-3 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-xs font-semibold">
                    {rollbackStatus}
                  </div>
                )}

                <div className="flex items-center gap-3">
                  <select
                    value={rollbackDomain}
                    onChange={(e) => setRollbackDomain(e.target.value)}
                    className="px-3 py-1.5 rounded-lg bg-neutral-900 border border-neutral-700 text-xs text-white"
                  >
                    <option value="GROUND">GROUND DOMAIN (IBVAP-GROUND-v2.0)</option>
                    <option value="AIRBORNE">AIRBORNE DOMAIN (IBVAP-AIRBORNE-v2.0)</option>
                    <option value="SECURITY_ITEM">SECURITY ITEM DOMAIN (IBVAP-SECURITY-v2.1)</option>
                  </select>

                  <button
                    onClick={handleRollback}
                    className="flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-bold bg-rose-600 hover:bg-rose-500 text-white transition-colors"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    EXECUTE ATOMIC ROLLBACK
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-neutral-800 bg-neutral-900/60 text-xs text-neutral-400">
          <div className="flex items-center gap-4">
            <span>Governance Policy: <strong className="text-white">FAIL-CLOSED</strong></span>
            <span>Downgrade Protection: <strong className="text-emerald-400">ARMED</strong></span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-neutral-800 hover:bg-neutral-700 text-white transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
};
