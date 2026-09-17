import React, { useState } from "react";
import {
  X,
  Share2,
  Cpu,
  Wifi,
  WifiOff,
  Lock,
  Unlock,
  Eye,
  Flame,
  CheckCircle2,
  Clock,
  ArrowRight,
  Database,
  Layers
} from "lucide-react";

interface EdgeNodeCard {
  node_id: string;
  camera_id: string;
  name: string;
  sensor_type: "OPTICAL" | "THERMAL" | "DUAL";
  health: "ONLINE" | "DEGRADED" | "OFFLINE";
  network_state: "NORMAL" | "PARTITIONED" | "DEGRADED" | "RECOVERING";
  lease_valid: boolean;
  lease_expires_in_sec: number;
  outbox_pending: number;
  ptz: boolean;
}

interface EdgeMeshModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const EdgeMeshModal: React.FC<EdgeMeshModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [globalLock, setGlobalLock] = useState<boolean>(false);
  const [partitionMode, setPartitionMode] = useState<"NORMAL" | "PARTITIONED" | "RECOVERING">("NORMAL");
  const [syncStatus, setSyncStatus] = useState<string>("SYNCHRONIZED");

  const [nodes, setNodes] = useState<EdgeNodeCard[]>([
    { node_id: "NODE-CAM-001", camera_id: "CAM-001", name: "North Sentry PTZ", sensor_type: "OPTICAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: true },
    { node_id: "NODE-CAM-002", camera_id: "CAM-002", name: "Cargo Logistics Gate", sensor_type: "OPTICAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: false },
    { node_id: "NODE-CAM-003", camera_id: "CAM-003", name: "South Sentry PTZ", sensor_type: "OPTICAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: true },
    { node_id: "NODE-CAM-004", camera_id: "CAM-004", name: "Airspace Tower", sensor_type: "OPTICAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: false },
    { node_id: "NODE-CAM-005", camera_id: "CAM-005", name: "North Thermal Outpost", sensor_type: "THERMAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: false },
    { node_id: "NODE-CAM-006", camera_id: "CAM-006", name: "East Corridor Thermal", sensor_type: "THERMAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: false },
    { node_id: "NODE-CAM-007", camera_id: "CAM-007", name: "South Sector Dual PTZ", sensor_type: "DUAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: true },
    { node_id: "NODE-CAM-008", camera_id: "CAM-008", name: "Airbase Boundary Mast", sensor_type: "OPTICAL", health: "ONLINE", network_state: "NORMAL", lease_valid: true, lease_expires_in_sec: 284, outbox_pending: 0, ptz: false },
  ]);

  if (!isOpen) return null;

  const toggleGlobalLock = () => {
    setGlobalLock(!globalLock);
  };

  const simulatePartition = () => {
    setPartitionMode("PARTITIONED");
    setNodes((prev) =>
      prev.map((n) => ({ ...n, network_state: "PARTITIONED", outbox_pending: n.outbox_pending + 1 }))
    );
    setSyncStatus("OFFLINE BUFFERING");
  };

  const recoverNetwork = () => {
    setPartitionMode("RECOVERING");
    setTimeout(() => {
      setPartitionMode("NORMAL");
      setNodes((prev) =>
        prev.map((n) => ({ ...n, network_state: "NORMAL", outbox_pending: 0 }))
      );
      setSyncStatus("RECONCILED & HASH VERIFIED (SHA-256 ✓)");
    }, 1000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-6xl max-h-[90vh] flex flex-col rounded-2xl bg-neutral-950 border border-neutral-800 shadow-2xl overflow-hidden">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-900/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold tracking-wide text-white">
                  Edge-Assisted Distributed Camera Mesh
                </h2>
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/40">
                  Phase XI
                </span>
                <span className={`px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase rounded-full border ${
                  partitionMode === "NORMAL" 
                    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                    : partitionMode === "PARTITIONED"
                    ? "bg-rose-500/20 text-rose-400 border-rose-500/40 animate-pulse"
                    : "bg-amber-500/20 text-amber-400 border-amber-500/40"
                }`}>
                  Mesh: {partitionMode}
                </span>
              </div>
              <p className="text-xs text-neutral-400">
                Peer-to-Peer Handover Negotiation • Short-Lived Leases • Cross-Spectral Association
              </p>
            </div>
          </div>

          {/* Right Action Header */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleGlobalLock}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide border transition-all ${
                globalLock
                  ? "bg-rose-500/20 text-rose-400 border-rose-500/50 shadow-lg shadow-rose-950/50"
                  : "bg-neutral-800 hover:bg-neutral-700 text-neutral-300 border-neutral-700"
              }`}
            >
              {globalLock ? <Lock className="w-3.5 h-3.5 text-rose-400" /> : <Unlock className="w-3.5 h-3.5 text-emerald-400" />}
              {globalLock ? "GLOBAL PTZ LOCKED" : "GLOBAL PTZ ARMED"}
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* Network Partition & Resilience Strip */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className={`p-2.5 rounded-xl border ${
                partitionMode === "NORMAL"
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                  : "bg-rose-500/10 border-rose-500/30 text-rose-400"
              }`}>
                {partitionMode === "NORMAL" ? <Wifi className="w-5 h-5" /> : <WifiOff className="w-5 h-5" />}
              </div>
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Control Plane Authority & Resilience Lease
                </h4>
                <p className="text-xs text-neutral-400">
                  Status: <span className="text-neutral-200 font-mono">{syncStatus}</span> • Leases Active: <span className="text-emerald-400 font-mono">8 / 8</span>
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {partitionMode === "NORMAL" ? (
                <button
                  onClick={simulatePartition}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30 transition-all"
                >
                  Simulate Partition
                </button>
              ) : (
                <button
                  onClick={recoverNetwork}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 transition-all animate-pulse"
                >
                  Reconnect & Reconcile
                </button>
              )}
            </div>
          </div>

          {/* 4-Camera Distributed Peer Corridor */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/30 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Share2 className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">
                  Live Peer Handover Corridor (CAM-001 → CAM-004)
                </h3>
              </div>
              <span className="text-xs text-neutral-400 font-mono">
                Entity: GLOBAL-PERSON-00001
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-2">
              {/* CAM-001 */}
              <div className="rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-emerald-400">CAM-001</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-mono">ORIGIN</span>
                </div>
                <div className="my-2">
                  <div className="text-xs text-white font-medium">Local Track #24</div>
                  <div className="text-[11px] text-neutral-400">YOLO11n • 0.91 Conf</div>
                </div>
                <div className="flex items-center gap-1 text-[10px] text-emerald-400">
                  <CheckCircle2 className="w-3 h-3" />
                  <span>Handed off to Peer</span>
                </div>
              </div>

              {/* CAM-002 */}
              <div className="rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-emerald-400">CAM-002</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-mono">PEER 1</span>
                </div>
                <div className="my-2">
                  <div className="text-xs text-white font-medium">Acquired #89</div>
                  <div className="text-[11px] text-neutral-400">Direct Peer Slew • 11.2s Lead</div>
                </div>
                <div className="flex items-center gap-1 text-[10px] text-emerald-400">
                  <CheckCircle2 className="w-3 h-3" />
                  <span>Confirmed ✓</span>
                </div>
              </div>

              {/* CAM-003 */}
              <div className="rounded-lg border border-cyan-500/50 bg-cyan-950/30 p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-cyan-400">CAM-003</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-cyan-500/20 text-cyan-300 font-mono">ACTIVE</span>
                </div>
                <div className="my-2">
                  <div className="text-xs text-white font-medium">Target in Dwell</div>
                  <div className="text-[11px] text-neutral-400">PTZ Locked • South Sentry</div>
                </div>
                <div className="flex items-center gap-1 text-[10px] text-cyan-300 animate-pulse">
                  <Clock className="w-3 h-3" />
                  <span>Tracking (Pre-Cueing CAM-004)</span>
                </div>
              </div>

              {/* CAM-004 */}
              <div className="rounded-lg border border-neutral-700 bg-neutral-900/50 p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-neutral-300">CAM-004</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-neutral-700 text-neutral-300 font-mono">PRE-CUED</span>
                </div>
                <div className="my-2">
                  <div className="text-xs text-white font-medium">Poised Vector</div>
                  <div className="text-[11px] text-neutral-400">ETA: ~18.0s • Airspace Tower</div>
                </div>
                <div className="flex items-center gap-1 text-[10px] text-amber-400">
                  <ArrowRight className="w-3 h-3" />
                  <span>Peer Accepted</span>
                </div>
              </div>
            </div>
          </div>

          {/* Cross-Spectral Association Showcase */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/30 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Flame className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-semibold text-white">
                  Thermal-Optical Cross-Spectral Association
                </h3>
              </div>
              <span className="px-2 py-0.5 text-[10px] font-semibold uppercase rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/40">
                Mode: SIMULATED (Rule 31/63 Compliant)
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
              <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                    <Eye className="w-4 h-4" />
                    <span>OPTICAL SENTRY (CAM-001)</span>
                  </div>
                  <span className="text-neutral-400 font-mono text-[11px]">RGB Visible Spectrum</span>
                </div>
                <p className="text-xs text-neutral-300">
                  Full color bounding box aspect ratio: 2.31 • Heading: 180° South • Velocity: 1.3 m/s
                </p>
                <div className="text-[11px] text-neutral-400 font-mono">
                  Detection Confidence: <span className="text-white">0.91</span>
                </div>
              </div>

              <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
                    <Flame className="w-4 h-4" />
                    <span>THERMAL OUTPOST (CAM-005)</span>
                  </div>
                  <span className="text-neutral-400 font-mono text-[11px]">LWIR Heat Signature</span>
                </div>
                <p className="text-xs text-neutral-300">
                  Heat silhouette aspect ratio: 2.28 • Heading: 184° South • Velocity: 1.2 m/s
                </p>
                <div className="text-[11px] text-neutral-400 font-mono">
                  Association: <span className="text-emerald-400 font-bold">PROBABLE (0.74)</span> • Geometry Match: 98%
                </div>
              </div>
            </div>
          </div>

          {/* 8-Camera Node Mesh Grid */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">
                  Distributed 8-Camera Sentry Mesh Nodes
                </h3>
              </div>
              <span className="text-xs text-neutral-400">
                8 Active Edge Nodes Connected
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {nodes.map((node) => (
                <div
                  key={node.node_id}
                  className={`rounded-xl border p-3 flex flex-col justify-between transition-all ${
                    node.health === "ONLINE"
                      ? "border-neutral-800 bg-neutral-900/40 hover:border-neutral-700"
                      : "border-rose-500/40 bg-rose-950/20"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-xs text-white">{node.camera_id}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                      node.sensor_type === "THERMAL"
                        ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                        : node.sensor_type === "DUAL"
                        ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                        : "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                    }`}>
                      {node.sensor_type}
                    </span>
                  </div>

                  <div className="my-2">
                    <div className="text-xs text-neutral-300 font-medium truncate">{node.name}</div>
                    <div className="text-[11px] text-neutral-400 flex items-center justify-between mt-1">
                      <span>Lease: {node.lease_expires_in_sec}s</span>
                      <span>Outbox: {node.outbox_pending}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[10px] pt-1 border-t border-neutral-800/80">
                    <span className="text-emerald-400 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                      {node.health}
                    </span>
                    <span className="text-neutral-400 font-mono">
                      {node.ptz ? "PTZ DOME" : "FIXED"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-neutral-800 bg-neutral-900/50 text-xs text-neutral-400">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-neutral-400" />
              Outbox Backlog: <strong className="text-white">0 events</strong>
            </span>
            <span>
              Crypto Audit: <strong className="text-emerald-400">SHA-256 PASS</strong>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-neutral-800 hover:bg-neutral-700 text-white transition-colors"
            >
              Close
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};
