import React, { useState } from "react";
import {
  X,
  BrainCircuit,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Lock,
  Search,
  Eye,
  Camera,
  FileCheck,
  ShieldAlert,
  UserCheck,
  Sparkles,
} from "lucide-react";

interface AiQualityDashboardModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AiQualityDashboardModal: React.FC<AiQualityDashboardModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<
    | "scorecard"
    | "feedback"
    | "mining"
    | "queue"
    | "clustering"
    | "matrix"
    | "drift"
    | "retraining"
    | "canary"
    | "defense"
  >("scorecard");

  const [filterDisposition, setFilterDisposition] = useState<string>("ALL");
  const [reviewActionMsg, setReviewActionMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleReviewFeedback = (id: string, action: string) => {
    setReviewActionMsg(`Review action recorded for ${id}: ${action}. Multi-reviewer consensus updated.`);
    setTimeout(() => setReviewActionMsg(null), 3500);
  };

  const tabs = [
    { id: "scorecard", label: "Executive Scorecard", icon: FileCheck },
    { id: "feedback", label: "Operator Feedback", icon: UserCheck },
    { id: "mining", label: "Hard-Case Mining", icon: Search },
    { id: "queue", label: "Active Learning", icon: BrainCircuit },
    { id: "clustering", label: "Failure Clusters", icon: Layers },
    { id: "matrix", label: "Model x Camera Matrix", icon: Camera },
    { id: "drift", label: "Drift Intelligence", icon: Activity },
    { id: "retraining", label: "Retraining Advisory", icon: Sparkles },
    { id: "canary", label: "Champion / Challenger", icon: Eye },
    { id: "defense", label: "Poisoning Defense & Lineage", icon: ShieldAlert },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-3 sm:p-5 animate-in fade-in duration-200">
      <div className="relative w-full max-w-7xl max-h-[94vh] flex flex-col rounded-2xl bg-neutral-950 border border-neutral-800 shadow-2xl overflow-hidden font-sans">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-900/80">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <BrainCircuit className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <h2 className="text-lg font-bold tracking-wide text-white">
                  Continuous Edge Learning & AI Quality Intelligence
                </h2>
                <span className="px-2.5 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                  Phase XIII
                </span>
                <span className="px-2.5 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                  Production Baselines Frozen
                </span>
                <span className="px-2.5 py-0.5 text-[10px] font-bold tracking-wider uppercase rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40">
                  Dual-Signature Required
                </span>
              </div>
              <p className="text-xs text-neutral-400 mt-0.5">
                Operator Feedback Loops • Hard-Case Mining • Failure Clustering • Shadow Canary • Anti-Poisoning Defense
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
              title="Close Modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation Strip */}
        <div className="flex items-center gap-1 px-4 border-b border-neutral-800 bg-neutral-900/40 overflow-x-auto text-xs font-semibold scrollbar-thin">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 py-3 px-3.5 border-b-2 transition-all whitespace-nowrap ${
                  isSelected
                    ? "border-cyan-500 text-cyan-400 font-bold bg-cyan-950/20"
                    : "border-transparent text-neutral-400 hover:text-neutral-200 hover:bg-neutral-900/60"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Action Notification Toast */}
        {reviewActionMsg && (
          <div className="px-6 py-2 bg-emerald-950/60 border-b border-emerald-500/40 text-emerald-300 text-xs flex items-center gap-2 animate-in fade-in">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>{reviewActionMsg}</span>
          </div>
        )}

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* TAB 1: EXECUTIVE SCORECARD */}
          {activeTab === "scorecard" && (
            <div className="space-y-6">
              {/* Claims Policy Alert Banner */}
              <div className="p-4 rounded-xl border border-amber-500/40 bg-amber-950/20 text-amber-300 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                <div className="text-xs space-y-1">
                  <div className="font-bold uppercase tracking-wider">Mandatory Claims Policy Interlock</div>
                  <p className="text-neutral-300 leading-relaxed">
                    Perception model accuracy claims must distinguish certified offline validation from live operational proxies.
                    Operator feedback alone does <span className="font-semibold text-amber-200">NOT</span> constitute certified ground truth.
                    Combined synthetic accuracy percentages are strictly prohibited under TRINETRA AI governance rules.
                  </p>
                </div>
              </div>

              {/* Domain Scorecards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {[
                  {
                    domain: "GROUND PERCEPTION",
                    model: "ibvap_detector (Ground v2.0.0)",
                    sha: "7DBF36027768...B0",
                    valRecall: "0.865",
                    valMap: "0.840",
                    smallRecall: "0.680",
                    alerts: 1420,
                    fpFeedbackRate: "0.035",
                    fnCount: 2,
                    ptzSuccess: "98.5%",
                    status: "HEALTHY",
                  },
                  {
                    domain: "AIRBORNE PERCEPTION",
                    model: "ibvap_airborne_v2 (Airborne v2.0.0)",
                    sha: "5229632C3D7A...84",
                    valRecall: "0.880",
                    valMap: "0.855",
                    smallRecall: "0.720",
                    alerts: 280,
                    fpFeedbackRate: "0.021",
                    fnCount: 0,
                    ptzSuccess: "99.1%",
                    status: "HEALTHY",
                  },
                  {
                    domain: "SECURITY ITEM PERCEPTION",
                    model: "ibvap_security_item_v2_1 (Security v2.1.0)",
                    sha: "72464C778DE5...DA",
                    valRecall: "0.845",
                    valMap: "0.825",
                    smallRecall: "0.650",
                    alerts: 490,
                    fpFeedbackRate: "0.038",
                    fnCount: 1,
                    ptzSuccess: "98.8%",
                    status: "HEALTHY",
                  },
                ].map((sc) => (
                  <div key={sc.domain} className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-neutral-200 tracking-wide uppercase">{sc.domain}</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                        {sc.status}
                      </span>
                    </div>

                    <div className="space-y-1">
                      <div className="text-[11px] text-neutral-400 truncate">{sc.model}</div>
                      <div className="text-[10px] font-mono text-neutral-500">SHA: {sc.sha}</div>
                    </div>

                    {/* Offline Validation Box */}
                    <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3 space-y-2">
                      <div className="flex items-center justify-between text-[10px] uppercase font-bold text-cyan-400">
                        <span>Offline Validation</span>
                        <span className="px-1.5 py-0.2 rounded bg-cyan-500/20 text-[9px]">IBVAP-GT-v1.0 (Frozen)</span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center pt-1">
                        <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
                          <div className="text-[9px] text-neutral-500">Recall</div>
                          <div className="text-xs font-bold text-neutral-200">{sc.valRecall}</div>
                        </div>
                        <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
                          <div className="text-[9px] text-neutral-500">mAP@50</div>
                          <div className="text-xs font-bold text-neutral-200">{sc.valMap}</div>
                        </div>
                        <div className="p-1.5 rounded bg-neutral-900 border border-neutral-800">
                          <div className="text-[9px] text-neutral-500">Small Obj</div>
                          <div className="text-xs font-bold text-neutral-200">{sc.smallRecall}</div>
                        </div>
                      </div>
                    </div>

                    {/* Operational Proxies Box */}
                    <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3 space-y-2">
                      <div className="flex items-center justify-between text-[10px] uppercase font-bold text-purple-400">
                        <span>Operational Proxies</span>
                        <span className="px-1.5 py-0.2 rounded bg-purple-500/20 text-[9px]">Live Mesh (7 Days)</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                        <div className="flex justify-between text-neutral-400">
                          <span>Alerts Analyzed:</span>
                          <span className="font-semibold text-neutral-200">{sc.alerts}</span>
                        </div>
                        <div className="flex justify-between text-neutral-400">
                          <span>FP Feedback:</span>
                          <span className="font-semibold text-amber-400">{sc.fpFeedbackRate}</span>
                        </div>
                        <div className="flex justify-between text-neutral-400">
                          <span>FN Reports:</span>
                          <span className="font-semibold text-neutral-200">{sc.fnCount}</span>
                        </div>
                        <div className="flex justify-between text-neutral-400">
                          <span>PTZ Cue Rate:</span>
                          <span className="font-semibold text-emerald-400">{sc.ptzSuccess}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 2: OPERATOR FEEDBACK */}
          {activeTab === "feedback" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-neutral-300">Filter Disposition:</span>
                  <select
                    className="text-xs bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
                    value={filterDisposition}
                    onChange={(e) => setFilterDisposition(e.target.value)}
                  >
                    <option value="ALL">ALL (11 Dispositions)</option>
                    <option value="FALSE_POSITIVE">FALSE_POSITIVE</option>
                    <option value="FALSE_NEGATIVE">FALSE_NEGATIVE</option>
                    <option value="MISCLASSIFICATION">MISCLASSIFICATION</option>
                    <option value="TRUE_POSITIVE">TRUE_POSITIVE</option>
                  </select>
                </div>

                <div className="text-xs text-neutral-400 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span>Multi-Reviewer Consensus Interlock Active</span>
                </div>
              </div>

              {/* Feedback List Table */}
              <div className="rounded-xl border border-neutral-800 overflow-hidden bg-neutral-900/30">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-neutral-800 bg-neutral-900/80 text-neutral-400 uppercase text-[10px] tracking-wider">
                      <th className="p-3">Feedback ID</th>
                      <th className="p-3">Camera</th>
                      <th className="p-3">Disposition</th>
                      <th className="p-3">Reason / Details</th>
                      <th className="p-3">Status</th>
                      <th className="p-3">Consensus Review</th>
                      <th className="p-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-800 text-neutral-300 font-mono text-[11px]">
                    {[
                      {
                        id: "FDB-CAM-002-1001",
                        cam: "CAM-002",
                        disp: "FALSE_POSITIVE",
                        reason: "Thermal reflection on chain-link perimeter fence flagged as person",
                        status: "REVIEW_REQUIRED",
                        reviews: "0/2 Reviews",
                      },
                      {
                        id: "FDB-CAM-004-1002",
                        cam: "CAM-004",
                        disp: "MISCLASSIFICATION",
                        reason: "Cordless drill held by perimeter contractor classified as firearm",
                        status: "DISPUTED",
                        reviews: "1 Agree / 1 Disagree",
                      },
                      {
                        id: "FDB-CAM-001-1003",
                        cam: "CAM-001",
                        disp: "FALSE_NEGATIVE",
                        reason: "Small pedestrian partially occluded in culvert shadow missed for 3 frames",
                        status: "VALIDATED",
                        reviews: "2/2 Agree (Admin Approved)",
                      },
                      {
                        id: "FDB-CAM-005-1004",
                        cam: "CAM-005",
                        disp: "TRUE_POSITIVE",
                        reason: "Operator confirmed tactical response team perimeter breach drill",
                        status: "VALIDATED",
                        reviews: "1/1 Agree",
                      },
                    ].map((row) => (
                      <tr key={row.id} className="hover:bg-neutral-800/40 transition-colors font-sans">
                        <td className="p-3 font-mono font-bold text-cyan-400">{row.id}</td>
                        <td className="p-3 font-semibold">{row.cam}</td>
                        <td className="p-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            row.disp === "FALSE_POSITIVE"
                              ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                              : row.disp === "MISCLASSIFICATION"
                              ? "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                              : row.disp === "FALSE_NEGATIVE"
                              ? "bg-rose-500/20 text-rose-400 border border-rose-500/40"
                              : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                          }`}>
                            {row.disp}
                          </span>
                        </td>
                        <td className="p-3 text-neutral-300 max-w-xs truncate">{row.reason}</td>
                        <td className="p-3">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                            row.status === "VALIDATED"
                              ? "text-emerald-400 bg-emerald-950/40"
                              : row.status === "DISPUTED"
                              ? "text-rose-400 bg-rose-950/40"
                              : "text-amber-400 bg-amber-950/40"
                          }`}>
                            {row.status}
                          </span>
                        </td>
                        <td className="p-3 text-neutral-400 text-[11px]">{row.reviews}</td>
                        <td className="p-3 text-right">
                          <div className="flex items-center justify-end gap-1.5">
                            <button
                              onClick={() => handleReviewFeedback(row.id, "VALIDATE")}
                              className="px-2 py-1 rounded bg-emerald-900/40 hover:bg-emerald-800/50 text-emerald-300 text-[10px] border border-emerald-500/30"
                              title="Agree & Validate"
                            >
                              Validate
                            </button>
                            <button
                              onClick={() => handleReviewFeedback(row.id, "DISPUTE")}
                              className="px-2 py-1 rounded bg-rose-900/40 hover:bg-rose-800/50 text-rose-300 text-[10px] border border-rose-500/30"
                              title="Dispute Disposition"
                            >
                              Dispute
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 3: HARD-CASE MINING */}
          {activeTab === "mining" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs text-neutral-400">
                <span className="font-semibold text-neutral-200">18 Automated Heuristic Triggers Active</span>
                <span>Threshold: Informative Score &gt; 0.65</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {[
                  {
                    id: "HC-CAM-002-8821",
                    trigger: "TRIGGER_HIGH_CONF_FP",
                    score: 0.95,
                    domain: "GROUND",
                    cam: "CAM-002",
                    details: "Fence shadow with confidence 0.88 confirmed FP by operator",
                    status: "MINED_FOR_ALQ",
                  },
                  {
                    id: "HC-CAM-004-8822",
                    trigger: "TRIGGER_TOOL_DISTRACTOR",
                    score: 0.90,
                    domain: "SECURITY_ITEM",
                    cam: "CAM-004",
                    details: "DeWalt Impact Driver with drill bit confused for pistol frame",
                    status: "MINED_FOR_ALQ",
                  },
                  {
                    id: "HC-CAM-001-8823",
                    trigger: "TRIGGER_DISTANT_PERSON",
                    score: 0.85,
                    domain: "GROUND",
                    cam: "CAM-001",
                    details: "Pedestrian bounding box 14x28px (< 0.05 area) near border post 12",
                    status: "MINED_FOR_ALQ",
                  },
                  {
                    id: "HC-CAM-006-8824",
                    trigger: "TRIGGER_AIRBORNE_CLUTTER",
                    score: 0.82,
                    domain: "AIRBORNE",
                    cam: "CAM-006",
                    details: "Flock of desert birds against sunset glare classified as multi-rotor drone",
                    status: "MINED_FOR_ALQ",
                  },
                  {
                    id: "HC-CAM-003-8825",
                    trigger: "TRIGGER_NIGHT_SCENE",
                    score: 0.78,
                    domain: "GROUND",
                    cam: "CAM-003",
                    details: "Luminance < 15 lux with infrared headlight bloom",
                    status: "MINED_FOR_ALQ",
                  },
                  {
                    id: "HC-CAM-008-8826",
                    trigger: "TRIGGER_PREDICTION_MISS",
                    score: 0.70,
                    domain: "GROUND",
                    cam: "CAM-008",
                    details: "Motion trajectory prediction diverged > 30m due to sudden ravine descent",
                    status: "MINED_FOR_ALQ",
                  },
                ].map((hc) => (
                  <div key={hc.id} className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-cyan-400">{hc.id}</span>
                      <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                        Score: {hc.score.toFixed(2)}
                      </span>
                    </div>

                    <div className="text-xs font-semibold text-neutral-200">{hc.trigger}</div>
                    <p className="text-[11px] text-neutral-400">{hc.details}</p>

                    <div className="flex items-center justify-between pt-2 border-t border-neutral-800/80 text-[10px] text-neutral-500">
                      <span>Node: {hc.cam}</span>
                      <span className="text-emerald-400 font-semibold">{hc.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 4: ACTIVE LEARNING */}
          {activeTab === "queue" && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/40 flex items-center justify-between flex-wrap gap-3">
                <div>
                  <div className="text-xs font-bold text-neutral-200">Active Learning Prioritization Queue</div>
                  <div className="text-[11px] text-neutral-400">
                    Formula: Score = 0.25·Uncertainty + 0.20·Disagreement + 0.15·ModelDiff + 0.15·Rarity + 0.15·Criticality + 0.10·CamWeight
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800 text-xs text-neutral-300">
                    Queue Size: <strong>142</strong>
                  </span>
                  <button
                    onClick={() => handleReviewFeedback("ALQ-NEXT", "DEQUEUED_FOR_ANNOTATION")}
                    className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold transition-colors"
                  >
                    Dequeue Top Candidate
                  </button>
                </div>
              </div>

              <div className="rounded-xl border border-neutral-800 overflow-hidden bg-neutral-900/30">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-neutral-800 bg-neutral-900/80 text-neutral-400 uppercase text-[10px] tracking-wider">
                      <th className="p-3">Priority Score</th>
                      <th className="p-3">Sample ID</th>
                      <th className="p-3">Domain</th>
                      <th className="p-3">Failure Type</th>
                      <th className="p-3">Uncertainty</th>
                      <th className="p-3">Disagreement</th>
                      <th className="p-3">Criticality</th>
                      <th className="p-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-800 text-neutral-300 text-[11px]">
                    {[
                      {
                        score: "0.8920",
                        id: "SMP-CAM-004-9101",
                        domain: "SECURITY_ITEM",
                        fail: "MISCLASSIFICATION",
                        unc: "0.91",
                        dis: "1.00",
                        crit: "1.00 (Firearm)",
                      },
                      {
                        score: "0.8415",
                        id: "SMP-CAM-001-9102",
                        domain: "GROUND",
                        fail: "FALSE_NEGATIVE",
                        unc: "0.85",
                        dis: "0.75",
                        crit: "0.90 (Person)",
                      },
                      {
                        score: "0.7930",
                        id: "SMP-CAM-006-9103",
                        domain: "AIRBORNE",
                        fail: "FALSE_POSITIVE",
                        unc: "0.82",
                        dis: "0.60",
                        crit: "0.85 (Drone)",
                      },
                      {
                        score: "0.7250",
                        id: "SMP-CAM-002-9104",
                        domain: "GROUND",
                        fail: "FALSE_POSITIVE",
                        unc: "0.74",
                        dis: "0.50",
                        crit: "0.90 (Person)",
                      },
                    ].map((row) => (
                      <tr key={row.id} className="hover:bg-neutral-800/40 transition-colors">
                        <td className="p-3 font-mono font-bold text-cyan-400">{row.score}</td>
                        <td className="p-3 font-mono text-neutral-300">{row.id}</td>
                        <td className="p-3 font-semibold">{row.domain}</td>
                        <td className="p-3 text-neutral-300">{row.fail}</td>
                        <td className="p-3 text-neutral-400">{row.unc}</td>
                        <td className="p-3 text-neutral-400">{row.dis}</td>
                        <td className="p-3 text-amber-300 font-semibold">{row.crit}</td>
                        <td className="p-3 text-right">
                          <button
                            onClick={() => handleReviewFeedback(row.id, "ANNOTATE_LABEL")}
                            className="px-2.5 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-[10px] font-bold border border-neutral-700"
                          >
                            Annotate
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 5: FAILURE CLUSTERS */}
          {activeTab === "clustering" && (
            <div className="space-y-4">
              <div className="text-xs text-neutral-400">
                Operational failure clusters group recurrent perception errors into targeted retraining campaigns.
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  {
                    id: "CLS-SECURITY-TOOL-DISTRACTOR",
                    name: "Handheld Tools vs Firearms Confusion",
                    domain: "SECURITY_ITEM",
                    type: "MISCLASSIFICATION",
                    samples: 42,
                    cameras: ["CAM-004", "CAM-005"],
                    severity: "HIGH",
                    target: "IBVAP-SECURITY-FEEDBACK-v1",
                    notes: "Cordless drills, impact drivers, and angle grinders confused for pistols",
                  },
                  {
                    id: "CLS-GROUND-SMALL-SHADOW-PERSON",
                    name: "Small Distant Pedestrians in Perimeter Shadows",
                    domain: "GROUND",
                    type: "FALSE_NEGATIVE",
                    samples: 68,
                    cameras: ["CAM-001", "CAM-002", "CAM-007"],
                    severity: "CRITICAL",
                    target: "IBVAP-GROUND-FEEDBACK-v1",
                    notes: "Distant intruders occluded by fence posts or vegetation in low-lux conditions",
                  },
                  {
                    id: "CLS-AIRBORNE-BIRD-SWARM",
                    name: "Bird Swarms and Sun Glare Drone Confusions",
                    domain: "AIRBORNE",
                    type: "FALSE_POSITIVE",
                    samples: 19,
                    cameras: ["CAM-006"],
                    severity: "MEDIUM",
                    target: "IBVAP-AIRBORNE-FEEDBACK-v1",
                    notes: "High-contrast silhouetted birds during dawn/dusk transitions",
                  },
                  {
                    id: "CLS-GROUND-THERMAL-REFLECTION",
                    name: "Perimeter Chain-Link Thermal Flare",
                    domain: "GROUND",
                    type: "FALSE_POSITIVE",
                    samples: 35,
                    cameras: ["CAM-002", "CAM-008"],
                    severity: "MEDIUM",
                    target: "IBVAP-GROUND-FEEDBACK-v1",
                    notes: "Thermal blooming on metallic boundary fence under afternoon desert heat",
                  },
                ].map((cl) => (
                  <div key={cl.id} className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-cyan-400">{cl.id}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        cl.severity === "CRITICAL"
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                          : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                      }`}>
                        {cl.severity}
                      </span>
                    </div>

                    <div>
                      <div className="text-xs font-bold text-neutral-200">{cl.name}</div>
                      <div className="text-[11px] text-neutral-400 mt-0.5">{cl.notes}</div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[11px] text-neutral-400 pt-2 border-t border-neutral-800">
                      <div>Samples: <strong className="text-neutral-200">{cl.samples}</strong></div>
                      <div>Target Dataset: <strong className="text-cyan-300 font-mono text-[10px]">{cl.target}</strong></div>
                      <div className="col-span-2">Affected Nodes: {cl.cameras.join(", ")}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 6: MODEL X CAMERA MATRIX */}
          {activeTab === "matrix" && (
            <div className="space-y-4">
              <div className="text-xs text-neutral-400 flex items-center justify-between">
                <span>Operational telemetry indicators across all 8 edge camera nodes.</span>
                <span className="font-mono text-[10px] text-neutral-500">[OPERATIONAL PROXY - NOT GROUND TRUTH]</span>
              </div>

              <div className="rounded-xl border border-neutral-800 overflow-hidden bg-neutral-900/30">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-neutral-800 bg-neutral-900/80 text-neutral-400 uppercase text-[10px] tracking-wider">
                      <th className="p-3">Camera Node</th>
                      <th className="p-3">Overall Health</th>
                      <th className="p-3">Alerts / Hr</th>
                      <th className="p-3">FP Feedback Rate</th>
                      <th className="p-3">Mean Confidence</th>
                      <th className="p-3">PTZ Cue Hit Rate</th>
                      <th className="p-3">Drift Indicator</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-800 text-neutral-300 text-[11px]">
                    {[
                      { cam: "CAM-001", health: "HEALTHY", rate: "12.4", fp: "0.024", conf: "0.82", ptz: "99.2%", drift: "STABLE" },
                      { cam: "CAM-002", health: "WARNING", rate: "24.1", fp: "0.058", conf: "0.74", ptz: "97.1%", drift: "DRIFT_SUSPECTED" },
                      { cam: "CAM-003", health: "HEALTHY", rate: "11.8", fp: "0.019", conf: "0.84", ptz: "99.5%", drift: "STABLE" },
                      { cam: "CAM-004", health: "HEALTHY", rate: "15.0", fp: "0.032", conf: "0.81", ptz: "98.9%", drift: "STABLE" },
                      { cam: "CAM-005", health: "HEALTHY", rate: "13.2", fp: "0.022", conf: "0.83", ptz: "99.0%", drift: "STABLE" },
                      { cam: "CAM-006", health: "HEALTHY", rate: "9.5", fp: "0.021", conf: "0.85", ptz: "99.4%", drift: "STABLE" },
                      { cam: "CAM-007", health: "HEALTHY", rate: "14.2", fp: "0.028", conf: "0.80", ptz: "98.7%", drift: "STABLE" },
                      { cam: "CAM-008", health: "HEALTHY", rate: "16.1", fp: "0.035", conf: "0.79", ptz: "98.2%", drift: "STABLE" },
                    ].map((row) => (
                      <tr key={row.cam} className="hover:bg-neutral-800/40 transition-colors">
                        <td className="p-3 font-mono font-bold text-cyan-400">{row.cam}</td>
                        <td className="p-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            row.health === "HEALTHY"
                              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                              : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                          }`}>
                            {row.health}
                          </span>
                        </td>
                        <td className="p-3 text-neutral-300">{row.rate}</td>
                        <td className="p-3 font-semibold text-neutral-200">{row.fp}</td>
                        <td className="p-3 text-neutral-400">{row.conf}</td>
                        <td className="p-3 text-emerald-400">{row.ptz}</td>
                        <td className="p-3">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                            row.drift === "STABLE" ? "text-emerald-400" : "text-amber-400 font-bold"
                          }`}>
                            {row.drift}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 7: DRIFT INTELLIGENCE */}
          {activeTab === "drift" && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-bold text-neutral-200">Windowed Distribution Drift Analysis</div>
                  <span className="text-[10px] font-mono text-neutral-500">Ref Window (30d) vs Current Window (7d)</span>
                </div>
                <p className="text-[11px] text-neutral-400 leading-relaxed">
                  Statistical Kolmogorov-Smirnov and Population Stability Index (PSI) tracking confidence shifts,
                  class balance drift, and operator rejection spikes without triggering autonomous production changes.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  {
                    domain: "GROUND PERCEPTION",
                    status: "DRIFT_SUSPECTED",
                    deltaConf: "-0.042",
                    deltaFP: "+0.023",
                    cams: "CAM-002",
                    action: "INCREASE_MONITORING_FREQUENCY",
                  },
                  {
                    domain: "AIRBORNE PERCEPTION",
                    status: "STABLE",
                    deltaConf: "+0.008",
                    deltaFP: "-0.004",
                    cams: "All Nodes Stable",
                    action: "MAINTAIN_STANDARD_MONITORING",
                  },
                  {
                    domain: "SECURITY ITEM PERCEPTION",
                    status: "STABLE",
                    deltaConf: "-0.012",
                    deltaFP: "+0.008",
                    cams: "All Nodes Stable",
                    action: "MAINTAIN_STANDARD_MONITORING",
                  },
                ].map((dr) => (
                  <div key={dr.domain} className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-neutral-200">{dr.domain}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        dr.status === "STABLE"
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                          : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                      }`}>
                        {dr.status}
                      </span>
                    </div>

                    <div className="space-y-1.5 text-xs text-neutral-300">
                      <div className="flex justify-between">
                        <span>Confidence Delta:</span>
                        <span className="font-mono text-neutral-200">{dr.deltaConf}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>FP Feedback Delta:</span>
                        <span className="font-mono text-neutral-200">{dr.deltaFP}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Affected Nodes:</span>
                        <span className="font-mono text-cyan-400 text-[11px]">{dr.cams}</span>
                      </div>
                    </div>

                    <div className="pt-2 border-t border-neutral-800 text-[10px] text-neutral-400">
                      Action: <strong className="text-neutral-200">{dr.action}</strong>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 8: RETRAINING ADVISORY */}
          {activeTab === "retraining" && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl border border-purple-500/40 bg-purple-950/20 text-purple-300 flex items-start gap-3">
                <Sparkles className="w-5 h-5 text-purple-400 shrink-0 mt-0.5" />
                <div className="text-xs space-y-1">
                  <div className="font-bold uppercase tracking-wider">Governed Retraining Advisory Engine</div>
                  <p className="text-neutral-300 leading-relaxed">
                    AI recommendations are purely advisory. Autonomous training or automatic promotion is strictly forbidden.
                    Retraining campaigns require ML Engineer architecture design and dual-signature administrative signoff.
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                {[
                  {
                    recId: "REC-GROUND-2026-001",
                    domain: "GROUND",
                    activeModel: "ibvap_detector (v2.0.0)",
                    level: "REVIEW_RECOMMENDED",
                    reason: "Perimeter shadow pedestrian challenge and CAM-002 thermal blooming cluster",
                    clusters: ["CLS-GROUND-SMALL-SHADOW-PERSON", "CLS-GROUND-THERMAL-REFLECTION"],
                    composition: "75% Baseline • 15% Mined Hard Cases • 5% Validated Feedback • 5% Distractors",
                    status: "PENDING_ML_ENGINEER_REVIEW",
                  },
                  {
                    recId: "REC-SECURITY-2026-002",
                    domain: "SECURITY_ITEM",
                    activeModel: "ibvap_security_item_v2_1 (v2.1.0)",
                    level: "REVIEW_RECOMMENDED",
                    reason: "Power tool vs firearm distractor confusion on perimeter contractor cameras",
                    clusters: ["CLS-SECURITY-TOOL-DISTRACTOR"],
                    composition: "70% Baseline • 20% Hard Negative Tool Distractors • 10% Validated Feedback",
                    status: "PENDING_ML_ENGINEER_REVIEW",
                  },
                ].map((rec) => (
                  <div key={rec.recId} className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-cyan-400">{rec.recId}</span>
                        <span className="text-xs font-bold text-neutral-200">({rec.domain})</span>
                      </div>
                      <span className="px-2.5 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40">
                        {rec.level}
                      </span>
                    </div>

                    <div className="text-xs text-neutral-300 leading-relaxed">
                      <strong>Hypothesis: </strong>{rec.reason}
                    </div>

                    <div className="rounded-lg bg-neutral-950/60 p-3 border border-neutral-800/80 text-xs space-y-1">
                      <div className="text-[10px] uppercase font-bold text-neutral-400">Target Failure Clusters:</div>
                      <div className="text-cyan-300 font-mono text-[11px]">{rec.clusters.join(", ")}</div>
                      <div className="text-[10px] uppercase font-bold text-neutral-400 pt-1">Suggested Balanced Composition:</div>
                      <div className="text-neutral-300 text-[11px]">{rec.composition}</div>
                    </div>

                    <div className="flex items-center justify-between pt-2 text-xs">
                      <span className="text-neutral-500">Active Model: {rec.activeModel}</span>
                      <button
                        onClick={() => handleReviewFeedback(rec.recId, "PROPOSE_CURATION_CAMPAIGN")}
                        className="px-3 py-1 rounded bg-purple-900/40 hover:bg-purple-800/50 text-purple-300 text-xs font-bold border border-purple-500/30"
                      >
                        Create Curation Campaign
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 9: CHAMPION / CHALLENGER */}
          {activeTab === "canary" && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl border border-cyan-500/40 bg-cyan-950/20 text-cyan-300 flex items-start gap-3">
                <Eye className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
                <div className="text-xs space-y-1">
                  <div className="font-bold uppercase tracking-wider">Shadow Inference Safety Guarantee</div>
                  <p className="text-neutral-300 leading-relaxed">
                    The active Production Champion governs 100% of live alerts, operators, and PTZ tracking.
                    Challenger models run strictly in read-only shadow mode for side-by-side telemetry verification.
                  </p>
                </div>
              </div>

              <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-neutral-200">Shadow Canary Evaluation: Ground Perception</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                    24h Shadow Session
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {/* Champion Card */}
                  <div className="rounded-xl border border-emerald-500/40 bg-neutral-950/60 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-emerald-400 uppercase">Active Champion</span>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400">
                        Operational Leader
                      </span>
                    </div>
                    <div className="text-xs text-neutral-200 font-mono">ibvap_detector (v2.0.0)</div>
                    <div className="space-y-1 text-xs text-neutral-400 pt-2 border-t border-neutral-800">
                      <div className="flex justify-between"><span>Alerts Generated:</span> <strong className="text-neutral-200">142</strong></div>
                      <div className="flex justify-between"><span>FP Feedback Rate:</span> <strong className="text-neutral-200">0.035</strong></div>
                      <div className="flex justify-between"><span>p95 Latency:</span> <strong className="text-neutral-200">15.2 ms</strong></div>
                      <div className="flex justify-between"><span>Operational Authority:</span> <strong className="text-emerald-400">100%</strong></div>
                    </div>
                  </div>

                  {/* Challenger Card */}
                  <div className="rounded-xl border border-purple-500/40 bg-neutral-950/60 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-purple-400 uppercase">Challenger Candidate</span>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">
                        Isolated Shadow Mode
                      </span>
                    </div>
                    <div className="text-xs text-neutral-200 font-mono">ibvap_ground_candidate_v2_1</div>
                    <div className="space-y-1 text-xs text-neutral-400 pt-2 border-t border-neutral-800">
                      <div className="flex justify-between"><span>Alerts Shadowed:</span> <strong className="text-neutral-200">138</strong></div>
                      <div className="flex justify-between"><span>FP Feedback Rate:</span> <strong className="text-emerald-400">0.018 (-48%)</strong></div>
                      <div className="flex justify-between"><span>Small Person Recovery:</span> <strong className="text-cyan-400">+14 samples</strong></div>
                      <div className="flex justify-between"><span>p95 Latency:</span> <strong className="text-neutral-200">14.8 ms</strong></div>
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-neutral-900 border border-neutral-800 text-xs text-neutral-300 flex items-center justify-between">
                  <span>Verdict: <strong className="text-emerald-400">CHALLENGER_SUPERIOR_APPROVED_FOR_PROMOTION_REVIEW</strong></span>
                  <span className="text-[11px] text-neutral-400">Requires Stage 13 Governance Promotion Signoff</span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 10: DEFENSE & LINEAGE */}
          {activeTab === "defense" && (
            <div className="space-y-4">
              {/* Benchmark Quarantine Audit Box */}
              <div className="p-5 rounded-xl border border-rose-500/40 bg-rose-950/20 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-rose-300 text-xs font-bold uppercase">
                    <Lock className="w-4 h-4 text-rose-400" />
                    <span>Frozen Benchmark Quarantine Status</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40">
                    QUARANTINE ENFORCED
                  </span>
                </div>
                <p className="text-xs text-neutral-300 leading-relaxed">
                  Dataset <strong className="text-white">IBVAP-GT-v1.0</strong> (100 benchmark frames) is permanently quarantined.
                  Any incoming operator feedback or mined hard case with a matching SHA-256 hash is rejected immediately with
                  <code className="text-rose-300 mx-1">BENCHMARK_LEAKAGE_REJECTED</code> to prevent contamination.
                </p>
                <div className="text-[11px] text-neutral-400 font-mono">
                  Quarantine Hashes Ingested: 100 • Leakage Rejections: 0 • Integrity Verified: YES
                </div>
              </div>

              {/* Data Poisoning Alerts */}
              <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-neutral-200">Data Poisoning & Anomaly Defense Log</span>
                  <span className="text-[10px] font-mono text-neutral-500">Active Defense Filters</span>
                </div>

                <div className="space-y-2">
                  {[
                    {
                      id: "PSN-FILTER-001",
                      type: "DUPLICATE_FLOOD_DEFENSE",
                      status: "CLEAN",
                      details: "Inspected 150 feedback submissions. Max duplicate ratio 0.00% (Threshold: 50%).",
                    },
                    {
                      id: "PSN-FILTER-002",
                      type: "CLASS_SKEW_DETECTION",
                      status: "CLEAN",
                      details: "Class balance variance within standard deviation (Person 60%, Vehicle 30%, Weapon 10%).",
                    },
                    {
                      id: "PSN-FILTER-003",
                      type: "CONTRADICTORY_LABEL_FILTER",
                      status: "CLEAN",
                      details: "0 conflicting labels detected for identical frame hashes across independent reviewers.",
                    },
                  ].map((al) => (
                    <div key={al.id} className="rounded-lg bg-neutral-950/60 p-3 border border-neutral-800 flex items-center justify-between text-xs">
                      <div>
                        <div className="font-bold text-neutral-200">{al.type}</div>
                        <div className="text-[11px] text-neutral-400">{al.details}</div>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                        {al.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-neutral-800 bg-neutral-900/60 text-xs text-neutral-400">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>TRINETRA Continuous Edge Learning Platform v2.0.0 • Phase XIII Governed</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 font-medium transition-colors text-xs"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
};
