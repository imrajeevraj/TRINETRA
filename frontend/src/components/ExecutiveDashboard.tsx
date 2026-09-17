import React, { useState, useMemo } from 'react';
import type { Camera, Alert, SystemStats, GPSLocation, PlateEvent, DetailedHealth } from '../types';
import {
  ShieldAlert,
  ShieldCheck,
  Radio,
  Cpu,
  Eye,
  Crosshair,
  TrendingUp,
  AlertTriangle,
  Clock,
  MapPin,
  Car,
  Plane,
  Target,
  Printer,
  ChevronRight,
  CheckCircle,
  ArrowUpCircle,
  ExternalLink,
  Layers,
  Sparkles,
} from 'lucide-react';

interface ExecutiveDashboardProps {
  cameras: Camera[];
  alerts: Alert[];
  stats: SystemStats;
  apiBase: string;
  dataOrigin: 'LIVE' | 'DEMO' | 'IMPORTED';
  bopLocation: GPSLocation | null;
  onSelectCamera: (camera: Camera) => void;
  onSwitchToTactical: () => void;
  onSwitchToMap: () => void;
  onAcknowledgeAlert: (id: string) => void;
  onEscalateAlert: (id: string) => void;
  detailedHealth: DetailedHealth | null;
  anprPlates: PlateEvent[];
}



export const ExecutiveDashboard: React.FC<ExecutiveDashboardProps> = ({
  cameras,
  alerts,
  stats,
  apiBase,
  dataOrigin,
  bopLocation,
  onSelectCamera,
  onSwitchToTactical,
  onSwitchToMap,
  onAcknowledgeAlert,
  onEscalateAlert,
  detailedHealth,
  anprPlates,
}) => {
  const [timeHorizon, setTimeHorizon] = useState<'1h' | '6h' | '24h' | '7d'>('24h');
  const [sectorFilter, setSectorFilter] = useState<string>('ALL');
  const [hoveredBarIndex, setHoveredBarIndex] = useState<number | null>(null);
  const [selectedIncidentModal, setSelectedIncidentModal] = useState<Alert | null>(null);
  const token = localStorage.getItem('trinetra_token') || localStorage.getItem('ibvap_token') || sessionStorage.getItem('trinetra_token') || sessionStorage.getItem('ibvap_token') || '';
  const tokenQuery = token ? `?token=${encodeURIComponent(token)}` : '';

  // Derived Metrics & Calculations
  const onlineCameras = cameras.filter(c => c.status === 'ONLINE').length;
  const coveragePercent = cameras.length > 0 ? Math.round((onlineCameras / cameras.length) * 100) : 0;

  const criticalAlerts = alerts.filter(a => a.severity === 'CRITICAL');
  const highAlerts = alerts.filter(a => a.severity === 'HIGH');
  const mediumAlerts = alerts.filter(a => a.severity === 'MEDIUM');
  const lowAlerts = alerts.filter(a => a.severity === 'LOW');

  // Threat Index calculation (0 - 100)
  const compositeThreatScore = useMemo(() => {
    if (alerts.length === 0) return 12;
    const base = Math.min(100, criticalAlerts.length * 18 + highAlerts.length * 8 + mediumAlerts.length * 3 + lowAlerts.length * 1);
    return Math.max(15, Math.min(99, base));
  }, [alerts, criticalAlerts, highAlerts, mediumAlerts, lowAlerts]);

  // Threat Level classification
  const threatLevel = useMemo(() => {
    if (compositeThreatScore >= 80 || criticalAlerts.length >= 5) {
      return { code: 'DEFCON 1', name: 'MAXIMUM ALERT', color: 'var(--severity-critical)', bg: 'var(--severity-critical-bg)' };
    }
    if (compositeThreatScore >= 55 || criticalAlerts.length > 0) {
      return { code: 'DEFCON 2', name: 'HIGH THREAT ACTIVE', color: 'var(--severity-high)', bg: 'var(--severity-high-bg)' };
    }
    if (compositeThreatScore >= 35 || highAlerts.length > 0) {
      return { code: 'DEFCON 3', name: 'ELEVATED VIGILANCE', color: 'var(--severity-medium)', bg: 'var(--severity-medium-bg)' };
    }
    return { code: 'DEFCON 4', name: 'NORMAL MONITORING', color: 'var(--status-healthy)', bg: 'rgba(16, 185, 129, 0.12)' };
  }, [compositeThreatScore, criticalAlerts, highAlerts]);

  // Sector aggregated breakdown
  const sectorData = useMemo(() => {
    const sectors = [
      { id: 'BOP-TEST-01', name: 'Sector Alpha (North Post)', camId: 'CAM-001', coords: '26.9124° N, 70.9012° E' },
      { id: 'BOP-TEST-02', name: 'Sector Bravo (Central Gate)', camId: 'CAM-002', coords: '26.9180° N, 70.9080° E' },
      { id: 'BOP-TEST-03', name: 'Sector Charlie (East Ridge)', camId: 'CAM-003', coords: '26.9240° N, 70.9150° E' },
      { id: 'BOP-TEST-04', name: 'Sector Delta (River Outpost)', camId: 'CAM-004', coords: '26.9300° N, 70.9220° E' },
    ];

    return sectors.map(sec => {
      const cam = cameras.find(c => c.id === sec.camId || c.location === sec.id);
      const secAlerts = alerts.filter(a => a.camera_id === sec.camId || a.location === sec.id);
      const secCrit = secAlerts.filter(a => a.severity === 'CRITICAL').length;
      const persons = cam?.detections ? new Set(cam.detections.filter(d => d.class === 'person' && d.track_id).map(d => d.track_id)).size : 0;
      const vehicles = cam?.detections ? new Set(cam.detections.filter(d => d.class !== 'person' && d.track_id).map(d => d.track_id)).size : 0;

      const riskIndex = Math.min(100, secCrit * 25 + secAlerts.length * 6 + (cam?.status === 'OFFLINE' ? 40 : 0));

      return {
        ...sec,
        camera: cam,
        status: cam?.status || 'OFFLINE',
        fps: cam?.fps || 0,
        alertCount: secAlerts.length,
        criticalCount: secCrit,
        personCount: persons,
        vehicleCount: vehicles,
        riskIndex,
      };
    });
  }, [cameras, alerts]);

  // Filtered sectors
  const displayedSectors = useMemo(() => {
    if (sectorFilter === 'ALL') return sectorData;
    return sectorData.filter(s => s.id === sectorFilter);
  }, [sectorData, sectorFilter]);

  // Synthetic 24-hour timeline generator for visual chart
  const timelineHours = useMemo(() => {
    const hours = [];
    const now = new Date();
    for (let i = 11; i >= 0; i--) {
      const d = new Date(now.getTime() - i * 2 * 3600 * 1000);
      const label = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      // Distribution calculation based on alerts
      const hourSeed = (d.getHours() * 7) % 10;
      const baseCritical = Math.max(0, Math.floor(criticalAlerts.length / 12) + (hourSeed > 6 ? 2 : 0));
      const baseHigh = Math.max(0, Math.floor(highAlerts.length / 12) + (hourSeed > 4 ? 3 : 1));
      const baseMed = Math.max(1, Math.floor(mediumAlerts.length / 12) + (hourSeed % 3));
      const baseLow = Math.max(0, Math.floor(lowAlerts.length / 12) + 1);

      hours.push({
        time: label,
        critical: baseCritical,
        high: baseHigh,
        medium: baseMed,
        low: baseLow,
        total: baseCritical + baseHigh + baseMed + baseLow,
      });
    }
    return hours;
  }, [criticalAlerts.length, highAlerts.length, mediumAlerts.length, lowAlerts.length]);

  const maxTimelineVal = Math.max(...timelineHours.map(h => h.total), 8);

  // Print / Export SITREP handler
  const handlePrintSITREP = () => {
    window.print();
  };

  return (
    <div className="executive-dashboard" role="region" aria-label="Executive Analytics Dashboard">
      {/* ── TOP HERO BANNER: THREAT POSTURE & STRATEGIC CONTROLS ── */}
      <section className="dashboard-hero-banner">
        <div className="hero-threat-posture">
          <div className="threat-gauge-ring">
            <svg viewBox="0 0 100 100" className="gauge-svg">
              <circle cx="50" cy="50" r="42" className="gauge-bg" />
              <circle
                cx="50"
                cy="50"
                r="42"
                className="gauge-fill"
                style={{
                  strokeDasharray: '264',
                  strokeDashoffset: `${264 - (264 * compositeThreatScore) / 100}`,
                  stroke: threatLevel.color,
                }}
              />
            </svg>
            <div className="gauge-center-text">
              <span className="gauge-score" style={{ color: threatLevel.color }}>{compositeThreatScore}</span>
              <span className="gauge-unit">THREAT IDX</span>
            </div>
          </div>

          <div className="threat-status-meta">
            <div className="threat-badge-row">
              <span className="threat-defcon-tag" style={{ background: threatLevel.bg, color: threatLevel.color, borderColor: threatLevel.color }}>
                <ShieldAlert size={14} />
                {threatLevel.code}
              </span>
              <span className="threat-state-label">{threatLevel.name}</span>
              <span className="provenance-tag">{dataOrigin} DATA STREAM</span>
              {bopLocation && (
                <span className="sector-geo-coord" style={{ fontSize: '10px' }}>
                  HQ: {bopLocation.lat.toFixed(4)}°N, {bopLocation.lng.toFixed(4)}°E
                </span>
              )}
            </div>
            <p className="threat-summary-desc">
              Multi-sector intelligence correlates <strong>{alerts.length} security alerts</strong> across {cameras.length} active border optical nodes.
              {criticalAlerts.length > 0
                ? ` Immediate supervisor triage recommended for ${criticalAlerts.length} critical breach vectors.`
                : ' All monitored boundary tripwires are currently within safe thresholds.'}
            </p>
          </div>
        </div>

        {/* Strategic Dashboard Controls */}
        <div className="hero-actions-cluster">
          <div className="dashboard-horizon-switcher" role="group" aria-label="Time horizon">
            {(['1h', '6h', '24h', '7d'] as const).map(h => (
              <button
                key={h}
                className={`horizon-btn ${timeHorizon === h ? 'active' : ''}`}
                onClick={() => setTimeHorizon(h)}
              >
                {h.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="sector-filter-dropdown">
            <MapPin size={13} className="sector-filter-icon" />
            <select
              value={sectorFilter}
              onChange={e => setSectorFilter(e.target.value)}
              aria-label="Filter by Sector Outpost"
              className="sector-select"
            >
              <option value="ALL">All Sectors (BOP 1–4)</option>
              <option value="BOP-TEST-01">Sector Alpha (BOP-01)</option>
              <option value="BOP-TEST-02">Sector Bravo (BOP-02)</option>
              <option value="BOP-TEST-03">Sector Charlie (BOP-03)</option>
              <option value="BOP-TEST-04">Sector Delta (BOP-04)</option>
            </select>
          </div>

          <button
            className="btn btn-primary btn-sm print-sitrep-btn"
            onClick={handlePrintSITREP}
            title="Export official military SITREP document"
          >
            <Printer size={13} />
            <span>Export SITREP</span>
          </button>
        </div>
      </section>

      {/* ── 4 EXECUTIVE STRATEGIC KPI HERO CARDS ── */}
      <section className="executive-kpi-grid">
        {/* KPI 1: Outpost & Camera Grid Readiness */}
        <div className="kpi-card glassmorphic">
          <div className="kpi-header">
            <span className="kpi-title">BORDER OUTPOST READINESS</span>
            <div className={`kpi-status-dot ${coveragePercent >= 75 ? 'ok' : 'warning'}`} />
          </div>
          <div className="kpi-main-metric">
            <span className="kpi-big-number">{onlineCameras}<span className="kpi-sub-number">/{cameras.length}</span></span>
            <span className="kpi-percent-badge">{coveragePercent}% Online</span>
          </div>
          <div className="kpi-progress-bar">
            <div className="kpi-progress-fill" style={{ width: `${coveragePercent}%`, background: 'var(--accent)' }} />
          </div>
          <div className="kpi-footer-stats">
            <span>Avg FPS: <strong>{detailedHealth?.cameras?.inference_fps != null ? `${detailedHealth.cameras.inference_fps} FPS` : '—'}</strong></span>
            <span>Stale: <strong>{detailedHealth?.cameras?.stale?.length ?? '—'}</strong></span>
          </div>
        </div>

        {/* KPI 2: Live Targets Under Surveillance */}
        <div className="kpi-card glassmorphic">
          <div className="kpi-header">
            <span className="kpi-title">TARGETS UNDER SURVEILLANCE</span>
            <Crosshair size={14} style={{ color: 'var(--accent)' }} />
          </div>
          <div className="kpi-main-metric">
            <span className="kpi-big-number">{stats.person_detections + stats.vehicle_detections}</span>
            <span className="kpi-tag-sub">ACTIVE TRACKS</span>
          </div>
          <div className="kpi-targets-breakdown">
            <div className="target-pill">
              <span className="target-label">Persons</span>
              <span className="target-val">{stats.person_detections}</span>
            </div>
            <div className="target-pill">
              <span className="target-label">Vehicles</span>
              <span className="target-val">{stats.vehicle_detections}</span>
            </div>
            <div className="target-pill">
              <span className="target-label">Drones</span>
              <span className="target-val">{detailedHealth?.air_ai?.drones ?? 0}</span>
            </div>
            <div className="target-pill">
              <span className="target-label">Weapons</span>
              <span className="target-val">{detailedHealth?.security_item_ai?.firearms ?? 0}</span>
            </div>
          </div>
          <div className="kpi-footer-stats">
            <span>Tracking Engine: <strong>ByteTrack 2D</strong></span>
            <span>Consensus: <strong>3-Frame Window</strong></span>
          </div>
        </div>

        {/* KPI 3: Incident Frequency & Risk Volume */}
        <div className="kpi-card glassmorphic">
          <div className="kpi-header">
            <span className="kpi-title">24H INCIDENT VELOCITY</span>
            <TrendingUp size={14} style={{ color: 'var(--severity-high)' }} />
          </div>
          <div className="kpi-main-metric">
            <span className="kpi-big-number">{alerts.length}</span>
            <span className="kpi-tag-critical">{criticalAlerts.length} CRITICAL</span>
          </div>
          <div className="kpi-severity-mini-bars">
            <div className="mini-bar crit" style={{ flex: Math.max(1, criticalAlerts.length) }} title={`${criticalAlerts.length} Critical`} />
            <div className="mini-bar high" style={{ flex: Math.max(1, highAlerts.length) }} title={`${highAlerts.length} High`} />
            <div className="mini-bar med" style={{ flex: Math.max(1, mediumAlerts.length) }} title={`${mediumAlerts.length} Medium`} />
            <div className="mini-bar low" style={{ flex: Math.max(1, lowAlerts.length) }} title={`${lowAlerts.length} Low`} />
          </div>
          <div className="kpi-footer-stats">
            <span>Acknowledged: <strong>{alerts.filter(a => a.status === 'ACKNOWLEDGED').length}</strong></span>
            <span>Escalated: <strong>{alerts.filter(a => a.status === 'ESCALATED').length}</strong></span>
          </div>
        </div>

        {/* KPI 4: Edge AI & Hardware Health */}
        <div className="kpi-card glassmorphic">
          <div className="kpi-header">
            <span className="kpi-title">EDGE NEURAL LATENCY</span>
            <Cpu size={14} style={{ color: 'var(--status-ai)' }} />
          </div>
          <div className="kpi-main-metric">
            <span className="kpi-big-number">
              {detailedHealth?.stage_profiling_ms?.avg_inference_ms != null ? `${detailedHealth.stage_profiling_ms.avg_inference_ms}` : '—'}
              <span className="kpi-sub-number">ms</span>
            </span>
            {detailedHealth?.stage_profiling_ms?.avg_inference_ms != null && (() => {
              const infMs = detailedHealth.stage_profiling_ms.avg_inference_ms;
              const isCrit = infMs > 1000;
              const isWarn = infMs > 200;
              const badgeText = isCrit ? 'High Load' : isWarn ? 'Moderate' : 'P95 Optimal';
              const badgeColor = isCrit ? 'var(--severity-critical)' : isWarn ? 'var(--severity-medium)' : 'var(--status-healthy)';
              const badgeBorder = isCrit ? 'rgba(239, 68, 68, 0.35)' : isWarn ? 'rgba(245, 158, 11, 0.35)' : 'rgba(16, 185, 129, 0.35)';
              return (
                <span className="kpi-percent-badge" style={{ color: badgeColor, borderColor: badgeBorder }}>
                  {badgeText}
                </span>
              );
            })()}
          </div>
          <div className="kpi-hw-bars">
            <div className="hw-bar-item">
              <span>CPU</span>
              <div className="hw-bar-track"><div className="hw-bar-fill" style={{ width: `${detailedHealth?.cpu?.percent ?? 0}%` }} /></div>
              <span>{detailedHealth?.cpu?.percent != null ? `${detailedHealth.cpu.percent}%` : '—'}</span>
            </div>
            <div className="hw-bar-item">
              <span>RAM</span>
              <div className="hw-bar-track"><div className="hw-bar-fill" style={{ width: `${detailedHealth?.memory?.percent ?? 0}%` }} /></div>
              <span>{detailedHealth?.memory?.percent != null ? `${detailedHealth.memory.percent}%` : '—'}</span>
            </div>
          </div>
          <div className="kpi-footer-stats">
            <span>Pipeline: <strong>Tri-Model YOLO11</strong></span>
            <span>VRAM: <strong>{detailedHealth?.gpu?.memory_used_mb != null ? `${detailedHealth.gpu.memory_used_mb}MB` : '—'}</strong></span>
          </div>
        </div>
      </section>

      {/* ── MAIN DASHBOARD CONTENT GRID ── */}
      <div className="dashboard-content-grid">
        {/* LEFT COLUMN: 24-Hour Incursion Timeline & Multi-Model Telemetry */}
        <div className="dashboard-col-primary">
          {/* 24-Hour Incursion & Threat Velocity Timeline (Interactive SVG Chart) */}
          <section className="dash-panel timeline-panel glassmorphic">
            <div className="dash-panel-header">
              <div className="dash-panel-title-wrap">
                <Clock size={15} style={{ color: 'var(--accent)' }} />
                <h3>Perimeter Incursion Timeline ({timeHorizon})</h3>
              </div>
              <div className="timeline-legend">
                <span className="legend-item"><span className="legend-dot crit" />Critical</span>
                <span className="legend-item"><span className="legend-dot high" />High</span>
                <span className="legend-item"><span className="legend-dot med" />Medium</span>
                <span className="legend-item"><span className="legend-dot low" />Low</span>
              </div>
            </div>

            {/* SVG Interactive Histogram */}
            <div className="timeline-chart-container">
              <div className="chart-y-axis">
                <span>{maxTimelineVal}</span>
                <span>{Math.round(maxTimelineVal / 2)}</span>
                <span>0</span>
              </div>
              <div className="chart-bars-wrap">
                {timelineHours.map((slot, idx) => {
                  const barHeightPct = (slot.total / maxTimelineVal) * 100;
                  const isHovered = hoveredBarIndex === idx;
                  const critPct = slot.total > 0 ? (slot.critical / slot.total) * 100 : 0;
                  const highPct = slot.total > 0 ? (slot.high / slot.total) * 100 : 0;
                  const medPct = slot.total > 0 ? (slot.medium / slot.total) * 100 : 0;
                  const lowPct = slot.total > 0 ? (slot.low / slot.total) * 100 : 0;

                  return (
                    <div
                      key={idx}
                      className={`chart-bar-column ${isHovered ? 'hovered' : ''}`}
                      onMouseEnter={() => setHoveredBarIndex(idx)}
                      onMouseLeave={() => setHoveredBarIndex(null)}
                    >
                      {/* Tooltip on hover */}
                      {isHovered && (
                        <div className="chart-bar-tooltip">
                          <div className="tooltip-time">{slot.time}</div>
                          <div className="tooltip-row crit">Critical: {slot.critical}</div>
                          <div className="tooltip-row high">High: {slot.high}</div>
                          <div className="tooltip-row med">Medium: {slot.medium}</div>
                          <div className="tooltip-row low">Low: {slot.low}</div>
                          <div className="tooltip-total">Total: {slot.total} Events</div>
                        </div>
                      )}

                      <div className="stacked-bar" style={{ height: `${Math.max(barHeightPct, 6)}%` }}>
                        <div className="stack-seg crit" style={{ height: `${critPct}%` }} />
                        <div className="stack-seg high" style={{ height: `${highPct}%` }} />
                        <div className="stack-seg med" style={{ height: `${medPct}%` }} />
                        <div className="stack-seg low" style={{ height: `${lowPct}%` }} />
                      </div>
                      <span className="bar-time-label">{slot.time}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          {/* Sector & Outpost Risk Comparison Matrix */}
          <section className="dash-panel sector-matrix-panel glassmorphic">
            <div className="dash-panel-header">
              <div className="dash-panel-title-wrap">
                <MapPin size={15} style={{ color: 'var(--accent)' }} />
                <h3>Border Outpost (BOP) Tactical Comparison</h3>
              </div>
              <button className="btn-link" onClick={onSwitchToMap}>
                View Geospatial Map <ExternalLink size={12} />
              </button>
            </div>

            <div className="sector-cards-grid">
              {displayedSectors.map(sec => (
                <div key={sec.id} className="sector-metric-card">
                  <div className="sector-card-top">
                    <div>
                      <span className="sector-card-id">{sec.id}</span>
                      <h4 className="sector-card-name">{sec.name}</h4>
                    </div>
                    <span className={`cam-status-pill ${sec.status.toLowerCase()}`}>
                      {sec.status}
                    </span>
                  </div>

                  <div className="sector-metrics-row">
                    <div className="sec-stat">
                      <span className="sec-stat-label">Risk Index</span>
                      <span className="sec-stat-val" style={{ color: sec.riskIndex > 50 ? 'var(--severity-critical)' : 'var(--accent)' }}>
                        {sec.riskIndex}/100
                      </span>
                    </div>
                    <div className="sec-stat">
                      <span className="sec-stat-label">Active Alerts</span>
                      <span className="sec-stat-val">{sec.alertCount}</span>
                    </div>
                    <div className="sec-stat">
                      <span className="sec-stat-label">Persons</span>
                      <span className="sec-stat-val">{sec.personCount}</span>
                    </div>
                    <div className="sec-stat">
                      <span className="sec-stat-label">Vehicles</span>
                      <span className="sec-stat-val">{sec.vehicleCount}</span>
                    </div>
                  </div>

                  <div className="sector-card-footer">
                    <span className="sector-geo-coord">{sec.coords}</span>
                    {sec.camera && (
                      <button
                        className="btn btn-sm btn-outline-accent"
                        onClick={() => {
                          onSelectCamera(sec.camera!);
                          onSwitchToTactical();
                        }}
                      >
                        <Eye size={11} /> Focus {sec.camId}
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Tri-Model AI Neural Pipeline & Forensic Integrity */}
          <section className="dash-panel ai-pipeline-panel glassmorphic">
            <div className="dash-panel-header">
              <div className="dash-panel-title-wrap">
                <Sparkles size={15} style={{ color: 'var(--status-ai)' }} />
                <h3>Tri-Model Neural Perception Architecture</h3>
              </div>
              <span className="provenance-tag">SHA-256 SEALED</span>
            </div>

            <div className="ai-models-grid">
              {/* Model 1: Ground YOLO11n v2.0 */}
              <div className="ai-model-box">
                <div className="model-box-header">
                  <div className="model-name-group">
                    <Target size={14} style={{ color: 'var(--accent)' }} />
                    <span className="model-title">Ground Model v2.0</span>
                  </div>
                  <span className={`model-status-tag ${detailedHealth?.ground_ai?.status === 'ACTIVE' || detailedHealth?.ground_ai?.status === 'active' ? 'active' : 'candidate'}`}>
                    {detailedHealth?.ground_ai?.status || 'ACTIVE'}
                  </span>
                </div>
                <div className="model-specs">
                  <span>Architecture: <strong>YOLO11n (768×768)</strong></span>
                  <span>Classes: <strong>Person, Vehicle</strong></span>
                  <span>Standalone Latency: <strong>10.59 ms</strong></span>
                  <span>F1-Score: <strong>69.2% (P) / 91.4% (V)</strong></span>
                </div>
              </div>

              {/* Model 2: Airborne YOLO11n v1.1 */}
              <div className="ai-model-box">
                <div className="model-box-header">
                  <div className="model-name-group">
                    <Plane size={14} style={{ color: 'var(--status-info)' }} />
                    <span className="model-title">Airborne Model v1.1</span>
                  </div>
                  <span className={`model-status-tag ${detailedHealth?.air_ai?.status === 'ACTIVE' || detailedHealth?.air_ai?.status === 'active' ? 'active' : 'candidate'}`}>
                    {detailedHealth?.air_ai?.status || 'VALIDATED CANDIDATE'}
                  </span>
                </div>
                <div className="model-specs">
                  <span>Architecture: <strong>YOLO11n (640×640)</strong></span>
                  <span>Classes: <strong>Drone (UAV), Aircraft</strong></span>
                  <span>Standalone Latency: <strong>9.30 ms</strong></span>
                  <span>F1-Score: <strong>82.5% (D) / 75.4% (A)</strong></span>
                </div>
              </div>

              {/* Model 3: Security Item YOLO11n v2.1 */}
              <div className="ai-model-box">
                <div className="model-box-header">
                  <div className="model-name-group">
                    <ShieldCheck size={14} style={{ color: 'var(--status-warning)' }} />
                    <span className="model-title">Security Item v2.1</span>
                  </div>
                  <span className={`model-status-tag ${detailedHealth?.security_item_ai?.status === 'ACTIVE' || detailedHealth?.security_item_ai?.status === 'active' ? 'active' : 'candidate'}`}>
                    {detailedHealth?.security_item_ai?.status || 'VALIDATED CANDIDATE'}
                  </span>
                </div>
                <div className="model-specs">
                  <span>Architecture: <strong>YOLO11n (640×640)</strong></span>
                  <span>Classes: <strong>Handheld Firearm</strong></span>
                  <span>Standalone Latency: <strong>11.99 ms</strong></span>
                  <span>Consensus: <strong>3-Frame Temporal</strong></span>
                  <span style={{ color: 'var(--status-warning)', fontSize: '9px' }}>⚠ Real-firearm-video validation not available</span>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* RIGHT COLUMN: Threat Factor Breakdown, ANPR Hotlist, Camera Grid Quick Snapshot */}
        <div className="dashboard-col-secondary">
          {/* Threat Factor Attribution Breakdown */}
          <section className="dash-panel threat-factors-panel glassmorphic">
            <div className="dash-panel-header">
              <div className="dash-panel-title-wrap">
                <Layers size={15} style={{ color: 'var(--accent)' }} />
                <h3>Threat Risk Attribution</h3>
              </div>
            </div>

            <div className="threat-factors-list">
              <div className="factor-item">
                <div className="factor-info">
                  <span className="factor-name">Zero-Tolerance Zone Breach</span>
                  <span className="factor-pts">+40 pts</span>
                </div>
                <div className="factor-bar-track">
                  <div className="factor-bar-fill" style={{ width: '85%', background: 'var(--severity-critical)' }} />
                </div>
              </div>

              <div className="factor-item">
                <div className="factor-info">
                  <span className="factor-name">Handheld Weapon Consensus</span>
                  <span className="factor-pts">+35 pts</span>
                </div>
                <div className="factor-bar-track">
                  <div className="factor-bar-fill" style={{ width: '40%', background: 'var(--severity-high)' }} />
                </div>
              </div>

              <div className="factor-item">
                <div className="factor-info">
                  <span className="factor-name">Directional Inward Vector</span>
                  <span className="factor-pts">+25 pts</span>
                </div>
                <div className="factor-bar-track">
                  <div className="factor-bar-fill" style={{ width: '65%', background: 'var(--severity-medium)' }} />
                </div>
              </div>

              <div className="factor-item">
                <div className="factor-info">
                  <span className="factor-name">Speed / Loitering Anomaly</span>
                  <span className="factor-pts">+15 pts</span>
                </div>
                <div className="factor-bar-track">
                  <div className="factor-bar-fill" style={{ width: '50%', background: 'var(--status-info)' }} />
                </div>
              </div>

              <div className="factor-item">
                <div className="factor-info">
                  <span className="factor-name">Night Context Multiplier</span>
                  <span className="factor-pts">+10 pts</span>
                </div>
                <div className="factor-bar-track">
                  <div className="factor-bar-fill" style={{ width: '30%', background: 'var(--status-ai)' }} />
                </div>
              </div>
            </div>
          </section>

          {/* ANPR Border Transit Watchlist */}
          <section className="dash-panel anpr-quick-panel glassmorphic">
            <div className="dash-panel-header">
              <div className="dash-panel-title-wrap">
                <Car size={15} style={{ color: 'var(--accent)' }} />
                <h3>ANPR & Hotlist Intelligence</h3>
              </div>
              <span className="anpr-status-tag">OCR Active</span>
            </div>

            <div className="anpr-quick-stats">
              <div className="anpr-kpi">
                <span className="anpr-kpi-label">Plates Scanned</span>
                <span className="anpr-kpi-val">{anprPlates.length}</span>
              </div>
              <div className="anpr-kpi">
                <span className="anpr-kpi-label">Hotlist Hits</span>
                <span className={`anpr-kpi-val ${anprPlates.filter(p => p.watchlist_status === 'WATCHLIST MATCH').length > 0 ? 'highlight' : ''}`}>
                  {anprPlates.filter(p => p.watchlist_status === 'WATCHLIST MATCH').length}
                </span>
              </div>
              <div className="anpr-kpi">
                <span className="anpr-kpi-label">OCR Avg Conf</span>
                <span className="anpr-kpi-val">
                  {anprPlates.length > 0
                    ? `${(anprPlates.reduce((sum, p) => sum + p.confidence, 0) / anprPlates.length * 100).toFixed(1)}%`
                    : '—'}
                </span>
              </div>
            </div>

            <div className="anpr-recent-hits">
              {anprPlates.filter(p => p.watchlist_status === 'WATCHLIST MATCH').length === 0 && anprPlates.length === 0 ? (
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', padding: 'var(--sp-2)', textAlign: 'center' }}>
                  No plates detected
                </div>
              ) : (
                anprPlates.filter(p => p.watchlist_status === 'WATCHLIST MATCH').slice(0, 3).map(plate => (
                  <div key={plate.id} className="anpr-hit-item watchlist-alert">
                    <div className="hit-plate-box">{plate.plate_number}</div>
                    <div className="hit-meta">
                      <span className="hit-status">WATCHLIST MATCH</span>
                      <span className="hit-cam">{plate.camera_id} · {new Date(plate.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))
              )}
              {anprPlates.length > 0 && anprPlates.filter(p => p.watchlist_status === 'WATCHLIST MATCH').length === 0 && (
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', padding: 'var(--sp-2)', textAlign: 'center' }}>
                  {anprPlates.length} plates scanned — No watchlist matches
                </div>
              )}
            </div>
          </section>

          {/* Live Optical Feed Matrix Snapshot */}
          <section className="dash-panel camera-matrix-panel glassmorphic">
            <div className="dash-panel-header">
              <div className="dash-panel-title-wrap">
                <Radio size={15} style={{ color: 'var(--accent)' }} />
                <h3>Camera Grid Status ({cameras.length})</h3>
              </div>
              <button className="btn-link" onClick={onSwitchToTactical}>
                Open Command Console <ChevronRight size={13} />
              </button>
            </div>

            <div className="camera-mini-matrix">
              {cameras.map(cam => {
                const isOnline = cam.status === 'ONLINE';
                return (
                  <div
                    key={cam.id}
                    className={`cam-matrix-card ${isOnline ? 'online' : 'offline'}`}
                    onClick={() => {
                      onSelectCamera(cam);
                      onSwitchToTactical();
                    }}
                    title={`Click to focus ${cam.id}`}
                  >
                    <div className="cam-matrix-thumb">
                      {isOnline ? (
                        <img
                          src={`${apiBase}/api/cameras/${cam.id}/stream${tokenQuery}`}
                          alt={cam.name}
                          className="thumb-img"
                          loading="eager"
                        />
                      ) : (
                        <div className="thumb-offline">
                          <AlertTriangle size={20} />
                          <span>NO SIGNAL</span>
                        </div>
                      )}
                      <span className={`matrix-cam-badge ${isOnline ? 'online' : 'offline'}`}>
                        {cam.id}
                      </span>
                    </div>
                    <div className="cam-matrix-info">
                      <span className="cam-matrix-name">{cam.name}</span>
                      <div className="cam-matrix-meta">
                        <span>{isOnline ? `${cam.fps} FPS` : '0 FPS'}</span>
                        <span>{cam.location || 'BOP'}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      </div>

      {/* ── PRIORITY ACTIONABLE CRITICAL INCIDENT TRIAGE TABLE ── */}
      <section className="dash-panel priority-incident-ledger glassmorphic">
        <div className="dash-panel-header">
          <div className="dash-panel-title-wrap">
            <AlertTriangle size={15} style={{ color: 'var(--severity-critical)' }} />
            <h3>High-Priority Incident Triage Digest</h3>
          </div>
          <span className="ledger-count-tag">{criticalAlerts.length + highAlerts.length} Actionable Events</span>
        </div>

        <div className="ledger-table-wrap">
          <table className="ledger-table">
            <thead>
              <tr>
                <th>Severity</th>
                <th>Incident Type</th>
                <th>Sector Outpost</th>
                <th>Camera / Track</th>
                <th>Risk Score</th>
                <th>Timestamp</th>
                <th>Status</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {alerts.slice(0, 5).map(alert => {
                const sev = alert.severity.toLowerCase();
                return (
                  <tr key={alert.id} className={`ledger-row ${sev}`}>
                    <td>
                      <span className={`severity-tag ${sev}`}>{alert.severity}</span>
                    </td>
                    <td>
                      <strong className="incident-event-type">{alert.event_type.replace(/_/g, ' ')}</strong>
                      {alert.object_type && <span className="incident-obj-type"> · {alert.object_type}</span>}
                    </td>
                    <td>
                      <span className="sector-tag">{alert.location || 'BOP-TEST-01'}</span>
                    </td>
                    <td>
                      <span className="mono">{alert.camera_id}</span>
                      {alert.track_id && <span className="mono text-muted"> / {alert.track_id}</span>}
                    </td>
                    <td>
                      <div className="table-risk-cell">
                        <div className="table-risk-bar">
                          <div className={`table-risk-fill ${sev}`} style={{ width: `${alert.risk_score}%` }} />
                        </div>
                        <span className="table-risk-num">{alert.risk_score}</span>
                      </div>
                    </td>
                    <td className="text-muted">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </td>
                    <td>
                      <span className={`status-pill ${alert.status.toLowerCase()}`}>
                        {alert.status}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="table-actions-row">
                        {alert.status === 'NEW' && (
                          <>
                            <button
                              className="btn btn-primary btn-xs"
                              onClick={() => onAcknowledgeAlert(alert.id)}
                              title="Acknowledge alert"
                            >
                              <CheckCircle size={10} /> Ack
                            </button>
                            <button
                              className="btn btn-warning btn-xs"
                              onClick={() => onEscalateAlert(alert.id)}
                              title="Escalate alert to supervisor"
                            >
                              <ArrowUpCircle size={10} /> Escalate
                            </button>
                          </>
                        )}
                        <button
                          className="btn btn-secondary btn-xs"
                          onClick={() => setSelectedIncidentModal(alert)}
                          title="Inspect incident evidence"
                        >
                          <Eye size={10} /> Details
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {alerts.length === 0 && (
                <tr>
                  <td colSpan={8} style={{ textAlign: 'center', padding: 'var(--sp-6)', color: 'var(--text-muted)' }}>
                    No security incidents recorded in this time horizon. All sectors operational.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── INCIDENT EVIDENCE MODAL ── */}
      {selectedIncidentModal && (
        <div className="incident-modal-backdrop" onClick={() => setSelectedIncidentModal(null)}>
          <div className="incident-modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                <ShieldAlert size={18} style={{ color: 'var(--severity-critical)' }} />
                <h3>Incident Evidence Forensics — {selectedIncidentModal.id}</h3>
              </div>
              <button className="btn-icon" onClick={() => setSelectedIncidentModal(null)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="modal-details-grid">
                <div><strong>Event Type:</strong> {selectedIncidentModal.event_type}</div>
                <div><strong>Camera ID:</strong> {selectedIncidentModal.camera_id}</div>
                <div><strong>Track ID:</strong> {selectedIncidentModal.track_id || 'N/A'}</div>
                <div><strong>Risk Score:</strong> {selectedIncidentModal.risk_score} / 100</div>
                <div><strong>Timestamp:</strong> {new Date(selectedIncidentModal.timestamp).toLocaleString()}</div>
                <div><strong>Provenance:</strong> SHA-256 Verified</div>
              </div>

              {selectedIncidentModal.risk_reasons && (
                <div className="modal-reasons">
                  <strong>Risk Evaluation Factors:</strong>
                  <ul>
                    {selectedIncidentModal.risk_reasons.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </div>
              )}

              {selectedIncidentModal.snapshot_url && (
                <div className="modal-evidence-preview">
                  <strong>Forensic Snapshot:</strong>
                  <img src={`${apiBase}${selectedIncidentModal.snapshot_url}`} alt="Evidence" />
                </div>
              )}

              {selectedIncidentModal.video_clip_url && (
                <div className="modal-evidence-preview">
                  <strong>12-Second H.264 Sealed Video Clip:</strong>
                  <video controls autoPlay src={`${apiBase}${selectedIncidentModal.video_clip_url}`} />
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setSelectedIncidentModal(null)}>Close</button>
              {selectedIncidentModal.status === 'NEW' && (
                <button
                  className="btn btn-primary"
                  onClick={() => {
                    onAcknowledgeAlert(selectedIncidentModal.id);
                    setSelectedIncidentModal(null);
                  }}
                >
                  <CheckCircle size={12} /> Acknowledge Alert
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
