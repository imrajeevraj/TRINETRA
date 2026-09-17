import React, { useState } from 'react';
import type { Alert, Camera, PlateEvent, DetailedHealth } from '../types';
import {
  ShieldAlert,
  Crosshair,
  Car as CarIcon,
  Activity,
  Search,
  X,
  CheckCircle,
  XCircle,
  ArrowUpCircle,
  Bell,
} from 'lucide-react';
import { SystemHealthPanel } from './SystemHealthPanel';
import { AnprPanel } from './AnprPanel';

interface EventFilters {
  search: string;
  cameraId: string;
  severity: string;
  eventType: string;
  status: string;
  timeRange: 'all' | '15m' | '1h' | '24h';
}

const DEFAULT_FILTERS: EventFilters = {
  search: '',
  cameraId: 'ALL',
  severity: 'ALL',
  eventType: 'ALL',
  status: 'ALL',
  timeRange: 'all',
};

interface IntelligencePanelProps {
  alerts: Alert[];
  cameras: Camera[];
  selectedCamera: Camera | null;
  apiBase: string;
  dataOrigin: 'LIVE' | 'DEMO' | 'IMPORTED';
  filters: EventFilters;
  setFilters: React.Dispatch<React.SetStateAction<EventFilters>>;
  detailedHealth: DetailedHealth | null;
  anprPlates: PlateEvent[];
  onAcknowledge: (id: string) => void;
  onDismiss: (id: string) => void;
  onEscalate: (id: string) => void;
}

export const IntelligencePanel: React.FC<IntelligencePanelProps> = ({
  alerts,
  cameras,
  selectedCamera,
  apiBase,
  dataOrigin,
  filters,
  setFilters,
  detailedHealth,
  anprPlates,
  onAcknowledge,
  onDismiss,
  onEscalate,
}) => {
  const [activeTab, setActiveTab] = useState<'alerts' | 'tracks' | 'anpr' | 'health'>('alerts');
  const [maxAlerts, setMaxAlerts] = useState(50);
  const [expandedReasons, setExpandedReasons] = useState<Set<string>>(new Set());

  const criticalCount = alerts.filter(a => a.severity === 'CRITICAL' && a.status === 'NEW').length;
  const highCount = alerts.filter(a => a.severity === 'HIGH' && a.status === 'NEW').length;
  const mediumCount = alerts.filter(a => a.severity === 'MEDIUM' && a.status === 'NEW').length;
  const lowCount = alerts.filter(a => a.severity === 'LOW' && a.status === 'NEW').length;

  const hasFilters = filters.search || filters.cameraId !== 'ALL' || filters.severity !== 'ALL' || filters.eventType !== 'ALL' || filters.status !== 'ALL' || filters.timeRange !== 'all';

  return (
    <section className="intelligence-panel" aria-label="Intelligence Panel">
      {/* Tabs */}
      <div className="panel-tabs" role="tablist">
        <button
          className={`panel-tab ${activeTab === 'alerts' ? 'active' : ''}`}
          onClick={() => setActiveTab('alerts')}
          role="tab"
          aria-selected={activeTab === 'alerts'}
        >
          <ShieldAlert size={13} /> Alerts
          {alerts.length > 0 && (
            <span className={`tab-count ${criticalCount > 0 ? 'has-critical' : ''}`}>
              {alerts.length}
            </span>
          )}
        </button>
        <button
          className={`panel-tab ${activeTab === 'tracks' ? 'active' : ''}`}
          onClick={() => setActiveTab('tracks')}
          role="tab"
          aria-selected={activeTab === 'tracks'}
        >
          <Crosshair size={13} /> Tracks
        </button>
        <button
          className={`panel-tab ${activeTab === 'anpr' ? 'active' : ''}`}
          onClick={() => setActiveTab('anpr')}
          role="tab"
          aria-selected={activeTab === 'anpr'}
        >
          <CarIcon size={13} /> ANPR
        </button>
        <button
          className={`panel-tab ${activeTab === 'health' ? 'active' : ''}`}
          onClick={() => setActiveTab('health')}
          role="tab"
          aria-selected={activeTab === 'health'}
        >
          <Activity size={13} /> Health
        </button>
      </div>

      <div className="panel-content">
        {/* ── ALERTS TAB ─────────────────────────────────── */}
        <div className={activeTab === 'alerts' ? '' : 'tab-hidden'}>
          <>
            {/* Severity Summary */}
            <div className="severity-summary">
              {criticalCount > 0 && (
                <div className="severity-count critical"><span className="sev-dot" />{criticalCount} Critical</div>
              )}
              {highCount > 0 && (
                <div className="severity-count high"><span className="sev-dot" />{highCount} High</div>
              )}
              {mediumCount > 0 && (
                <div className="severity-count medium"><span className="sev-dot" />{mediumCount} Medium</div>
              )}
              {lowCount > 0 && (
                <div className="severity-count low"><span className="sev-dot" />{lowCount} Low</div>
              )}
              {alerts.length === 0 && (
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>No active alerts</span>
              )}
            </div>

            {/* Filters */}
            <div className="alert-filters">
              <div className="alert-search">
                <Search size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                <input
                  type="text"
                  placeholder="Search alerts..."
                  value={filters.search}
                  onChange={e => setFilters(prev => ({ ...prev, search: e.target.value }))}
                  aria-label="Search alerts"
                />
                {filters.search && (
                  <button className="alert-search-clear" onClick={() => setFilters(prev => ({ ...prev, search: '' }))}>
                    <X size={11} />
                  </button>
                )}
              </div>

              <div className="filter-row">
                <select className="filter-select" value={filters.severity} onChange={e => setFilters(prev => ({ ...prev, severity: e.target.value }))} aria-label="Filter by severity">
                  <option value="ALL">All Severity</option>
                  <option value="CRITICAL">Critical</option>
                  <option value="HIGH">High</option>
                  <option value="MEDIUM">Medium</option>
                  <option value="LOW">Low</option>
                </select>
                <select className="filter-select" value={filters.cameraId} onChange={e => setFilters(prev => ({ ...prev, cameraId: e.target.value }))} aria-label="Filter by camera">
                  <option value="ALL">All Cameras</option>
                  {cameras.map(c => (
                    <option key={c.id} value={c.id}>{c.id}</option>
                  ))}
                </select>
              </div>

              <div className="filter-row">
                <select className="filter-select" value={filters.eventType} onChange={e => setFilters(prev => ({ ...prev, eventType: e.target.value }))} aria-label="Filter by event type">
                  <option value="ALL">All Types</option>
                  <option value="ZONE_ENTRY">Zone Entry</option>
                  <option value="ZONE_EXIT">Zone Exit</option>
                  <option value="VIRTUAL_FENCE_CROSSING">Fence Cross</option>
                </select>
                <select className="filter-select" value={filters.status} onChange={e => setFilters(prev => ({ ...prev, status: e.target.value }))} aria-label="Filter by status">
                  <option value="ALL">All Status</option>
                  <option value="NEW">New</option>
                  <option value="ACKNOWLEDGED">Acknowledged</option>
                  <option value="ESCALATED">Escalated</option>
                </select>
              </div>

              <div className="time-filters">
                {(['all', '15m', '1h', '24h'] as const).map(t => (
                  <button
                    key={t}
                    className={`time-btn ${filters.timeRange === t ? 'active' : ''}`}
                    onClick={() => setFilters(prev => ({ ...prev, timeRange: t }))}
                  >
                    {t === 'all' ? 'All' : t.toUpperCase()}
                  </button>
                ))}
              </div>

              <div className="filter-meta">
                <span>{alerts.length} events</span>
                {hasFilters && (
                  <button className="filter-reset" onClick={() => setFilters(DEFAULT_FILTERS)}>Reset</button>
                )}
              </div>
            </div>

            {/* Alert List */}
            <div className="alert-list">
              {alerts.length === 0 ? (
                <div className="empty-state">
                  <Bell size={24} className="empty-state-icon" />
                  <span className="empty-state-title">No Active Incidents</span>
                  <span className="empty-state-sub">All monitored zones are currently clear.</span>
                </div>
              ) : (
                alerts.slice(0, maxAlerts).map(alert => {
                  const sev = alert.severity.toLowerCase();
                  const showReasons = expandedReasons.has(alert.id);
                  return (
                    <div key={alert.id} className={`alert-item ${sev}`}>
                      <div className="alert-item-header">
                        <div className="alert-event-info">
                          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                            <span className="alert-event-name">{alert.event_type.replace(/_/g, ' ')}</span>
                            <span className={`severity-tag ${sev}`}>{alert.severity}</span>
                          </div>
                          <div className="alert-event-sub">
                            <span className="mono">{alert.camera_id}</span>
                            {alert.track_id && <span>Track: <span className="mono">{alert.track_id}</span></span>}
                            {alert.zone_id && <span>Zone: <span className="mono">{alert.zone_id}</span></span>}
                          </div>
                        </div>
                        <span className="alert-timestamp">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      {/* Details Grid */}
                      <div className="alert-details">
                        {alert.object_type && (
                          <div className="alert-detail-item">
                            <span className="alert-detail-label">Object</span>
                            <span className="alert-detail-value">{alert.object_type}</span>
                          </div>
                        )}
                        {alert.direction && (
                          <div className="alert-detail-item">
                            <span className="alert-detail-label">Direction</span>
                            <span className="alert-detail-value">{alert.direction}</span>
                          </div>
                        )}
                      </div>

                      {/* Risk Bar */}
                      <div className="risk-bar-row">
                        <span className="risk-label">Risk</span>
                        <div className="risk-bar-track">
                          <div className={`risk-bar-fill ${sev}`} style={{ width: `${alert.risk_score}%` }} />
                        </div>
                        <span className={`risk-score-value ${sev}`}>{alert.risk_score}</span>
                      </div>

                      {/* Risk Reasons */}
                      {alert.risk_reasons && alert.risk_reasons.length > 0 && (
                        <div className="risk-reasons">
                          <button
                            className="risk-reasons-toggle"
                            onClick={() => {
                              setExpandedReasons(prev => {
                                const next = new Set(prev);
                                if (next.has(alert.id)) next.delete(alert.id);
                                else next.add(alert.id);
                                return next;
                              });
                            }}
                          >
                            {showReasons ? '▾' : '▸'} {alert.risk_reasons.length} risk factor{alert.risk_reasons.length > 1 ? 's' : ''}
                          </button>
                          {showReasons && alert.risk_reasons.map((reason, i) => (
                            <span key={i} className="risk-reason">{reason}</span>
                          ))}
                        </div>
                      )}

                      {/* Evidence */}
                      {(alert.snapshot_url || alert.video_clip_url) && (
                        <details className="evidence-section">
                          <summary>Evidence {alert.video_clip_url ? '· clip ready' : '· preparing'}</summary>
                          {alert.snapshot_url && <img src={`${apiBase}${alert.snapshot_url}`} alt={`Evidence for ${alert.id}`} />}
                          {alert.video_clip_url && <video controls preload="metadata" src={`${apiBase}${alert.video_clip_url}`} />}
                        </details>
                      )}

                      {/* Actions */}
                      {alert.status === 'NEW' && (
                        <div className="alert-actions">
                          <button className="btn btn-primary btn-sm" onClick={() => onAcknowledge(alert.id)}>
                            <CheckCircle size={11} /> Acknowledge
                          </button>
                          <button className="btn btn-warning btn-sm" onClick={() => onEscalate(alert.id)}>
                            <ArrowUpCircle size={11} /> Escalate
                          </button>
                          <button className="btn btn-sm" onClick={() => onDismiss(alert.id)}>
                            <XCircle size={11} /> Dismiss
                          </button>
                        </div>
                      )}

                      {alert.status === 'ACKNOWLEDGED' && (
                        <div className="alert-status-ack">
                          <CheckCircle size={11} />
                          <span>Acknowledged{alert.handled_by ? ` by ${alert.handled_by}` : ''}</span>
                        </div>
                      )}

                      {alert.status === 'ESCALATED' && (
                        <div className="alert-status-escalated">
                          <ArrowUpCircle size={11} />
                          <span>Escalated — Awaiting supervisor</span>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
              {alerts.length > maxAlerts && (
                <button
                  className="btn btn-sm"
                  style={{ margin: 'var(--sp-2) auto', display: 'block' }}
                  onClick={() => setMaxAlerts(prev => prev + 50)}
                >
                  Show {Math.min(50, alerts.length - maxAlerts)} more of {alerts.length} total
                </button>
              )}
            </div>
          </>
        </div>

        {/* ── TRACKS TAB ─────────────────────────────────── */}
        <div className={`track-list ${activeTab === 'tracks' ? '' : 'tab-hidden'}`}>
          {!selectedCamera?.detections || selectedCamera.detections.filter(d => d.track_id).length === 0 ? (
            <div className="empty-state">
              <Crosshair size={24} className="empty-state-icon" />
              <span className="empty-state-title">No Active Tracks</span>
              <span className="empty-state-sub">No objects are currently being tracked on this camera.</span>
            </div>
          ) : (
            selectedCamera.detections.filter(d => d.track_id).map(track => (
              <div key={track.track_id} className="track-row">
                <div>
                  <span className="track-id">{track.track_id}</span>
                  <span className="track-class" style={{ marginLeft: 'var(--sp-2)' }}>{track.class}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-3)' }}>
                  <span className="track-conf">{Math.round(track.confidence * 100)}%</span>
                  <span className="track-status">ACTIVE</span>
                </div>
              </div>
            ))
          )}
        </div>

        {/* ── ANPR TAB ─────────────────────────────────── */}
        <div className={`anpr-container ${activeTab === 'anpr' ? '' : 'tab-hidden'}`} style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          <AnprPanel plates={anprPlates} dataOrigin={dataOrigin} />
        </div>

        {/* ── HEALTH TAB ─────────────────────────────────── */}
        <div className={`health-container ${activeTab === 'health' ? '' : 'tab-hidden'}`} style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          <SystemHealthPanel health={detailedHealth} />
        </div>
      </div>
    </section>
  );
};

export type { EventFilters };
export { DEFAULT_FILTERS };
