import type { DetailedHealth } from '../types';

function barClass(value: number): string {
  if (value >= 90) return 'critical';
  if (value >= 70) return 'warning';
  return 'ok';
}

export function SystemHealthPanel({ health }: { health: DetailedHealth | null }) {
  if (!health) {
    return (
      <div className="health-section">
        <div className="empty-state">
          <span className="empty-state-sub">Loading system health…</span>
        </div>
      </div>
    );
  }

  const statusClass = health.status === 'HEALTHY' ? 'ok' : health.status === 'DEGRADED' ? 'warning' : 'error';
  const ramPercent = health.memory?.percent ?? 0;
  const cpuPercent = health.cpu?.percent ?? 0;
  const gpuPercent = health.gpu?.percent ?? 0;
  const vramPercent = health.gpu?.memory_used_mb != null && health.gpu?.memory_total_mb != null
    ? Math.round((health.gpu.memory_used_mb / health.gpu.memory_total_mb) * 100) 
    : 0;

  const staleText = health.cameras.stale.length
    ? `AI results stale: ${health.cameras.stale.join(', ')}`
    : 'All AI results current';

  // Determine degradation reason
  let degradeReason = '';
  if (health.status !== 'HEALTHY') {
    if (ramPercent >= 90) degradeReason = `RAM usage elevated at ${ramPercent}%`;
    else if (cpuPercent >= 90) degradeReason = `CPU usage elevated at ${cpuPercent}%`;
    else if (health.cameras.stale.length > 0) degradeReason = `Stale AI results on ${health.cameras.stale.length} camera(s)`;
    else degradeReason = 'Performance below optimal thresholds';
  }

  return (
    <div className="health-section">
      {/* Overall Status */}
      <div className="health-status-bar">
        <span className="health-status-label">System Status</span>
        <span className={`health-status-value ${statusClass}`}>{health.status}</span>
      </div>

      {degradeReason && (
        <div className={`health-stale-warn warning`}>
          {degradeReason}
        </div>
      )}

      {/* Progress Bar Metrics */}
      <div className="health-metrics">
        <div className="health-metric">
          <span className="health-metric-label">CPU</span>
          <div className="health-bar-track">
            <div className={`health-bar-fill ${barClass(cpuPercent)}`} style={{ width: `${cpuPercent}%` }} />
          </div>
          <span className="health-metric-value">{cpuPercent}%</span>
        </div>

        <div className="health-metric">
          <span className="health-metric-label">RAM</span>
          <div className="health-bar-track">
            <div className={`health-bar-fill ${barClass(ramPercent)}`} style={{ width: `${ramPercent}%` }} />
          </div>
          <span className="health-metric-value">{ramPercent}%</span>
        </div>

        {health.gpu && (
          <>
            <div className="health-metric">
              <span className="health-metric-label">GPU</span>
              <div className="health-bar-track">
                <div className={`health-bar-fill ${barClass(gpuPercent)}`} style={{ width: `${gpuPercent}%` }} />
              </div>
              <span className="health-metric-value">{gpuPercent}%</span>
            </div>

            <div className="health-metric">
              <span className="health-metric-label">VRAM</span>
              <div className="health-bar-track">
                <div className={`health-bar-fill ${barClass(vramPercent)}`} style={{ width: `${vramPercent}%` }} />
              </div>
              <span className="health-metric-value">{vramPercent}%</span>
            </div>
          </>
        )}
      </div>

      {/* Detail Grid */}
      <div className="health-detail-grid">
        <div className="health-detail">
          <span className="health-detail-label">Cameras</span>
          <span className="health-detail-value">{health.cameras.online}/{health.cameras.total}</span>
        </div>
        <div className="health-detail">
          <span className="health-detail-label">AI FPS</span>
          <span className="health-detail-value">{health.cameras.inference_fps}</span>
        </div>
        <div className="health-detail">
          <span className="health-detail-label">ANPR Queue</span>
          <span className="health-detail-value">{health.anpr?.queue_depth ?? 0}</span>
        </div>
        <div className="health-detail">
          <span className="health-detail-label">Database</span>
          <span className="health-detail-value">{health.database_status}</span>
        </div>

        {health.ground_ai && (
          <div className="health-detail" style={{ gridColumn: '1 / -1' }}>
            <span className="health-detail-label">Ground AI (v2.0)</span>
            <span className="health-detail-value" style={{ fontSize: 'var(--text-xs)' }}>
              [{health.ground_ai.status}] {health.ground_ai.fps} FPS · Persons: {health.ground_ai.persons} · Vehicles: {health.ground_ai.vehicles}
            </span>
          </div>
        )}

        {health.air_ai && (
          <div className="health-detail" style={{ gridColumn: '1 / -1' }}>
            <span className="health-detail-label">Airborne AI (v1.1)</span>
            <span className="health-detail-value" style={{ fontSize: 'var(--text-xs)' }}>
              [{health.air_ai.status}] {health.air_ai.fps} FPS · Drones: {health.air_ai.drones} · Aircraft: {health.air_ai.aircraft}
            </span>
          </div>
        )}

        {health.security_item_ai && (
          <div className="health-detail" style={{ gridColumn: '1 / -1' }}>
            <span className="health-detail-label">Security Item AI ({health.security_item_ai.version})</span>
            <span className="health-detail-value" style={{ fontSize: 'var(--text-xs)' }}>
              [{health.security_item_ai.status}] {health.security_item_ai.fps} FPS · Firearms: {health.security_item_ai.firearms} · Model: {health.security_item_ai.model}
            </span>
          </div>
        )}

        {health.virtual_fence && (
          <div className="health-detail" style={{ gridColumn: '1 / -1' }}>
            <span className="health-detail-label">Virtual Fence</span>
            <span className="health-detail-value" style={{ fontSize: 'var(--text-xs)' }}>
              Ground Zones: {health.virtual_fence.ground_zones} · Air Zones: {health.virtual_fence.air_zones} · Tripwires: {health.virtual_fence.tripwires}
            </span>
          </div>
        )}

        {health.gpu && (
          <div className="health-detail" style={{ gridColumn: '1 / -1' }}>
            <span className="health-detail-label">GPU</span>
            <span className="health-detail-value" style={{ fontSize: 'var(--text-xs)' }}>{health.gpu.name}</span>
          </div>
        )}
      </div>

      {/* Stale Warning */}
      <div className={`health-stale-warn ${health.cameras.stale.length ? 'warning' : 'ok'}`}>
        {staleText}
      </div>
    </div>
  );
}
