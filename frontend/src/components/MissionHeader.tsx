import React from 'react';
import type { SystemStats } from '../types';
import {
  LogOut,
  User as UserIcon,
  Video,
  BarChart3,
  Map as MapIcon,
} from 'lucide-react';

interface MissionHeaderProps {
  time: string;
  username: string;
  role: string;
  stats: SystemStats;
  dataOrigin: 'LIVE' | 'DEMO' | 'IMPORTED';
  systemStatus: 'operational' | 'degraded' | 'critical';
  activeAppView: 'tactical' | 'dashboard' | 'map';
  onSelectAppView: (view: 'tactical' | 'dashboard' | 'map') => void;
  onLogout: () => void;
}

const STATUS_LABELS: Record<string, string> = {
  operational: 'ALL SYSTEMS OPERATIONAL',
  degraded: 'SYSTEM DEGRADED',
  critical: 'INCIDENT ACTIVE',
};

export const MissionHeader: React.FC<MissionHeaderProps> = ({
  time,
  username,
  role,
  stats,
  dataOrigin,
  systemStatus,
  activeAppView,
  onSelectAppView,
  onLogout,
}) => {
  return (
    <header className="mission-header" role="banner">
      {/* Left: Brand + Status */}
      <div className="header-left">
        <div className="brand" onClick={() => onSelectAppView('tactical')} style={{ cursor: 'pointer' }}>
          <span className="brand-name">TRINETRA</span>
          <span className="brand-subtitle">Command Center</span>
        </div>

        <div className={`system-status-pill ${systemStatus}`} role="status" aria-label={`System status: ${STATUS_LABELS[systemStatus]}`}>
          <span className="status-indicator" />
          {STATUS_LABELS[systemStatus]}
        </div>
      </div>

      {/* Primary Navigation Modes */}
      <nav className="header-nav-modes" aria-label="Primary Navigation">
        <button
          className={`nav-mode-btn ${activeAppView === 'tactical' ? 'active' : ''}`}
          onClick={() => onSelectAppView('tactical')}
          title="Live multi-camera tactical surveillance console"
        >
          <Video size={13} />
          <span>Tactical Console</span>
        </button>

        <button
          className={`nav-mode-btn ${activeAppView === 'dashboard' ? 'active' : ''}`}
          onClick={() => onSelectAppView('dashboard')}
          title="High-level strategic threat & operational analytics dashboard"
        >
          <BarChart3 size={13} />
          <span>Executive Dashboard</span>
          {stats.critical_alerts > 0 && (
            <span className="nav-badge-critical">{stats.critical_alerts}</span>
          )}
        </button>

        <button
          className={`nav-mode-btn ${activeAppView === 'map' ? 'active' : ''}`}
          onClick={() => onSelectAppView('map')}
          title="Full-screen geospatial border outpost map"
        >
          <MapIcon size={13} />
          <span>Geospatial Map</span>
        </button>
      </nav>

      {/* Center: Compact Telemetry */}
      <div className="header-telemetry" aria-label="Operational telemetry">
        <div className="telemetry-item">
          <span className="telemetry-label">Cameras</span>
          <span className="telemetry-value">{stats.cameras_online}/{stats.total_cameras}</span>
        </div>
        <span className="telemetry-separator" />
        <div className="telemetry-item">
          <span className="telemetry-label">Persons</span>
          <span className="telemetry-value">{String(stats.person_detections).padStart(2, '0')}</span>
        </div>
        <div className="telemetry-item">
          <span className="telemetry-label">Vehicles</span>
          <span className="telemetry-value">{String(stats.vehicle_detections).padStart(2, '0')}</span>
        </div>
        <span className="telemetry-separator" />
        <div className="telemetry-item">
          <span className="telemetry-label">Critical</span>
          <span className="telemetry-value" style={{ color: stats.critical_alerts > 0 ? 'var(--severity-critical)' : undefined }}>
            {String(stats.critical_alerts).padStart(2, '0')}
          </span>
        </div>
      </div>

      {/* Right: Clock, Env Badge, User, Logout */}
      <div className="header-right">
        <span className="header-clock" aria-label="Current time">{time}</span>

        <span className={`env-badge ${dataOrigin === 'LIVE' ? 'live' : ''}`}>
          {dataOrigin === 'LIVE' ? '● LIVE' : '◉ SIMULATION'}
        </span>

        <div className="user-chip" aria-label={`Operator: ${username}`}>
          <UserIcon size={13} />
          <span>{username || 'Operator'}</span>
          <span className="user-role-tag">{role || 'VIEWER'}</span>
        </div>

        <button className="btn-icon" onClick={onLogout} title="Sign out" aria-label="Sign out">
          <LogOut size={16} />
        </button>
      </div>
    </header>
  );
};

