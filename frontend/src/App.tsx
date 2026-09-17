import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import type { Camera, Alert, SystemStats, GPSLocation, DetailedHealth, PlateEvent } from './types';
import { MissionHeader } from './components/MissionHeader';
import { CameraDirectory } from './components/CameraDirectory';
import { SurveillanceWorkspace } from './components/SurveillanceWorkspace';
import { IntelligencePanel, DEFAULT_FILTERS } from './components/IntelligencePanel';
import type { EventFilters } from './components/IntelligencePanel';
import { OperationalStrip } from './components/OperationalStrip';
import { ExecutiveDashboard } from './components/ExecutiveDashboard';
import { TacticalMap } from './components/TacticalMap';

function App() {
  const host = (!window.location.hostname || window.location.hostname === 'localhost') ? '127.0.0.1' : window.location.hostname;
  const apiBase = import.meta.env.VITE_API_BASE_URL || `${window.location.protocol}//${host}:8000`;

  // ── Auth State ──────────────────────────────────────────
  const [authenticated, setAuthenticated] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [username, setUsername] = useState('ibvap-admin');
  const [role, setRole] = useState('');
  const [password, setPassword] = useState('Admin@4163');
  const [loginError, setLoginError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // ── Data State ──────────────────────────────────────────
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [cameraStats, setCameraStats] = useState({
    cameras_online: 0,
    total_cameras: 0,
    person_detections: 0,
    vehicle_detections: 0,
  });
  const [filters, setFilters] = useState<EventFilters>(DEFAULT_FILTERS);
  const [dataOrigin, setDataOrigin] = useState<'LIVE' | 'DEMO' | 'IMPORTED'>('LIVE');
  const [bopLocation, setBopLocation] = useState<GPSLocation | null>(null);
  const [detailedHealth, setDetailedHealth] = useState<DetailedHealth | null>(null);
  const [anprPlates, setAnprPlates] = useState<PlateEvent[]>([]);

  // ── UI State ────────────────────────────────────────────
  const [appView, setAppView] = useState<'tactical' | 'dashboard' | 'map'>('dashboard');
  const [viewMode, setViewMode] = useState<'focus' | 'grid2x2' | 'map'>('focus');
  const [time, setTime] = useState('');

  // ── Derived State ───────────────────────────────────────
  const stats: SystemStats = useMemo(() => {
    const active = alerts.filter(a => a.status === 'NEW').length;
    const critical = alerts.filter(a => a.status === 'NEW' && a.severity === 'CRITICAL').length;
    return {
      cameras_online: cameraStats.cameras_online,
      total_cameras: cameraStats.total_cameras,
      person_detections: cameraStats.person_detections,
      vehicle_detections: cameraStats.vehicle_detections,
      active_alerts: active,
      critical_alerts: critical,
    };
  }, [cameraStats, alerts]);

  const systemStatus: 'operational' | 'degraded' | 'critical' = 
    stats.critical_alerts > 0 ? 'critical' :
    stats.cameras_online < stats.total_cameras ? 'degraded' : 'operational';

  // ── Clock ───────────────────────────────────────────────
  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setTime(
        now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
        ' · ' +
        now.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' })
      );
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // ── Auth-aware Fetch Helper ──────────────────────────────
  const authFetch = useCallback((url: string, init: RequestInit = {}) => {
    const token = localStorage.getItem('trinetra_token') || localStorage.getItem('ibvap_token');
    const headers = new Headers(init.headers || {});
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    return fetch(url, {
      ...init,
      credentials: 'include',
      headers,
    });
  }, []);

  // ── Fetch Operations (declared before effects) ────────────
  const fetchCameras = useCallback(async () => {
    try {
      const response = await authFetch(`${apiBase}/api/cameras`);
      if (response.status === 401) { setAuthenticated(false); return; }
      if (!response.ok) return;
      const data: Camera[] = await response.json();
      setCameras(data);

      setSelectedCamera(prev => {
        if (prev) {
          const updated = data.find(c => c.id === prev.id);
          return updated || prev;
        }
        return data.length > 0 ? data[0] : null;
      });

      const onlineCount = data.filter(c => c.status === 'ONLINE').length;
      let persons = 0, vehicles = 0;
      data.forEach(c => {
        if (c.detections) {
          persons += new Set(c.detections.filter(d => d.class === 'person' && d.track_id).map(d => d.track_id)).size;
          vehicles += new Set(c.detections.filter(d => d.class !== 'person' && d.track_id).map(d => d.track_id)).size;
        }
      });
      setCameraStats({
        cameras_online: onlineCount,
        total_cameras: data.length,
        person_detections: persons,
        vehicle_detections: vehicles,
      });
    } catch (error) { console.error('Failed to fetch cameras:', error); }
  }, [apiBase, authFetch]);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await authFetch(`${apiBase}/api/system/health/detailed`);
      if (res.ok) setDetailedHealth(await res.json());
    } catch { /* silent */ }
  }, [apiBase, authFetch]);

  const fetchAnprPlates = useCallback(async () => {
    try {
      const res = await authFetch(`${apiBase}/api/anpr/plates?data_origin=${dataOrigin}`);
      if (res.ok) setAnprPlates(await res.json());
    } catch { /* silent */ }
  }, [apiBase, authFetch, dataOrigin]);

  const fetchEvents = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (filters.search) params.append('search', filters.search);
      if (filters.cameraId !== 'ALL') params.append('camera_id', filters.cameraId);
      if (filters.severity !== 'ALL') params.append('severity', filters.severity);
      if (filters.eventType !== 'ALL') params.append('event_type', filters.eventType);
      if (filters.status !== 'ALL') params.append('status', filters.status);
      params.append('data_origin', dataOrigin);
      if (filters.timeRange !== 'all') {
        const now = new Date();
        const mins = filters.timeRange === '1h' ? 60 : filters.timeRange === '24h' ? 1440 : 15;
        params.append('start_time', new Date(now.getTime() - mins * 60000).toISOString());
      }
      const qs = params.toString();
      const response = await authFetch(`${apiBase}/api/events/recent${qs ? `?${qs}` : ''}`);
      if (response.status === 401) { setAuthenticated(false); return; }
      if (response.ok) {
        const data = await response.json();
        setAlerts(data.filter((evt: Alert) => evt.status !== 'DISMISSED'));
      }
    } catch (error) { console.error('Failed to fetch events:', error); }
  }, [apiBase, authFetch, filters, dataOrigin]);

  // ── Stable ref for WebSocket events callback ──────────────
  const fetchEventsRef = useRef(fetchEvents);
  useEffect(() => {
    fetchEventsRef.current = fetchEvents;
  }, [fetchEvents]);

  // ── Auth Check ──────────────────────────────────────────
  useEffect(() => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    authFetch(`${apiBase}/api/auth/me`, { signal: controller.signal })
      .then(async res => {
        if (res.ok) {
          const me = await res.json();
          setUsername(me.username);
          setRole(me.role);
          setAuthenticated(true);
        } else {
          setAuthenticated(false);
        }
      })
      .catch(() => {
        setAuthenticated(false);
      })
      .finally(() => {
        clearTimeout(timeoutId);
        setAuthChecked(true);
      });

    return () => {
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, [apiBase, authFetch]);

  // ── WebSocket lifecycle (separate from filter/polling) ──────────────
  useEffect(() => {
    if (!authenticated) return;

    const token = localStorage.getItem('trinetra_token') || localStorage.getItem('ibvap_token');
    const wsUrl = apiBase.replace('http', 'ws') + '/ws/events' + (token ? `?token=${encodeURIComponent(token)}` : '');
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'NEW_ALERT' || msg.type === 'ALERT_UPDATED') {
          fetchEventsRef.current();
        }
        if (msg.event_type === 'gps_update' && msg.data?.bop_location) {
          setBopLocation(msg.data.bop_location);
        }
      } catch (e) { console.error("WS error", e); }
    };

    return () => { ws.close(); };
  }, [authenticated, apiBase]);

  // ── Camera polling (3s interval, independent of filters) ────────────
  useEffect(() => {
    if (!authenticated) return;
    // oxlint-disable-next-line react/set-state-in-effect
    fetchCameras();
    const interval = setInterval(fetchCameras, 3000);
    return () => { clearInterval(interval); };
  }, [authenticated, fetchCameras]);

  // ── Event fetching (re-fetch when filters/dataOrigin change) ────────
  useEffect(() => {
    if (!authenticated) return;
    // oxlint-disable-next-line react/set-state-in-effect
    fetchEvents();
  }, [authenticated, fetchEvents]);

  // ── Health & ANPR polling ───────────────────────────────────────────
  useEffect(() => {
    if (!authenticated) return;
    // oxlint-disable-next-line react/set-state-in-effect
    fetchHealth();
    // oxlint-disable-next-line react/set-state-in-effect
    fetchAnprPlates();
    const healthInterval = setInterval(fetchHealth, 5000);
    const anprInterval = setInterval(fetchAnprPlates, 10000);
    return () => { clearInterval(healthInterval); clearInterval(anprInterval); };
  }, [authenticated, fetchHealth, fetchAnprPlates]);

  // ── Alert Actions ───────────────────────────────────────
  const updateAlertDisposition = async (alertId: string, status: 'ACKNOWLEDGED' | 'DISMISSED' | 'ESCALATED') => {
    const response = await authFetch(`${apiBase}/api/events/${alertId.replace('evt-', '')}/disposition`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    });
    if (response.ok) setAlerts(prev => prev.map(a => a.id === alertId ? { ...a, status } : a));
  };

  // ── Auth Actions ────────────────────────────────────────
  const handleLogin = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoginError('');
    setIsLoggingIn(true);
    try {
      const response = await fetch(`${apiBase}/api/auth/login`, {
        method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        credentials: 'include', body: new URLSearchParams({ username, password }),
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        setLoginError(errData.detail || 'Invalid username or password');
        setIsLoggingIn(false);
        return;
      }
      const data = await response.json();
      if (data.access_token) {
        localStorage.setItem('trinetra_token', data.access_token);
        localStorage.setItem('ibvap_token', data.access_token);
      }
      setAuthenticated(true);
      setUsername(data.username || username);
      setRole(data.role || '');
      setPassword('');
    } catch {
      setLoginError('Backend service unreachable (is FastAPI running on :8000?)');
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleLogout = async () => {
    try { await authFetch(`${apiBase}/api/auth/logout`, { method: 'POST' }); } catch {}
    localStorage.removeItem('trinetra_token');
    localStorage.removeItem('ibvap_token');
    setAuthenticated(false); setCameras([]); setSelectedCamera(null); setAlerts([]); setUsername('ibvap-admin'); setPassword('Admin@4163'); setRole('');
  };

  const runDemoScenario = async (scenario: 'seed-intrusion' | 'seed-watchlist') => {
    const response = await fetch(`${apiBase}/api/demo/${scenario}`, { method: 'POST', credentials: 'include' });
    if (response.ok) { setDataOrigin('DEMO'); return; }
    const data = await response.json().catch(() => ({}));
    setLoginError(data.detail || 'Demo camera is not ready yet');
  };

  // ── Pre-render Guards (Sleek loading state, never blank) ─
  if (!authChecked) {
    return (
      <main className="login-screen">
        <div className="login-modal" style={{ alignItems: 'center', textAlign: 'center' }}>
          <div className="login-header">
            <div className="login-brand">TRINETRA <span>Command Center</span></div>
            <p className="login-subtitle">Threat Recognition, Intelligent Networked Eyes & Tracking for Real-time Analysis</p>
            <p className="login-tagline" style={{ fontSize: '11px', color: 'var(--accent)', marginTop: '4px', letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Three Eyes. One Secure Border.</p>
          </div>
          <div style={{ padding: 'var(--sp-6) 0', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--sp-3)' }}>
            <span className="live-dot" style={{ width: 14, height: 14 }} />
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              CONNECTING TO SECURITY GATEWAY...
            </span>
          </div>
        </div>
      </main>
    );
  }

  // ── Login Screen ────────────────────────────────────────
  if (!authenticated) {
    return (
      <main className="login-screen">
        <div className="login-modal">
          <div className="login-header">
            <div className="login-brand">TRINETRA <span>Command Center</span></div>
            <p className="login-subtitle">Threat Recognition, Intelligent Networked Eyes & Tracking for Real-time Analysis</p>
            <p className="login-tagline" style={{ fontSize: '11px', color: 'var(--accent)', marginTop: '4px', letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Three Eyes. One Secure Border.</p>
          </div>
          <form className="login-form" onSubmit={handleLogin}>
            <div className="form-group">
              <label className="form-label" htmlFor="login-username">Operator Username</label>
              <input
                className="form-input" id="login-username" type="text"
                value={username} onChange={e => setUsername(e.target.value)}
                placeholder="e.g. ibvap-admin" required autoFocus disabled={isLoggingIn}
              />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="login-password">Password</label>
              <input
                className="form-input" id="login-password" type="password"
                value={password} onChange={e => setPassword(e.target.value)}
                placeholder="Enter password" required disabled={isLoggingIn}
              />
            </div>
            {loginError && <p className="login-error" role="alert">{loginError}</p>}
            <button className="btn btn-primary login-submit" type="submit" disabled={isLoggingIn}>
              {isLoggingIn ? 'Authenticating…' : 'Sign In to Console'}
            </button>
          </form>
          <div className="login-footer">
            <button
              type="button"
              className="login-quick-hint"
              onClick={() => {
                setUsername('ibvap-admin');
                setPassword('Admin@4163');
              }}
              title="Click to fill default admin credentials"
            >
              Default Admin: <strong>ibvap-admin</strong> / <strong>Admin@4163</strong>
            </button>
            <span>Restricted Access · Authorized Defense & Surveillance Personnel Only</span>
          </div>
        </div>
      </main>
    );
  }

  // ── Main Application Shell ──────────────────────────────
  return (
    <div className="app-shell">
      <MissionHeader
        time={time}
        username={username}
        role={role}
        stats={stats}
        dataOrigin={dataOrigin}
        systemStatus={systemStatus}
        activeAppView={appView}
        onSelectAppView={setAppView}
        onLogout={handleLogout}
      />

      {appView === 'dashboard' ? (
        <ExecutiveDashboard
          cameras={cameras}
          alerts={alerts}
          stats={stats}
          apiBase={apiBase}
          dataOrigin={dataOrigin}
          bopLocation={bopLocation}
          onSelectCamera={(cam) => {
            setSelectedCamera(cam);
            setViewMode('focus');
          }}
          onSwitchToTactical={() => setAppView('tactical')}
          onSwitchToMap={() => setAppView('map')}
          onAcknowledgeAlert={(id) => void updateAlertDisposition(id, 'ACKNOWLEDGED')}
          onEscalateAlert={(id) => void updateAlertDisposition(id, 'ESCALATED')}
          detailedHealth={detailedHealth}
          anprPlates={anprPlates}
        />
      ) : appView === 'map' ? (
        <div className="workspace full-map-workspace">
          <TacticalMap
            cameras={cameras}
            bopLocation={bopLocation}
            onSelectCamera={(cam) => {
              setSelectedCamera(cam);
              setViewMode('focus');
              setAppView('tactical');
            }}
          />
        </div>
      ) : (
        <div className="workspace">
          <CameraDirectory
            cameras={cameras}
            selectedCamera={selectedCamera}
            onSelectCamera={setSelectedCamera}
          />

          <SurveillanceWorkspace
            viewMode={viewMode}
            setViewMode={setViewMode}
            selectedCamera={selectedCamera}
            cameras={cameras}
            onSelectCamera={setSelectedCamera}
            bopLocation={bopLocation}
          />

          <IntelligencePanel
            alerts={alerts}
            cameras={cameras}
            selectedCamera={selectedCamera}
            apiBase={apiBase}
            dataOrigin={dataOrigin}
            filters={filters}
            setFilters={setFilters}
            onAcknowledge={(id) => void updateAlertDisposition(id, 'ACKNOWLEDGED')}
            onDismiss={(id) => void updateAlertDisposition(id, 'DISMISSED')}
            onEscalate={(id) => void updateAlertDisposition(id, 'ESCALATED')}
            detailedHealth={detailedHealth}
            anprPlates={anprPlates}
          />
        </div>
      )}

      <OperationalStrip
        stats={stats}
        dataOrigin={dataOrigin}
        onToggleDataOrigin={() => {
          setDataOrigin(d => d === 'LIVE' ? 'DEMO' : d === 'DEMO' ? 'IMPORTED' : 'LIVE');
        }}
        onRunDemo={runDemoScenario}
      />
    </div>
  );
}

export default App;

