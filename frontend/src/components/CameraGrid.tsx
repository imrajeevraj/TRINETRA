import React, { useState } from 'react';
import type { Camera } from '../types';
import { VideoOff } from 'lucide-react';

interface CameraGridProps {
  cameras: Camera[];
  selectedCamera: Camera | null;
  onSelectCamera: (camera: Camera) => void;
  onFocusCamera: (camera: Camera) => void;
}

export const CameraGrid: React.FC<CameraGridProps> = ({
  cameras,
  selectedCamera,
  onSelectCamera,
  onFocusCamera,
}) => {
  const [retryCounts, setRetryCounts] = useState<Record<string, number>>({});
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  const token = localStorage.getItem('trinetra_token') || localStorage.getItem('ibvap_token') || sessionStorage.getItem('trinetra_token') || sessionStorage.getItem('ibvap_token') || '';
  const tokenParam = token ? `&token=${encodeURIComponent(token)}` : '';

  if (cameras.length === 0) {
    return (
      <div className="video-empty">
        <VideoOff size={36} style={{ opacity: 0.3 }} />
        <span className="empty-state-title">No Cameras Available</span>
        <span className="empty-state-sub">No active cameras in directory.</span>
      </div>
    );
  }

  const sortedCameras = [...cameras].sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  return (
    <div className="camera-grid">
      {sortedCameras.map(camera => {
        const isSelected = selectedCamera?.id === camera.id;
        const retry = retryCounts[camera.id] || 0;
        const streamUrl = `${apiBase}/api/cameras/${camera.id}/stream?k=${retry}${tokenParam}`;
        const personCount = new Set(camera.detections?.filter(d => d.class === 'person' && d.track_id).map(d => d.track_id)).size;
        const vehicleCount = new Set(camera.detections?.filter(d => d.class !== 'person' && d.track_id).map(d => d.track_id)).size;
        const statusClass = camera.status.toLowerCase();
        const aiStale = (camera.ai_result_age_seconds ?? 0) > 5;

        return (
          <div
            key={camera.id}
            className={`grid-tile ${isSelected ? 'selected' : ''}`}
            onClick={() => { onSelectCamera(camera); onFocusCamera(camera); }}
            role="button"
            tabIndex={0}
            aria-label={`${camera.name} - ${camera.status}`}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { onSelectCamera(camera); onFocusCamera(camera); } }}
          >
            {/* Top bar overlay */}
            <div className="grid-tile-bar">
              <div className="tile-label">
                <span className={`cam-status-dot ${statusClass}`} />
                <span>{camera.id}</span>
              </div>
              <div className="tile-meta">
                <span>{camera.status === 'ONLINE' ? `${camera.fps} FPS` : camera.status}</span>
              </div>
            </div>

            {/* Video */}
            <div className="grid-tile-viewer">
              {camera.status === 'ONLINE' || camera.status === 'DEGRADED' ? (
                <img
                  src={streamUrl}
                  alt={camera.name}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  loading="eager"
                  onError={() => {
                    setTimeout(() => {
                      setRetryCounts(prev => ({ ...prev, [camera.id]: (prev[camera.id] || 0) + 1 }));
                    }, 2500);
                  }}
                />
              ) : (
                <div className="grid-offline">
                  <VideoOff size={28} style={{ color: 'var(--status-offline)', opacity: 0.5 }} />
                  <span>{camera.status}</span>
                </div>
              )}
            </div>

            {/* HUD badges */}
            {(camera.status === 'ONLINE' || camera.status === 'DEGRADED') && (
              <div className="grid-tile-hud">
                <div className="hud-tag live"><span className="live-dot" />LIVE</div>
                {personCount > 0 && <div className="hud-tag" style={{ color: 'var(--status-info)' }}>P:{personCount}</div>}
                {vehicleCount > 0 && <div className="hud-tag" style={{ color: 'var(--status-info)' }}>V:{vehicleCount}</div>}
                {camera.inference_fps !== undefined && camera.inference_fps > 0 && (
                  <div className={`hud-tag ${aiStale ? 'stale' : 'ai-active'}`}>
                    {aiStale ? 'AI:STALE' : `AI:${camera.inference_fps}fps`}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
