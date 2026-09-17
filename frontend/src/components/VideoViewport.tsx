import React, { useState } from 'react';
import type { Camera } from '../types';
import { VideoOff, RefreshCw } from 'lucide-react';

interface VideoViewportProps {
  camera: Camera | null;
}

export const VideoViewport: React.FC<VideoViewportProps> = ({ camera }) => {
  const [key, setKey] = useState(0);
  const [streamMode, setStreamMode] = useState<'live' | 'webrtc'>('live');

  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  const webrtcHost = window.location.hostname;

  if (!camera) {
    return (
      <div className="video-viewport">
        <div className="video-empty">
          <VideoOff size={36} style={{ opacity: 0.3 }} />
          <span className="empty-state-title">No Camera Selected</span>
          <span className="empty-state-sub">Select a camera from the directory to start monitoring.</span>
        </div>
      </div>
    );
  }

  const webrtcUrl = `http://${webrtcHost}:8889/${camera.id}/`;
  const token = localStorage.getItem('trinetra_token') || localStorage.getItem('ibvap_token') || sessionStorage.getItem('trinetra_token') || sessionStorage.getItem('ibvap_token') || '';
  const tokenParam = token ? `&token=${encodeURIComponent(token)}` : '';
  const mjpegUrl = `${apiBase}/api/cameras/${camera.id}/stream?k=${key}${tokenParam}`;
  const isStreaming = camera.status === 'ONLINE' || camera.status === 'DEGRADED';

  const personCount = camera.detections
    ? new Set(camera.detections.filter(d => d.class === 'person' && d.track_id).map(d => d.track_id)).size
    : 0;
  const vehicleCount = camera.detections
    ? new Set(camera.detections.filter(d => d.class !== 'person' && d.track_id).map(d => d.track_id)).size
    : 0;

  const aiStale = (camera.ai_result_age_seconds ?? 0) > 5;

  return (
    <div className="video-viewport">
      {isStreaming ? (
        streamMode === 'webrtc' ? (
          <iframe
            key={key}
            src={webrtcUrl}
            title={`${camera.name} WebRTC Stream`}
            className="video-frame"
            style={{ border: 'none', width: '100%', height: '100%' }}
            allow="autoplay; fullscreen"
          />
        ) : (
          <img
            key={key}
            src={mjpegUrl}
            alt={camera.name}
            className="video-frame"
            loading="eager"
          />
        )
      ) : (
        <div className="video-offline">
          <VideoOff size={40} className="video-offline-icon" />
          <span className="video-offline-title">Camera {camera.status}</span>
          <span className="video-offline-sub">{camera.location || 'Location unavailable'}</span>
          <button className="btn" onClick={() => setKey(k => k + 1)}>
            <RefreshCw size={12} /> Reconnect
          </button>
        </div>
      )}

      {/* HUD Overlay */}
      {isStreaming && (
        <div className="video-hud">
          {/* Top-Left: LIVE + Camera ID */}
          <div className="hud-corner hud-top-left">
            <div className="hud-tag live">
              <span className="live-dot" />
              LIVE
            </div>
            <div className="hud-tag">
              {camera.id}
            </div>
          </div>

          {/* Top-Right: Resolution + FPS + AI */}
          <div className="hud-corner hud-top-right">
            <div className="hud-tag">{camera.resolution || '1920×1080'}</div>
            <div className="hud-tag">{camera.fps} FPS</div>
            <div className={`hud-tag ${aiStale ? 'stale' : 'ai-active'}`}>
              {aiStale
                ? 'AI: STALE'
                : `AI: ${camera.inference_fps || 0} FPS · ${camera.inference_latency_ms || 0}ms`}
            </div>
          </div>

          {/* Bottom-Left: Detection Counts */}
          <div className="hud-corner hud-bottom-left hud-detection-counts">
            {personCount > 0 && (
              <div className="hud-tag" style={{ color: 'var(--status-info)' }}>
                PERSONS {String(personCount).padStart(2, '0')}
              </div>
            )}
            {vehicleCount > 0 && (
              <div className="hud-tag" style={{ color: 'var(--status-info)' }}>
                VEHICLES {String(vehicleCount).padStart(2, '0')}
              </div>
            )}
          </div>

          {/* Bottom-Right: Stream mode toggle + Reconnect */}
          <div className="hud-corner hud-bottom-right" style={{ pointerEvents: 'auto' }}>
            <button
              className="btn btn-sm"
              onClick={() => setStreamMode(m => m === 'live' ? 'webrtc' : 'live')}
              title="Switch streaming protocol"
            >
              {streamMode === 'live' ? '⚡ MJPEG' : '🌐 WebRTC'}
            </button>
            <button className="btn btn-sm" onClick={() => setKey(k => k + 1)} title="Reconnect">
              <RefreshCw size={11} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
