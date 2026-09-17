import React, { useState, useMemo } from 'react';
import type { Camera, GPSLocation } from '../types';
import Map, { Marker, Popup, NavigationControl } from 'react-map-gl/maplibre';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Shield, Radio, Video, Maximize2 } from 'lucide-react';

interface TacticalMapProps {
  cameras: Camera[];
  onSelectCamera: (camera: Camera) => void;
  bopLocation: GPSLocation | null;
}

// Tactical Border Outpost sectors in Rajasthan (matches Executive Dashboard BOP-TEST-01 to 04)
const DEFAULT_BOP_LOCATION: GPSLocation = { lat: 26.9180, lng: 70.9080 };

const SECTOR_COORDINATES: Record<string, { lat: number; lng: number; sector: string; bopId: string }> = {
  'CAM-001': { lat: 26.9124, lng: 70.9012, sector: 'Sector Alpha (North Post)', bopId: 'BOP-TEST-01' },
  'CAM-002': { lat: 26.9180, lng: 70.9080, sector: 'Sector Bravo (Central Gate)', bopId: 'BOP-TEST-02' },
  'CAM-003': { lat: 26.9240, lng: 70.9150, sector: 'Sector Charlie (East Ridge)', bopId: 'BOP-TEST-03' },
  'CAM-004': { lat: 26.9300, lng: 70.9220, sector: 'Sector Delta (River Outpost)', bopId: 'BOP-TEST-04' },
};

export const TacticalMap: React.FC<TacticalMapProps> = ({ cameras, onSelectCamera, bopLocation }) => {
  const [selectedPopupCam, setSelectedPopupCam] = useState<Camera | null>(null);
  const center = bopLocation || DEFAULT_BOP_LOCATION;

  // Enrich camera data with sector coordinates when DB has null lat/lng
  const enrichedCameras = useMemo(() => {
    return cameras.map(cam => {
      const fallback = SECTOR_COORDINATES[cam.id];
      return {
        ...cam,
        latitude: cam.latitude ?? fallback?.lat ?? (DEFAULT_BOP_LOCATION.lat + 0.004),
        longitude: cam.longitude ?? fallback?.lng ?? (DEFAULT_BOP_LOCATION.lng + 0.004),
        sectorName: fallback?.sector ?? 'Tactical Surveillance Sector',
        bopId: fallback?.bopId ?? 'BOP-HQ'
      };
    });
  }, [cameras]);

  const onlineCount = enrichedCameras.filter(c => c.status === 'ONLINE').length;

  return (
    <div className="tactical-map-container">
      {/* ── Top-Left Tactical HUD Info Overlay ───────────────────────── */}
      <div className="map-hud-overlay">
        <div className="map-hud-card glassmorphic">
          <div className="map-hud-title">
            <Radio size={14} style={{ color: 'var(--accent)' }} />
            <span>TRINETRA GEOSPATIAL SURVEILLANCE RADAR</span>
          </div>
          <div className="map-hud-meta">
            <span>Sector: <strong style={{ color: 'var(--text-primary)' }}>Rajasthan Border (BOP 1–4)</strong></span>
            <span>Sensors: <strong style={{ color: 'var(--status-healthy)' }}>{onlineCount}/{enrichedCameras.length} Online</strong></span>
            <span>Grid: <strong style={{ color: 'var(--text-muted)' }}>{center.lat.toFixed(4)}°N, {center.lng.toFixed(4)}°E</strong></span>
          </div>
        </div>
      </div>

      {/* ── Bottom-Left Tactical Legend ─────────────────────────────── */}
      <div className="map-legend-overlay">
        <div className="map-legend-item">
          <span className="map-legend-dot" style={{ background: 'var(--status-critical)', boxShadow: '0 0 6px var(--status-critical)' }} />
          <span>BOP Command Post</span>
        </div>
        <div className="map-legend-item">
          <span className="map-legend-dot" style={{ background: 'var(--accent)', boxShadow: '0 0 6px var(--accent)' }} />
          <span>Active Optical Sensor</span>
        </div>
        <div className="map-legend-item">
          <span className="map-legend-dot" style={{ background: 'var(--status-warning)' }} />
          <span>Surveillance Sector Point</span>
        </div>
      </div>

      {/* ── MapLibre GL Interactive Viewport ────────────────────────── */}
      <Map
        initialViewState={{
          longitude: center.lng,
          latitude: center.lat,
          zoom: 13.8,
          pitch: 30,
        }}
        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        mapLib={maplibregl}
        style={{ width: '100%', height: '100%' }}
        attributionControl={false}
      >
        <NavigationControl position="top-right" />

        {/* BOP Command Post Marker */}
        <Marker
          longitude={center.lng}
          latitude={center.lat}
          anchor="bottom"
        >
          <div
            className="bop-radar-marker"
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              cursor: 'pointer',
              filter: 'drop-shadow(0 0 8px rgba(239, 68, 68, 0.8))'
            }}
          >
            <div
              style={{
                background: 'linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)',
                borderRadius: '50%',
                padding: '6px',
                border: '2px solid #FFFFFF',
                boxShadow: '0 0 14px rgba(239, 68, 68, 0.9)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Shield size={16} color="#FFFFFF" />
            </div>
            <span
              style={{
                fontSize: '10px',
                fontWeight: 800,
                letterSpacing: '0.6px',
                background: 'rgba(8, 12, 20, 0.9)',
                color: '#EF4444',
                padding: '2px 6px',
                borderRadius: '4px',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                marginTop: '3px',
                whiteSpace: 'nowrap',
              }}
            >
              BOP COMMAND POST
            </span>
          </div>
        </Marker>

        {/* Camera Optical Sensor Markers */}
        {enrichedCameras.map(camera => {
          const isOnline = camera.status === 'ONLINE';
          return (
            <Marker
              key={camera.id}
              longitude={camera.longitude}
              latitude={camera.latitude}
              anchor="bottom"
              onClick={(e: any) => {
                e.originalEvent.stopPropagation();
                setSelectedPopupCam(camera);
              }}
            >
              <div
                className="camera-map-pin"
                style={{
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  transition: 'transform 0.15s ease',
                }}
              >
                <div
                  style={{
                    background: isOnline ? 'var(--surface-1)' : 'rgba(239,68,68,0.2)',
                    border: `2px solid ${isOnline ? 'var(--accent)' : 'var(--status-critical)'}`,
                    borderRadius: '50%',
                    padding: '5px',
                    boxShadow: isOnline ? '0 0 10px rgba(0, 210, 235, 0.7)' : '0 0 6px rgba(239,68,68,0.5)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Video size={14} color={isOnline ? 'var(--accent)' : 'var(--status-critical)'} />
                </div>
                <span
                  style={{
                    fontSize: '9px',
                    fontWeight: 700,
                    fontFamily: 'var(--font-mono)',
                    background: 'rgba(13, 17, 23, 0.92)',
                    color: isOnline ? 'var(--text-primary)' : 'var(--status-critical)',
                    padding: '1px 5px',
                    borderRadius: '3px',
                    border: `1px solid ${isOnline ? 'var(--border-accent)' : 'rgba(239,68,68,0.4)'}`,
                    marginTop: '2px',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {camera.id}
                </span>
              </div>
            </Marker>
          );
        })}

        {/* Interactive Camera Details Popup */}
        {selectedPopupCam && (
          <Popup
            longitude={(selectedPopupCam as any).longitude}
            latitude={(selectedPopupCam as any).latitude}
            anchor="top"
            onClose={() => setSelectedPopupCam(null)}
            closeButton={true}
            closeOnClick={false}
          >
            <div
              style={{
                color: 'var(--text-primary)',
                background: 'var(--surface-1)',
                padding: '8px 10px',
                borderRadius: '6px',
                minWidth: '200px',
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                border: '1px solid var(--border-default)',
                fontFamily: 'var(--font-sans)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: '12px' }}>
                  {selectedPopupCam.id}
                </strong>
                <span
                  style={{
                    fontSize: '9px',
                    fontWeight: 700,
                    padding: '1px 5px',
                    borderRadius: '3px',
                    background: selectedPopupCam.status === 'ONLINE' ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
                    color: selectedPopupCam.status === 'ONLINE' ? 'var(--status-healthy)' : 'var(--status-critical)',
                    border: `1px solid ${selectedPopupCam.status === 'ONLINE' ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
                  }}
                >
                  {selectedPopupCam.status}
                </span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                {selectedPopupCam.name}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                {(selectedPopupCam as any).sectorName}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', borderTop: '1px solid var(--border-subtle)', paddingTop: '4px' }}>
                <span>FPS: <strong>{selectedPopupCam.fps}</strong></span>
                <span>AI: <strong>{selectedPopupCam.inference_fps ? `${selectedPopupCam.inference_fps} fps` : 'Active'}</strong></span>
              </div>
              <button
                style={{
                  marginTop: '4px',
                  background: 'var(--accent)',
                  color: 'var(--text-inverse)',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '5px 8px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '5px',
                }}
                onClick={() => {
                  onSelectCamera(selectedPopupCam);
                  setSelectedPopupCam(null);
                }}
              >
                <Maximize2 size={12} /> Focus in Tactical Console
              </button>
            </div>
          </Popup>
        )}
      </Map>
    </div>
  );
};
