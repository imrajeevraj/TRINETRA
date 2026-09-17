import React, { useState } from 'react';
import type { Camera } from '../types';
import { Search, Video } from 'lucide-react';

interface CameraDirectoryProps {
  cameras: Camera[];
  selectedCamera: Camera | null;
  onSelectCamera: (camera: Camera) => void;
}

type CameraFilter = 'ALL' | 'ONLINE' | 'WARNING' | 'OFFLINE';

export const CameraDirectory: React.FC<CameraDirectoryProps> = ({
  cameras,
  selectedCamera,
  onSelectCamera,
}) => {
  const [filter, setFilter] = useState<CameraFilter>('ALL');
  const [search, setSearch] = useState('');

  const sortedCameras = [...cameras].sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  const filtered = sortedCameras.filter(cam => {
    if (search) {
      const q = search.toLowerCase();
      if (!cam.name.toLowerCase().includes(q) && !cam.id.toLowerCase().includes(q)) return false;
    }
    switch (filter) {
      case 'ONLINE': return cam.status === 'ONLINE';
      case 'WARNING': return cam.status === 'DEGRADED';
      case 'OFFLINE': return cam.status === 'OFFLINE' || cam.status === 'NO_SIGNAL';
      default: return true;
    }
  });

  const onlineCount = cameras.filter(c => c.status === 'ONLINE').length;

  return (
    <aside className="camera-directory" aria-label="Camera Directory">
      {/* Header */}
      <div className="directory-header">
        <div className="directory-title">
          <span className="section-label">Cameras</span>
          <span className="camera-count">{onlineCount}/{cameras.length}</span>
        </div>
        <div className="camera-search">
          <Search size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          <input
            type="text"
            placeholder="Search cameras..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            aria-label="Search cameras"
          />
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="camera-filters" role="tablist">
        {(['ALL', 'ONLINE', 'WARNING', 'OFFLINE'] as CameraFilter[]).map(f => (
          <button
            key={f}
            className={`cam-filter-btn ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
            role="tab"
            aria-selected={filter === f}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Camera List */}
      <div className="camera-list" role="listbox" aria-label="Camera list">
        {filtered.map(cam => {
          const statusClass = cam.status.toLowerCase().replace('no_signal', 'offline');
          return (
            <div
              key={cam.id}
              className={`camera-row ${selectedCamera?.id === cam.id ? 'selected' : ''}`}
              onClick={() => onSelectCamera(cam)}
              role="option"
              aria-selected={selectedCamera?.id === cam.id}
              tabIndex={0}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') onSelectCamera(cam); }}
            >
              <span className={`cam-status-dot ${statusClass}`} aria-hidden="true" />
              <div className="cam-info">
                <div className="cam-id">{cam.id}</div>
                <div className="cam-name">{cam.name}</div>
              </div>
              <div className="cam-meta">
                <div className={`cam-status-label ${statusClass}`}>{cam.status === 'NO_SIGNAL' ? 'OFFLINE' : cam.status}</div>
                <div className="cam-fps">{cam.status === 'ONLINE' ? `${cam.fps} FPS` : ''}</div>
              </div>
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div className="empty-state">
            <Video size={20} className="empty-state-icon" />
            <span className="empty-state-sub">No cameras match filter</span>
          </div>
        )}
      </div>
    </aside>
  );
};
