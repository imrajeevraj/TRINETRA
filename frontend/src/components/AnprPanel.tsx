import React from 'react';
import type { PlateEvent } from '../types';
import { Car } from 'lucide-react';

interface Props {
  plates: PlateEvent[];
  dataOrigin: 'LIVE' | 'DEMO' | 'IMPORTED';
}

export const AnprPanel: React.FC<Props> = ({ plates, dataOrigin }) => {
  return (
    <div className="anpr-section">
      <div className="anpr-header">
        <h3>Vehicle Intelligence (ANPR) — {dataOrigin}</h3>
      </div>

      <div className="anpr-list">
        {plates.length === 0 ? (
          <div className="empty-state">
            <Car size={20} className="empty-state-icon" />
            <span className="empty-state-title">No Vehicles Detected</span>
            <span className="empty-state-sub">No license plates captured recently.</span>
          </div>
        ) : (
          plates.map(plate => (
            <div
              key={plate.id}
              className={`plate-card ${plate.watchlist_status === 'WATCHLIST MATCH' ? 'watchlist' : ''}`}
            >
              <div className="plate-header">
                <span className="plate-number">{plate.plate_number}</span>
                {plate.watchlist_status === 'WATCHLIST MATCH' && (
                  <span className="plate-match-badge">MATCH</span>
                )}
              </div>
              <div className="plate-meta">
                <span>{plate.camera_id} · {plate.track_id}</span>
                <span>Conf: {(plate.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="plate-time">
                {new Date(plate.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
