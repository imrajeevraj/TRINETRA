export interface Camera {
  id: string;
  name: string;
  source?: string;
  location?: string;
  latitude?: number;
  longitude?: number;
  resolution?: string;
  fps: number;
  is_active: boolean;
  status: 'ONLINE' | 'OFFLINE' | 'DEGRADED' | 'NO_SIGNAL';
  last_seen?: string;
  detections?: Detection[];
  inference_fps?: number;
  inference_latency_ms?: number;
  ai_result_age_seconds?: number | null;
}

export interface GPSLocation {
  lat: number;
  lng: number;
}

export interface Detection {
  class: string;
  confidence: number;
  box: [number, number, number, number];
  track_id?: string;
}

export interface Alert {
  id: string;
  event_type: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  camera_id: string;
  location?: string;
  timestamp: string;
  track_id?: string;
  zone_id?: string;
  object_type?: string;
  direction?: string;
  risk_score: number;
  risk_reasons: string[];
  reasons_summary?: string;
  x?: number;
  y?: number;
  snapshot_path?: string;
  video_clip_path?: string;
  status: 'NEW' | 'ACKNOWLEDGED' | 'DISMISSED' | 'ESCALATED';
  operator_notes?: string;
  handled_by?: string;
  handled_at?: string;
  snapshot_available?: boolean;
  video_clip_available?: boolean;
  snapshot_url?: string;
  video_clip_url?: string;
  data_origin: 'LIVE' | 'DEMO' | 'TEST' | 'IMPORTED';
}

export interface SystemStats {
  cameras_online: number;
  total_cameras: number;
  person_detections: number;
  vehicle_detections: number;
  active_alerts: number;
  critical_alerts: number;
}

export interface PlateEvent {
  id: number;
  camera_id: string;
  track_id: string;
  plate_number: string;
  confidence: number;
  watchlist_status: 'CLEAN' | 'WATCHLIST MATCH' | 'NEEDS_VERIFICATION';
  timestamp: string;
  data_origin: 'LIVE' | 'DEMO' | 'TEST' | 'IMPORTED';
}

export interface DetailedHealth {
  status: string;
  cpu: { percent: number };
  memory: { percent: number; used_mb: number; total_mb: number };
  disk: { percent: number; free_gb: number };
  gpu?: { available: boolean; percent?: number; memory_percent?: number; memory_used_mb?: number; memory_total_mb?: number; name?: string; };
  database_status: string;
  cameras: { online: number; total: number; inference_fps: number; stale: string[] };
  stage_profiling_ms?: {
    avg_inference_ms: number;
    avg_tracking_ms: number;
    avg_zone_ms: number;
    avg_anpr_ms: number;
    avg_total_pipeline_ms: number;
  };
  ground_ai?: { status: string; fps: number; persons: number; vehicles: number };
  air_ai?: { status: string; fps: number; drones: number; aircraft: number };
  security_item_ai?: { status: string; fps: number; firearms: number; model: string; version: string };
  virtual_fence?: { ground_zones: number; air_zones: number; tripwires: number };
  anpr?: { queue_depth: number };
}
