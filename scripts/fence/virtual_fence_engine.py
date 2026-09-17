"""
IBVAP — Virtual Fence & Geospatial Perimeter Engine
Supports Ground Fences (polygon, tripwire, restricted zone) and
Airborne Virtual Fences (sky corridors, aerial polygons, restricted air volume).
Computes line-crossing vectors, polygon containment, dwell-time alerts, and event de-duplication.
"""

import time
import math
from typing import List, Dict, Any, Tuple, Optional

def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """Ray casting algorithm for 2D polygon inclusion test."""
    x, y = point[0], point[1]
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def line_segments_intersect(p1: List[float], p2: List[float], q1: List[float], q2: List[float]) -> bool:
    """Check if segment (p1, p2) intersects with segment (q1, q2)."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

class VirtualFenceZone:
    def __init__(
        self,
        zone_id: str,
        name: str,
        fence_type: str, # "polygon", "tripwire", "sky_corridor", "restricted_airspace"
        domain: str,     # "ground", "airborne", "hybrid"
        coordinates: List[List[float]],
        allowed_classes: Optional[List[str]] = None,
        dwell_time_threshold_sec: float = 3.0
    ):
        self.zone_id = zone_id
        self.name = name
        self.fence_type = fence_type
        self.domain = domain
        self.coordinates = coordinates
        self.allowed_classes = allowed_classes or []
        self.dwell_time_threshold_sec = dwell_time_threshold_sec

class VirtualFenceEngine:
    def __init__(self):
        self.zones: Dict[str, VirtualFenceZone] = {}
        # Track state tracking: {track_id: {zone_id: {"first_seen": t, "last_seen": t, "inside": bool, "alerted_entry": bool, "alerted_dwell": bool}}}
        self.track_states: Dict[int, Dict[str, Dict[str, Any]]] = {}

    def add_zone(self, zone: VirtualFenceZone):
        self.zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str):
        if zone_id in self.zones:
            del self.zones[zone_id]

    def process_tracks(
        self,
        tracked_objects: List[Dict[str, Any]],
        camera_id: str = "CAM-01",
        detector_version_map: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Processes active tracked detections against all virtual fences.
        Returns:
          (annotated_detections_with_schema, triggered_events)
        """
        now = time.time()
        det_map = detector_version_map or {"ground": "v2.0", "airborne": "v1.1"}
        events = []
        annotated_detections = []

        current_active_track_ids = set(d["track_id"] for d in tracked_objects)
        # Purge stale tracks older than 60s
        for tid in list(self.track_states.keys()):
            if tid not in current_active_track_ids:
                # If all zones for this track haven't been seen recently, clean up
                max_last_seen = max((z["last_seen"] for z in self.track_states[tid].values()), default=0)
                if now - max_last_seen > 30.0:
                    del self.track_states[tid]

        for det in tracked_objects:
            track_id = det["track_id"]
            cname = det.get("class_name", "unknown")
            cid = det.get("class_id", 0)
            det_id = det.get("detector_id", "ground")
            centroid = det.get("centroid", [(det["bbox"][0] + det["bbox"][2])/2.0, (det["bbox"][1] + det["bbox"][3])/2.0])
            history = det.get("centroid_history", [centroid])
            
            if track_id not in self.track_states:
                self.track_states[track_id] = {}

            overall_fence_status = "CLEAR"

            for zone_id, zone in self.zones.items():
                if zone_id not in self.track_states[track_id]:
                    self.track_states[track_id][zone_id] = {
                        "first_seen": now,
                        "last_seen": now,
                        "inside": False,
                        "alerted_entry": False,
                        "alerted_dwell": False,
                        "crossed": False
                    }

                state = self.track_states[track_id][zone_id]
                state["last_seen"] = now

                # 1. Polygon / Sky Corridor Containment
                if zone.fence_type in ["polygon", "sky_corridor", "restricted_airspace"]:
                    is_inside = point_in_polygon(centroid, zone.coordinates)
                    was_inside = state["inside"]
                    state["inside"] = is_inside

                    if is_inside:
                        overall_fence_status = f"IN_ZONE_{zone_id}"
                        # Check entry event
                        if not was_inside and not state["alerted_entry"]:
                            state["first_seen"] = now
                            state["alerted_entry"] = True
                            event_type = f"{cname}_entering"
                            events.append(self._build_event(
                                event_type=event_type,
                                severity="HIGH" if cname in ["drone", "person"] else "MEDIUM",
                                zone=zone,
                                track_id=track_id,
                                cname=cname,
                                cid=cid,
                                det=det,
                                camera_id=camera_id,
                                det_id=det_id,
                                det_version=det_map.get(det_id, "v1.0")
                            ))
                        # Check dwell time event
                        dwell_time = now - state["first_seen"]
                        if dwell_time >= zone.dwell_time_threshold_sec and not state["alerted_dwell"]:
                            state["alerted_dwell"] = True
                            events.append(self._build_event(
                                event_type=f"{cname}_dwelling",
                                severity="CRITICAL",
                                zone=zone,
                                track_id=track_id,
                                cname=cname,
                                cid=cid,
                                det=det,
                                camera_id=camera_id,
                                det_id=det_id,
                                det_version=det_map.get(det_id, "v1.0"),
                                extra_meta={"dwell_time_sec": round(dwell_time, 1)}
                            ))
                    else:
                        # Left the zone
                        state["alerted_entry"] = False
                        state["alerted_dwell"] = False

                # 2. Tripwire / Line Crossing
                elif zone.fence_type == "tripwire":
                    if len(zone.coordinates) >= 2 and len(history) >= 2:
                        p1, p2 = history[-2], history[-1]
                        q1, q2 = zone.coordinates[0], zone.coordinates[1]
                        if line_segments_intersect(p1, p2, q1, q2) and not state["crossed"]:
                            state["crossed"] = True
                            overall_fence_status = f"CROSSED_{zone_id}"
                            events.append(self._build_event(
                                event_type=f"{cname}_crossing",
                                severity="CRITICAL",
                                zone=zone,
                                track_id=track_id,
                                cname=cname,
                                cid=cid,
                                det=det,
                                camera_id=camera_id,
                                det_id=det_id,
                                det_version=det_map.get(det_id, "v1.0")
                            ))

            # Conform strictly to standardized schema (Phase 10)
            annotated_det = {
                "object_id": f"trk_{det_id[:3]}_{track_id:04d}",
                "track_id": track_id,
                "class_id": cid,
                "class_name": cname,
                "confidence": round(float(det.get("confidence", 0.0)), 3),
                "bbox": [round(float(v), 1) for v in det["bbox"]],
                "centroid": centroid,
                "velocity": det.get("velocity", [0.0, 0.0]),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "camera_id": camera_id,
                "detector_id": det_id,
                "detector_version": det_map.get(det_id, "v1.0"),
                "fence_status": overall_fence_status
            }
            annotated_detections.append(annotated_det)

        return annotated_detections, events

    def _build_event(
        self,
        event_type: str,
        severity: str,
        zone: VirtualFenceZone,
        track_id: int,
        cname: str,
        cid: int,
        det: Dict[str, Any],
        camera_id: str,
        det_id: str,
        det_version: str,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        evt = {
            "event_id": f"evt_{int(time.time()*1000)}_{track_id}",
            "event_type": event_type,
            "severity": severity,
            "zone_id": zone.zone_id,
            "zone_name": zone.name,
            "domain": zone.domain,
            "track_id": track_id,
            "object_id": f"trk_{det_id[:3]}_{track_id:04d}",
            "class_name": cname,
            "class_id": cid,
            "detector_id": det_id,
            "detector_version": det_version,
            "camera_id": camera_id,
            "bbox": det["bbox"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        if extra_meta:
            evt.update(extra_meta)
        return evt
