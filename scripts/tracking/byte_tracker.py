"""
IBVAP — ByteTrack-inspired Multi-Object Tracker (MOT)
Maintains stable track IDs across video frames for:
  - person
  - vehicle
  - drone
  - aircraft
Implements stateful track lifecycle, velocity smoothing, and IoU association.
"""

import numpy as np
from typing import List, Dict, Any, Optional

def calc_box_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area <= 0.0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / float(union) if union > 0 else 0.0

class Track:
    def __init__(self, track_id: int, bbox: List[float], score: float, class_id: int, class_name: str, detector_id: str):
        self.track_id = track_id
        self.bbox = [float(v) for v in bbox]
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.detector_id = str(detector_id)
        
        self.history = [self.bbox]
        self.centroid_history = [self.get_centroid()]
        self.velocity = (0.0, 0.0)
        
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = "tentative"  # "tentative", "confirmed", "lost"
        self.confirmed = False

    def get_centroid(self) -> List[float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        return [cx, cy]

    def update(self, bbox: List[float], score: float):
        old_cx, old_cy = self.get_centroid()
        self.bbox = [float(v) for v in bbox]
        self.score = float(score)
        self.history.append(self.bbox)
        if len(self.history) > 30:
            self.history.pop(0)
            
        new_cx, new_cy = self.get_centroid()
        self.centroid_history.append([new_cx, new_cy])
        if len(self.centroid_history) > 30:
            self.centroid_history.pop(0)

        # Exponential moving average velocity
        vx = new_cx - old_cx
        vy = new_cy - old_cy
        alpha = 0.6
        self.velocity = (alpha * vx + (1 - alpha) * self.velocity[0],
                         alpha * vy + (1 - alpha) * self.velocity[1])

        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 2:
            self.state = "confirmed"
            self.confirmed = True

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 2:
            self.state = "lost"

class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.40,
        low_thresh: float = 0.15,
        match_thresh: float = 0.50,
        max_time_lost: int = 15
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes raw detections matching the common schema:
          [{'bbox': [x1, y1, x2, y2], 'confidence': 0.85, 'class_id': 0, 'class_name': 'person', 'detector_id': 'ground'}, ...]
        Returns tracked detections with assigned 'track_id', 'velocity', and 'centroid'.
        """
        # Split detections by confidence threshold (ByteTrack principle)
        high_dets = [d for d in detections if d.get("confidence", 0.0) >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.get("confidence", 0.0) < self.high_thresh]

        unmatched_tracks = list(self.tracks)
        matched_tracks = []

        # Association Step 1: Match high-confidence detections with existing tracks
        unmatched_high_dets = []
        if unmatched_tracks and high_dets:
            cost_matrix = np.zeros((len(unmatched_tracks), len(high_dets)), dtype=float)
            for t_idx, trk in enumerate(unmatched_tracks):
                for d_idx, det in enumerate(high_dets):
                    cost_matrix[t_idx, d_idx] = calc_box_iou(trk.bbox, det["bbox"])

            matched_t = set()
            matched_d = set()
            while True:
                if len(matched_t) == len(unmatched_tracks) or len(matched_d) == len(high_dets):
                    break
                # Find maximum IoU
                max_val = -1.0
                best_t, best_d = -1, -1
                for t in range(len(unmatched_tracks)):
                    if t in matched_t: continue
                    for d in range(len(high_dets)):
                        if d in matched_d: continue
                        if cost_matrix[t, d] > max_val:
                            max_val = cost_matrix[t, d]
                            best_t, best_d = t, d
                if max_val >= self.match_thresh:
                    matched_t.add(best_t)
                    matched_d.add(best_d)
                    trk = unmatched_tracks[best_t]
                    det = high_dets[best_d]
                    trk.update(det["bbox"], det["confidence"])
                    matched_tracks.append(trk)
                else:
                    break

            unmatched_tracks = [t for idx, t in enumerate(unmatched_tracks) if idx not in matched_t]
            unmatched_high_dets = [d for idx, d in enumerate(high_dets) if idx not in matched_d]
        else:
            unmatched_high_dets = list(high_dets)

        # Association Step 2: Match low-confidence detections with remaining tracks
        if unmatched_tracks and low_dets:
            cost_matrix = np.zeros((len(unmatched_tracks), len(low_dets)), dtype=float)
            for t_idx, trk in enumerate(unmatched_tracks):
                for d_idx, det in enumerate(low_dets):
                    cost_matrix[t_idx, d_idx] = calc_box_iou(trk.bbox, det["bbox"])

            matched_t = set()
            matched_d = set()
            while True:
                if len(matched_t) == len(unmatched_tracks) or len(matched_d) == len(low_dets):
                    break
                max_val = -1.0
                best_t, best_d = -1, -1
                for t in range(len(unmatched_tracks)):
                    if t in matched_t: continue
                    for d in range(len(low_dets)):
                        if d in matched_d: continue
                        if cost_matrix[t, d] > max_val:
                            max_val = cost_matrix[t, d]
                            best_t, best_d = t, d
                if max_val >= 0.40: # slightly lower threshold for second association
                    matched_t.add(best_t)
                    matched_d.add(best_d)
                    trk = unmatched_tracks[best_t]
                    det = low_dets[best_d]
                    trk.update(det["bbox"], det["confidence"])
                    matched_tracks.append(trk)
                else:
                    break

            unmatched_tracks = [t for idx, t in enumerate(unmatched_tracks) if idx not in matched_t]

        # Step 3: Handle unmatched tracks (mark missed)
        for trk in unmatched_tracks:
            trk.mark_missed()

        # Step 4: Initialize new tracks for unmatched high-confidence detections
        for det in unmatched_high_dets:
            new_trk = Track(
                track_id=self._next_id,
                bbox=det["bbox"],
                score=det["confidence"],
                class_id=det.get("class_id", 0),
                class_name=det.get("class_name", "object"),
                detector_id=det.get("detector_id", "unknown")
            )
            self._next_id += 1
            self.tracks.append(new_trk)
            matched_tracks.append(new_trk)

        # Clean up dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_time_lost]

        # Package active confirmed tracks
        active_tracked_detections = []
        for trk in self.tracks:
            if trk.time_since_update == 0 and trk.state == "confirmed":
                cx, cy = trk.get_centroid()
                active_tracked_detections.append({
                    "track_id": trk.track_id,
                    "bbox": trk.bbox,
                    "confidence": trk.score,
                    "class_id": trk.class_id,
                    "class_name": trk.class_name,
                    "detector_id": trk.detector_id,
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "velocity": [round(trk.velocity[0], 2), round(trk.velocity[1], 2)],
                    "centroid_history": trk.centroid_history[-10:]
                })

        return active_tracked_detections
