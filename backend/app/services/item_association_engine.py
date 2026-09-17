"""
TRINETRA — Human/Security Item Spatial Association & Temporal Confirmation Engine
Enforces:
1. Spatial association between Person Tracks (CAM-001:P-001) and Firearm detections (CAM-001:I-001)
2. Temporal confirmation (N of M recent frames) before confirming firearm presence
3. Non-presumptive threat staging:
   - Single frame -> SECURITY_ITEM_CANDIDATE
   - N of M frames confirmed -> SECURITY_ITEM_CONFIRMED
   - Never infers possession or 'PERSON ARMED' without explicit validated rule criteria
"""

import time
import logging
from typing import List, Dict, Any, Tuple
from collections import deque

logger = logging.getLogger("ItemAssociationEngine")


class ItemAssociationEngine:
    def __init__(
        self, temporal_n: int = 3, temporal_m: int = 5, proximity_margin: float = 0.15
    ):
        self.temporal_n = temporal_n
        self.temporal_m = temporal_m
        self.proximity_margin = proximity_margin

        # Rolling history per camera & track: camera_id -> item_track_id -> deque of bools
        self.temporal_history: Dict[str, Dict[str, deque]] = {}
        # Association memory: camera_id -> item_track_id -> dict
        self.active_associations: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def reset_camera(self, camera_id: str):
        self.temporal_history[camera_id] = {}
        self.active_associations[camera_id] = {}

    def compute_spatial_association(
        self, person_bbox: List[float], item_bbox: List[float]
    ) -> Tuple[bool, float]:
        """
        Determines whether an item is spatially associated with a person.
        Calculates item centroid relative to the person bounding box expanded
        by the configurable proximity margin.
        """
        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1

        # Expanded person bounding box
        exp_px1 = px1 - pw * self.proximity_margin
        exp_py1 = py1 - ph * self.proximity_margin
        exp_px2 = px2 + pw * self.proximity_margin
        exp_py2 = py2 + ph * self.proximity_margin

        ix1, iy1, ix2, iy2 = item_bbox
        icx = (ix1 + ix2) / 2.0
        icy = (iy1 + iy2) / 2.0

        # Check if centroid is within expanded person box
        inside = (exp_px1 <= icx <= exp_px2) and (exp_py1 <= icy <= exp_py2)
        if not inside:
            return False, 0.0

        # Calculate distance to person centroid normalized by diagonal
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        diag = max(1.0, (pw**2 + ph**2) ** 0.5)
        dist = ((icx - pcx) ** 2 + (icy - pcy) ** 2) ** 0.5
        norm_dist = min(1.0, dist / diag)
        assoc_conf = max(0.5, round(1.0 - norm_dist * 0.5, 3))

        return True, assoc_conf

    def associate_items_with_persons(
        self,
        camera_id: str,
        frame_id: int,
        person_tracks: List[Dict[str, Any]],
        item_tracks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Correlates item tracks with person tracks on a camera frame.
        Outputs association records.
        """
        if camera_id not in self.active_associations:
            self.active_associations[camera_id] = {}

        associations = []
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for item in item_tracks:
            i_track_id = item.get("track_id", "UNKNOWN_ITEM")
            i_bbox = item.get("bbox") or item.get("box", [0, 0, 0, 0])
            best_person = None
            best_conf = 0.0

            for person in person_tracks:
                p_track_id = person.get("track_id", "UNKNOWN_PERSON")
                p_bbox = person.get("bbox") or person.get("box", [0, 0, 0, 0])

                is_assoc, conf = self.compute_spatial_association(p_bbox, i_bbox)
                if is_assoc and conf > best_conf:
                    best_conf = conf
                    best_person = p_track_id

            if best_person is not None:
                record = {
                    "person_track_id": best_person,
                    "item_track_id": i_track_id,
                    "association_confidence": best_conf,
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "spatial_relation": "SPATIALLY_ASSOCIATED",
                }
                associations.append(record)
                self.active_associations[camera_id][i_track_id] = record

        return associations

    def update_temporal_confirmation(
        self, camera_id: str, item_track_id: str, detected: bool
    ) -> Tuple[str, int, int]:
        """
        Updates sliding window confirmation history.
        Returns: (status: 'CANDIDATE' or 'CONFIRMED', count: int, window_size: int)
        """
        if camera_id not in self.temporal_history:
            self.temporal_history[camera_id] = {}

        if item_track_id not in self.temporal_history[camera_id]:
            self.temporal_history[camera_id][item_track_id] = deque(
                maxlen=self.temporal_m
            )

        hist = self.temporal_history[camera_id][item_track_id]
        hist.append(detected)

        positive_count = sum(hist)
        status = "CONFIRMED" if positive_count >= self.temporal_n else "CANDIDATE"

        return status, positive_count, len(hist)


# Global singleton instance
item_association_engine = ItemAssociationEngine()
