"""
TRINETRA — Security Item Threat Verification Engine (Phase VI)
Secondary geometric, contextual, and temporal verifier designed to suppress
handheld power tools, cordless drills, and duty-belt accessories without
degrading true firearm recall.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger("ThreatVerifier")


class ThreatVerifier:
    def __init__(
        self,
        min_aspect_ratio: float = 0.55,
        max_aspect_ratio: float = 2.40,
        max_person_dist_px: float = 180.0,
        temporal_window: int = 5,
        temporal_threshold: int = 3,
        high_conf_bypass: float = 0.75
    ):
        self.min_ar = min_aspect_ratio
        self.max_ar = max_aspect_ratio
        self.max_pdist = max_person_dist_px
        self.temporal_window = temporal_window
        self.temporal_threshold = temporal_threshold
        self.high_conf_bypass = high_conf_bypass

        # Temporal history: camera_id -> deque of detections per frame
        self._history = defaultdict(lambda: deque(maxlen=self.temporal_window))

    def _calc_aspect_ratio(self, bbox: List[float]) -> float:
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        return w / h

    def _calc_dist_to_person(self, item_bbox: List[float], person_bboxes: List[List[float]]) -> float:
        if not person_bboxes:
            return 9999.0
        ix = (item_bbox[0] + item_bbox[2]) / 2.0
        iy = (item_bbox[1] + item_bbox[3]) / 2.0

        min_d = 9999.0
        for pb in person_bboxes:
            # Distance to closest point on person box
            cx = max(pb[0], min(ix, pb[2]))
            cy = max(pb[1], min(iy, pb[3]))
            d = np.hypot(ix - cx, iy - cy)
            if d < min_d:
                min_d = d
        return min_d

    def verify_detection(
        self,
        item_det: Dict[str, Any],
        person_detections: Optional[List[Dict[str, Any]]] = None,
        use_temporal: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates candidate firearm detection against geometric, contextual,
        and temporal verification criteria.
        Fail-Safe: High confidence detections (>= 0.75) immediately pass.
        """
        bbox = item_det.get("bbox", [0, 0, 0, 0])
        conf = item_det.get("confidence", 0.0)
        camera_id = item_det.get("camera_id", "default")

        # 1. High Confidence Immediate Bypass
        if conf >= self.high_conf_bypass:
            return {
                "decision": "CONFIRMED_THREAT",
                "verified": True,
                "confidence_adjusted": conf,
                "reason": "HIGH_CONFIDENCE_BYPASS"
            }

        # 2. Geometric Shape Verification
        ar = self._calc_aspect_ratio(bbox)
        if ar < self.min_ar or ar > self.max_ar:
            # Extreme aspect ratio typical of drills with tall batteries or power cords
            return {
                "decision": "REJECTED_TOOL_DISTRACTOR",
                "verified": False,
                "confidence_adjusted": conf * 0.4,
                "reason": f"ABNORMAL_ASPECT_RATIO_{ar:.2f}"
            }

        # 3. Contextual Person Association
        person_boxes = [p["bbox"] for p in (person_detections or []) if p.get("class_name") == "person" or p.get("class_id") == 0]
        if person_boxes:
            p_dist = self._calc_dist_to_person(bbox, person_boxes)
            if p_dist > self.max_pdist:
                # Firearm far from any person in background clutter
                return {
                    "decision": "REJECTED_TOOL_DISTRACTOR",
                    "verified": False,
                    "confidence_adjusted": conf * 0.5,
                    "reason": f"NO_PERSON_PROXIMITY_DIST_{int(p_dist)}PX"
                }

        # 4. Temporal Persistence Verification
        if use_temporal:
            # Record detection in temporal history
            hist = self._history[camera_id]
            hist.append(time.time())
            recent_count = len(hist)
            if recent_count >= self.temporal_threshold:
                return {
                    "decision": "CONFIRMED_THREAT",
                    "verified": True,
                    "confidence_adjusted": conf,
                    "reason": f"TEMPORAL_CONFIRMED_{recent_count}_OF_{self.temporal_window}"
                }
            else:
                return {
                    "decision": "PENDING_TEMPORAL_CONFIRMATION",
                    "verified": True,  # Keep active in pipeline, flag for operator caution
                    "confidence_adjusted": conf,
                    "reason": f"PENDING_TEMPORAL_{recent_count}_OF_{self.temporal_threshold}"
                }

        # Default confirmed threat if geometric and contextual pass
        return {
            "decision": "CONFIRMED_THREAT",
            "verified": True,
            "confidence_adjusted": conf,
            "reason": "GEOMETRIC_CONTEXTUAL_PASS"
        }
