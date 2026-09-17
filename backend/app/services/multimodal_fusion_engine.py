"""
TRINETRA Phase XIV — Multimodal Sensor Fusion Engine
Performs detection-level and track-level cross-spectral fusion.
Preserves explicit, transparent breakdown of:
- optical_confidence
- thermal_confidence
- fusion_confidence
- association_confidence
without collapsing into a single unexplained score.
"""

from __future__ import annotations
import math
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("MultimodalFusionEngine")


class FusionLevel(str, Enum):
    DETECTION_LEVEL = "DETECTION_LEVEL"
    TRACK_LEVEL = "TRACK_LEVEL"
    DECISION_LEVEL = "DECISION_LEVEL"


@dataclass
class FusedDetection:
    fused_id: str
    camera_id: str
    class_name: str
    bbox: List[float]                  # [x1, y1, x2, y2]
    optical_confidence: Optional[float]
    thermal_confidence: Optional[float]
    association_confidence: Optional[float]
    fusion_confidence: float
    fusion_level: FusionLevel
    is_cross_spectral: bool
    weights_used: Dict[str, float]
    sensor_ids: List[str]
    timestamp: float = field(default_factory=time.time)


class MultimodalFusionEngine:
    """
    Fuses visible optical and LWIR thermal detections with transparent confidence tracking.
    """

    def __init__(self):
        self._fused_history: List[FusedDetection] = []

    def compute_iou(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        """Computes intersection-over-union for [x1, y1, x2, y2]."""
        xa = max(bbox_a[0], bbox_b[0])
        ya = max(bbox_a[1], bbox_b[1])
        xb = min(bbox_a[2], bbox_b[2])
        yb = min(bbox_a[3], bbox_b[3])

        inter_w = max(0.0, xb - xa)
        inter_h = max(0.0, yb - ya)
        inter_area = inter_w * inter_h

        area_a = max(0.0, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
        area_b = max(0.0, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def fuse_detections(
        self,
        camera_id: str,
        optical_detections: List[Dict[str, Any]],
        thermal_detections: List[Dict[str, Any]],
        scene_lux: float = 120.0,
        registration_error_px: float = 1.0,
        optical_sensor_id: str = "SNS-CAM001-RGB",
        thermal_sensor_id: str = "SNS-CAM005-LWIR",
    ) -> List[FusedDetection]:
        """
        Fuses optical and thermal detections using adaptive environmental weighting.
        """
        results: List[FusedDetection] = []
        matched_thermal_indices = set()

        # Dynamic environmental weighting
        # At night (low lux), thermal weight increases; in bright day, optical dominates
        if scene_lux < 10.0:  # Night
            w_opt, w_thm, w_assoc = 0.20, 0.60, 0.20
        elif scene_lux < 50.0: # Dusk / Dawn
            w_opt, w_thm, w_assoc = 0.40, 0.40, 0.20
        else: # Day
            w_opt, w_thm, w_assoc = 0.60, 0.25, 0.15

        weights = {"w_opt": w_opt, "w_thm": w_thm, "w_assoc": w_assoc}

        # 1. Match optical detections with thermal
        for o_idx, opt in enumerate(optical_detections):
            opt_bbox = opt.get("bbox", [0.0, 0.0, 0.1, 0.1])
            opt_conf = opt.get("confidence", 0.85)
            opt_class = opt.get("class", "person")

            best_iou = 0.0
            best_t_idx = None

            for t_idx, thm in enumerate(thermal_detections):
                if t_idx in matched_thermal_indices:
                    continue
                thm_bbox = thm.get("bbox", [0.0, 0.0, 0.1, 0.1])
                iou = self.compute_iou(opt_bbox, thm_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_t_idx = t_idx

            if best_t_idx is not None and best_iou >= 0.25:
                # Joint Cross-Spectral Detection
                matched_thermal_indices.add(best_t_idx)
                thm = thermal_detections[best_t_idx]
                thm_conf = thm.get("confidence", 0.80)
                thm_class = thm.get("class", opt_class)

                assoc_conf = round(min(1.0, best_iou * (1.0 / max(1.0, registration_error_px))), 4)
                fus_conf = round(min(1.0, (w_opt * opt_conf) + (w_thm * thm_conf) + (w_assoc * assoc_conf)), 4)

                # Interpolate fused bbox
                fused_bbox = [
                    round(0.5 * (opt_bbox[0] + thm["bbox"][0]), 4),
                    round(0.5 * (opt_bbox[1] + thm["bbox"][1]), 4),
                    round(0.5 * (opt_bbox[2] + thm["bbox"][2]), 4),
                    round(0.5 * (opt_bbox[3] + thm["bbox"][3]), 4),
                ]

                det = FusedDetection(
                    fused_id=f"FUS-PAIR-{len(results):03d}",
                    camera_id=camera_id,
                    class_name=opt_class,
                    bbox=fused_bbox,
                    optical_confidence=opt_conf,
                    thermal_confidence=thm_conf,
                    association_confidence=assoc_conf,
                    fusion_confidence=fus_conf,
                    fusion_level=FusionLevel.DETECTION_LEVEL,
                    is_cross_spectral=True,
                    weights_used=weights,
                    sensor_ids=[optical_sensor_id, thermal_sensor_id],
                )
                results.append(det)
            else:
                # Optical only detection
                det = FusedDetection(
                    fused_id=f"FUS-OPT-{len(results):03d}",
                    camera_id=camera_id,
                    class_name=opt_class,
                    bbox=opt_bbox,
                    optical_confidence=opt_conf,
                    thermal_confidence=None,
                    association_confidence=None,
                    fusion_confidence=round(opt_conf * 0.90, 4),
                    fusion_level=FusionLevel.DETECTION_LEVEL,
                    is_cross_spectral=False,
                    weights_used=weights,
                    sensor_ids=[optical_sensor_id],
                )
                results.append(det)

        # 2. Add remaining un-matched thermal detections
        for t_idx, thm in enumerate(thermal_detections):
            if t_idx not in matched_thermal_indices:
                thm_bbox = thm.get("bbox", [0.0, 0.0, 0.1, 0.1])
                thm_conf = thm.get("confidence", 0.75)
                thm_class = thm.get("class", "person")

                det = FusedDetection(
                    fused_id=f"FUS-THM-{len(results):03d}",
                    camera_id=camera_id,
                    class_name=thm_class,
                    bbox=thm_bbox,
                    optical_confidence=None,
                    thermal_confidence=thm_conf,
                    association_confidence=None,
                    fusion_confidence=round(thm_conf * 0.85, 4),
                    fusion_level=FusionLevel.DETECTION_LEVEL,
                    is_cross_spectral=False,
                    weights_used=weights,
                    sensor_ids=[thermal_sensor_id],
                )
                results.append(det)

        self._fused_history.extend(results)
        if len(self._fused_history) > 500:
            self._fused_history = self._fused_history[-500:]

        return results

    def reset(self):
        self._fused_history.clear()


multimodal_fusion_engine = MultimodalFusionEngine()
