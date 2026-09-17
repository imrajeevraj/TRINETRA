"""
TRINETRA — Adaptive ROI Escalation Orchestrator (Phase VII)
Coordinates 4-tier inference escalation across Full-Frame, Track ROI,
Dynamic Crop, and SAHI Tiled Inspection synchronized with PTZ stabilization.
"""

from __future__ import annotations
import time
import logging
from enum import IntEnum
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from backend.app.services.sahi_engine import SAHIEngine
from backend.app.services.dynamic_roi_engine import DynamicROIEngine
from backend.app.services.ptz_cue_engine import ptz_cue_engine, PTZState, CuePriority

logger = logging.getLogger("AdaptiveROIOrchestrator")


class EscalationLevel(IntEnum):
    LEVEL_0_FULL_FRAME = 0   # Normal 24/7 continuous monitoring
    LEVEL_1_TRACK_ROI = 1    # Track-level bounding box focus
    LEVEL_2_DYNAMIC_CROP = 2 # High-res crop on suspicious/boundary trigger
    LEVEL_3_SAHI_ZOOM = 3    # Full SAHI tiled inspection on stable zoomed frame


class AdaptiveROIOrchestrator:
    def __init__(
        self,
        dynamic_roi_engine: Optional[DynamicROIEngine] = None,
        sahi_engine: Optional[SAHIEngine] = None
    ):
        self.roi_engine = dynamic_roi_engine or DynamicROIEngine()
        self.sahi_engine = sahi_engine or SAHIEngine(grid_rows=2, grid_cols=2, overlap=0.25)
        self.camera_levels: Dict[str, EscalationLevel] = {}

    def determine_escalation_level(
        self,
        camera_id: str,
        tracks: List[Dict[str, Any]],
        ptz_state: PTZState
    ) -> EscalationLevel:
        """
        Determines the appropriate inference escalation level:
        - LEVEL 3 (SAHI): High-priority threat AND PTZ is fully STABLE.
        - LEVEL 2 (Dynamic Crop): Suspicious target or PTZ settling.
        - LEVEL 1 (Track ROI): Normal active tracks.
        - LEVEL 0 (Full Frame): No active tracks or low risk.
        """
        if not tracks:
            return EscalationLevel.LEVEL_0_FULL_FRAME

        max_risk = max([int(t.get("risk_score", 0)) for t in tracks] or [0])
        has_fence_violation = any([t.get("fence_violation", False) for t in tracks])
        has_small_target = any([
            (t.get("class_name") == "person" and (t.get("bbox", [0, 0, 0, 100])[3] - t.get("bbox", [0, 0, 0, 100])[1]) < 90)
            for t in tracks
        ])

        # If camera is moving, stay at LEVEL 0 to avoid blurred computation
        if ptz_state == PTZState.MOVING:
            return EscalationLevel.LEVEL_0_FULL_FRAME

        # Level 3: PTZ is STABLE and target is high-risk / fence violation / small distant person
        if ptz_state == PTZState.STABLE and (has_fence_violation or max_risk >= 70 or (has_small_target and max_risk >= 50)):
            return EscalationLevel.LEVEL_3_SAHI_ZOOM

        # Level 2: Suspicious boundary movement or small target
        if has_fence_violation or max_risk >= 45 or has_small_target:
            return EscalationLevel.LEVEL_2_DYNAMIC_CROP

        # Level 1: Normal tracks under observation
        return EscalationLevel.LEVEL_1_TRACK_ROI

    def process_adaptive_frame(
        self,
        model: YOLO,
        frame: np.ndarray,
        camera_id: str,
        primary_detections: List[Dict[str, Any]],
        active_tracks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], EscalationLevel, float]:
        """
        Processes frame with adaptive escalation level matching current PTZ and threat state.
        Fail-Safe: Returns primary detections if escalated inference fails or times out.
        """
        t0 = time.perf_counter()
        ctrl = ptz_cue_engine.get_or_create_controller(camera_id)
        ptz_status = ctrl.get_status()
        ptz_state = PTZState(ptz_status["state"])

        level = self.determine_escalation_level(camera_id, active_tracks, ptz_state)
        self.camera_levels[camera_id] = level

        try:
            if level == EscalationLevel.LEVEL_3_SAHI_ZOOM:
                logger.info(f"[{camera_id}] Escalating to LEVEL 3 (SAHI Zoom Inspection) on STABLE frame.")
                sahi_dets, _ = self.sahi_engine.predict_sahi(model, frame, include_full_frame=True)
                dur = (time.perf_counter() - t0) * 1000.0
                return sahi_dets, level, dur

            elif level == EscalationLevel.LEVEL_2_DYNAMIC_CROP:
                logger.debug(f"[{camera_id}] Escalating to LEVEL 2 (Dynamic Crop).")
                roi_dets, _ = self.roi_engine.predict_dynamic_roi(
                    model, frame, primary_detections=primary_detections, primary_conf=0.25
                )
                dur = (time.perf_counter() - t0) * 1000.0
                return roi_dets, level, dur

            # Level 0 and Level 1 use primary detections
            dur = (time.perf_counter() - t0) * 1000.0
            return primary_detections, level, dur

        except Exception as e:
            logger.error(f"[{camera_id}] Adaptive ROI escalation error ({e}), safely falling back to Level 0.")
            dur = (time.perf_counter() - t0) * 1000.0
            return primary_detections, EscalationLevel.LEVEL_0_FULL_FRAME, dur


adaptive_roi_orchestrator = AdaptiveROIOrchestrator()
