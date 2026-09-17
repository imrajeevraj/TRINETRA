"""
TRINETRA — Hard-Case Mining Service (Phase XIII)
Discovers operational perception edge cases across 18 heuristic triggers
and scores informativeness for active learning queues.
"""

from __future__ import annotations
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("HardCaseMiner")


class HardCaseTrigger(str):
    LOW_CONFIDENCE = "TRIGGER_LOW_CONFIDENCE"
    HIGH_CONF_FP = "TRIGGER_HIGH_CONF_FP"
    REPEATED_FP = "TRIGGER_REPEATED_FP"
    OPERATOR_REJECTED = "TRIGGER_OPERATOR_REJECTED"
    SMALL_PERSON = "TRIGGER_SMALL_PERSON"
    DISTANT_PERSON = "TRIGGER_DISTANT_PERSON"
    HEAVY_OCCLUSION = "TRIGGER_HEAVY_OCCLUSION"
    SHADOW_ZONE = "TRIGGER_SHADOW_ZONE"
    NIGHT_SCENE = "TRIGGER_NIGHT_SCENE"
    ADVERSE_WEATHER = "TRIGGER_ADVERSE_WEATHER"
    UNUSUAL_VIEWPOINT = "TRIGGER_UNUSUAL_VIEWPOINT"
    CROWDED_SCENE = "TRIGGER_CROWDED_SCENE"
    UNUSUAL_VEHICLE = "TRIGGER_UNUSUAL_VEHICLE"
    AIRBORNE_CLUTTER = "TRIGGER_AIRBORNE_CLUTTER"
    TOOL_DISTRACTOR = "TRIGGER_TOOL_DISTRACTOR"
    TRACKING_JUMP = "TRIGGER_TRACKING_JUMP"
    CROSS_CAMERA_FAIL = "TRIGGER_CROSS_CAMERA_FAIL"
    PREDICTION_MISS = "TRIGGER_PREDICTION_MISS"


class HardCaseCandidate(BaseModel):
    candidate_id: str
    camera_id: str
    frame_id: str
    model_id: str
    model_version: str
    trigger_type: str
    hard_case_score: float
    confidence: float
    bounding_box: Optional[List[float]] = None  # [x, y, w, h] normalized
    scene_metadata: Dict[str, Any] = Field(default_factory=dict)
    evidence_reference: Optional[str] = None
    source_frame_hash: str
    discovered_at: float = Field(default_factory=time.time)
    review_status: str = "PENDING_REVIEW"


class HardCaseMiner:
    """
    Automated hard-case discovery engine evaluating live detections and sensor metadata.
    Output candidates are explicitly NOT labeled as ground truth.
    """

    def __init__(self):
        self.mined_cases: Dict[str, HardCaseCandidate] = {}
        self.camera_recent_fp: Dict[str, List[float]] = {}  # camera_id -> timestamps of FPs
        self._counter = 0

    def evaluate_detection(
        self,
        camera_id: str,
        frame_id: str,
        model_id: str,
        model_version: str,
        confidence: float,
        bbox: Optional[List[float]] = None,
        class_name: str = "person",
        scene_metadata: Optional[Dict[str, Any]] = None,
        operator_feedback_disp: Optional[str] = None,
        source_frame_hash: Optional[str] = None,
        distance_meters: Optional[float] = None,
        weather: Optional[str] = None,
        target_count_in_frame: int = 1,
        tracking_jump_sigma: float = 0.0,
        prediction_divergence_meters: float = 0.0,
        is_operator_fp: bool = False,
        is_operator_rejected: bool = False,
        is_tool_distractor: bool = False,
    ) -> Optional[HardCaseCandidate]:
        """
        Evaluates a frame/detection event against all 18 hard-case heuristic triggers.
        """
        now = time.time()
        meta = scene_metadata or {}
        triggers: List[Tuple[str, float]] = []

        if is_operator_fp:
            operator_feedback_disp = "FALSE_POSITIVE"
        elif is_operator_rejected and not operator_feedback_disp:
            operator_feedback_disp = "REJECTED"

        if is_tool_distractor:
            triggers.append((HardCaseTrigger.TOOL_DISTRACTOR, 0.95))

        # 1. Low confidence trigger
        if 0.25 <= confidence < 0.45:
            triggers.append((HardCaseTrigger.LOW_CONFIDENCE, 0.75))

        # 2. High confidence FP trigger
        if operator_feedback_disp == "FALSE_POSITIVE":
            if confidence > 0.80:
                triggers.append((HardCaseTrigger.HIGH_CONF_FP, 0.95))
            else:
                triggers.append((HardCaseTrigger.OPERATOR_REJECTED, 0.85))

            # Track repeated FP
            if camera_id not in self.camera_recent_fp:
                self.camera_recent_fp[camera_id] = []
            self.camera_recent_fp[camera_id].append(now)
            # Prune > 10 min
            self.camera_recent_fp[camera_id] = [t for t in self.camera_recent_fp[camera_id] if now - t <= 600]
            if len(self.camera_recent_fp[camera_id]) >= 3:
                triggers.append((HardCaseTrigger.REPEATED_FP, 0.90))

        # 3. Small person trigger
        if bbox and class_name.lower() == "person":
            _, _, w, h = bbox
            area = w * h
            if area < 0.0025:  # ~32x32 on 640x640
                triggers.append((HardCaseTrigger.SMALL_PERSON, 0.80))

        # 4. Distant person
        if distance_meters and distance_meters > 150.0 and class_name.lower() == "person":
            triggers.append((HardCaseTrigger.DISTANT_PERSON, 0.75))

        # 5. Night / Low illumination
        lux = meta.get("illumination_lux", 100.0)
        if lux < 15.0:
            triggers.append((HardCaseTrigger.NIGHT_SCENE, 0.75))

        # 6. Adverse weather
        w_state = weather or meta.get("weather", "CLEAR")
        if w_state.upper() in ["RAIN", "FOG", "DUST", "STORM"]:
            triggers.append((HardCaseTrigger.ADVERSE_WEATHER, 0.80))

        # 7. Crowded scene
        if target_count_in_frame >= 10:
            triggers.append((HardCaseTrigger.CROWDED_SCENE, 0.70))

        # 8. Tool distractor
        if class_name.lower() in ["drill", "power_tool", "grinder", "wrench"]:
            triggers.append((HardCaseTrigger.TOOL_DISTRACTOR, 0.95))

        # 9. Airborne clutter
        if class_name.lower() in ["bird", "kite", "balloon", "clutter"]:
            triggers.append((HardCaseTrigger.AIRBORNE_CLUTTER, 0.85))

        # 10. Kinematic tracking jump
        if tracking_jump_sigma > 3.5:
            triggers.append((HardCaseTrigger.TRACKING_JUMP, 0.75))

        # 11. Trajectory prediction divergence
        if prediction_divergence_meters > 25.0:
            triggers.append((HardCaseTrigger.PREDICTION_MISS, 0.70))

        if not triggers:
            return None

        # Choose highest priority trigger
        triggers.sort(key=lambda x: x[1], reverse=True)
        primary_trigger, score = triggers[0]

        self._counter += 1
        cid = f"HC-{camera_id}-{int(now * 1000)}-{self._counter:04d}"
        frame_sha = source_frame_hash or hashlib.sha256(f"{camera_id}:{frame_id}:{now}".encode("utf-8")).hexdigest().upper()

        candidate = HardCaseCandidate(
            candidate_id=cid,
            camera_id=camera_id,
            frame_id=frame_id,
            model_id=model_id,
            model_version=model_version,
            trigger_type=primary_trigger,
            hard_case_score=score,
            confidence=confidence,
            bounding_box=bbox,
            scene_metadata=meta,
            source_frame_hash=frame_sha,
            discovered_at=now,
        )

        self.mined_cases[cid] = candidate
        logger.info(f"Mined hard-case {cid} ({primary_trigger}, score={score:.2f}) from {camera_id}")

        publish_event("ai.hard_case_discovered", {
            "candidate_id": cid,
            "camera_id": camera_id,
            "trigger_type": primary_trigger,
            "score": score,
            "model_version": model_version,
        })

        return candidate

    def get_candidate(self, candidate_id: str) -> Optional[HardCaseCandidate]:
        return self.mined_cases.get(candidate_id)

    def list_candidates(self, trigger_type: Optional[str] = None, limit: int = 100) -> List[HardCaseCandidate]:
        res = list(self.mined_cases.values())
        if trigger_type:
            res = [c for c in res if c.trigger_type == trigger_type]
        res.sort(key=lambda x: x.hard_case_score, reverse=True)
        return res[:limit]

    def reset(self):
        self.mined_cases.clear()
        self.camera_recent_fp.clear()
        self._counter = 0


hard_case_miner = HardCaseMiner()
