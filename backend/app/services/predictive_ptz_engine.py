"""
TRINETRA — Predictive PTZ Pre-Cue Engine (Phase IX)
Orchestrates anticipatory camera slewing, lead-time positioning, and readiness states:
PREDICT -> PRE-CUE -> WAIT -> OBSERVE -> CONFIRM
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from backend.app.services.ptz_cue_engine import ptz_cue_engine, PTZState
from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("PredictivePTZEngine")


class PreCueState(str, Enum):
    PREDICTION_CREATED = "PREDICTION_CREATED"
    PRE_CUE_REQUESTED = "PRE_CUE_REQUESTED"
    PRE_CUE_ACCEPTED = "PRE_CUE_ACCEPTED"
    PTZ_POSITIONING = "PTZ_POSITIONING"
    PTZ_READY = "PTZ_READY"
    TARGET_OBSERVED = "TARGET_OBSERVED"
    INSPECTION = "INSPECTION"
    CONFIRMED = "CONFIRMED"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    PREDICTION_EXPIRED = "PREDICTION_EXPIRED"
    PREDICTION_CANCELLED = "PREDICTION_CANCELLED"


@dataclass
class PreCueAction:
    action_id: str
    prediction_id: str
    entity_id: str
    target_camera: str
    requested_at: float = field(default_factory=time.time)
    ready_at: Optional[float] = None
    target_observed_at: Optional[float] = None
    lead_time_sec: Optional[float] = None
    state: PreCueState = PreCueState.PREDICTION_CREATED
    target_pan: float = 0.0
    target_tilt: float = 0.0
    target_zoom: float = 2.0
    reason: str = "PREDICTIVE_CUE"


class PredictivePTZEngine:
    def __init__(
        self,
        min_prediction_confidence: float = 0.65,
        pre_cue_cooldown_sec: float = 6.0
    ):
        self.min_conf = min_prediction_confidence
        self.cooldown_sec = pre_cue_cooldown_sec
        self.active_pre_cues: Dict[str, PreCueAction] = {} # camera_id -> PreCueAction
        self.last_pre_cue_time: Dict[str, float] = {}      # camera_id -> timestamp
        self._action_counter = 0

    def reset(self) -> None:
        self.active_pre_cues.clear()
        self.last_pre_cue_time.clear()
        self._action_counter = 0

    def _next_action_id(self) -> str:
        self._action_counter += 1
        return f"PRECUE-ACT-{self._action_counter:05d}"

    def evaluate_and_precue(
        self,
        prediction_id: str,
        entity_id: str,
        target_camera: str,
        confidence: float,
        eta_sec: float,
        predicted_pan: float = 0.0,
        predicted_tilt: float = -5.0,
        predicted_zoom: float = 2.0,
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str, Optional[PreCueAction]]:
        """
        Evaluates safety gates and commands predictive camera positioning.
        Returns: (accepted: bool, reason: str, action: Optional[PreCueAction])
        """
        now = timestamp if timestamp is not None else time.time()

        # Gate 1: Confidence check
        if confidence < self.min_conf:
            return False, f"Confidence {confidence:.2f} below threshold {self.min_conf}", None

        # Gate 2: Cooldown check
        last_t = self.last_pre_cue_time.get(target_camera, 0.0)
        if (now - last_t) < self.cooldown_sec:
            return False, f"Camera {target_camera} in cooldown ({now - last_t:.1f}s < {self.cooldown_sec}s)", None

        # Gate 3: Check if target camera already serving higher-priority target
        ctrl = ptz_cue_engine.get_or_create_controller(target_camera)
        status = ctrl.get_status()
        if status.get("state") == PTZState.MOVING.value and not status.get("is_predictive", False):
            return False, f"Camera {target_camera} currently executing active live target cue", None

        # Accept Pre-Cue
        act_id = self._next_action_id()
        action = PreCueAction(
            action_id=act_id,
            prediction_id=prediction_id,
            entity_id=entity_id,
            target_camera=target_camera,
            requested_at=now,
            state=PreCueState.PRE_CUE_ACCEPTED,
            target_pan=predicted_pan,
            target_tilt=predicted_tilt,
            target_zoom=predicted_zoom,
            reason="PREDICTIVE_CUE"
        )

        # Dispatch command to PTZ controller
        ok_slew = ctrl.slew_to(pan=predicted_pan, tilt=predicted_tilt, zoom=predicted_zoom, speed=1.5)
        if not ok_slew:
            return False, f"PTZ controller on {target_camera} rejected slew command", None

        action.state = PreCueState.PTZ_POSITIONING

        self.active_pre_cues[target_camera] = action
        self.last_pre_cue_time[target_camera] = now

        publish_event("ptz.predictive_cue_requested", {
            "action_id": act_id,
            "prediction_id": prediction_id,
            "entity_id": entity_id,
            "camera_id": target_camera,
            "pan": predicted_pan,
            "zoom": predicted_zoom,
            "eta_sec": eta_sec
        })

        logger.info(f"Predictive PTZ pre-cue dispatched to {target_camera} for entity {entity_id} (ETA: {eta_sec:.1f}s)")
        return True, "PRE_CUE_DISPATCHED", action

    def update_camera_state(self, camera_id: str, timestamp: Optional[float] = None) -> Optional[PreCueState]:
        """Polls controller settling state and updates action to PTZ_READY when settled."""
        action = self.active_pre_cues.get(camera_id)
        if not action:
            return None

        now = timestamp if timestamp is not None else time.time()
        ctrl = ptz_cue_engine.get_or_create_controller(camera_id)
        c_state = ctrl.update()

        if action.state == PreCueState.PTZ_POSITIONING:
            if c_state == PTZState.STABLE:
                action.state = PreCueState.PTZ_READY
                action.ready_at = now
                publish_event("ptz.predictive_cue_ready", {
                    "action_id": action.action_id,
                    "camera_id": camera_id,
                    "ready_at": now
                })
                logger.info(f"Predictive PTZ pre-cue on {camera_id} reached PTZ_READY state.")

        return action.state

    def record_target_observation(
        self,
        camera_id: str,
        entity_id: str,
        timestamp: Optional[float] = None
    ) -> Optional[float]:
        """
        Marks that the forecasted target was actually observed on camera.
        Computes pre-cue lead time (time camera was ready before target arrival).
        """
        action = self.active_pre_cues.get(camera_id)
        if not action or action.entity_id != entity_id:
            return None

        now = timestamp if timestamp is not None else time.time()
        action.target_observed_at = now
        action.state = PreCueState.TARGET_OBSERVED

        if action.ready_at is not None:
            lead_time = max(0.0, now - action.ready_at)
            action.lead_time_sec = round(lead_time, 2)
            logger.info(f"Target observed on {camera_id}! Predictive pre-cue lead time: {lead_time:.2f} seconds.")
            return action.lead_time_sec

        return 0.0

    def cancel_pre_cue(self, camera_id: str, reason: str = "MANUAL_CANCEL") -> bool:
        action = self.active_pre_cues.get(camera_id)
        if not action:
            return False
        action.state = PreCueState.PREDICTION_CANCELLED
        publish_event("ptz.predictive_cue_cancelled", {
            "action_id": action.action_id,
            "camera_id": camera_id,
            "reason": reason
        })
        del self.active_pre_cues[camera_id]
        logger.info(f"Cancelled predictive PTZ pre-cue on {camera_id}: {reason}")
        return True


predictive_ptz_engine = PredictivePTZEngine()
