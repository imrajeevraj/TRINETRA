"""
TRINETRA — PTZ Camera Resource Arbitration & Lock Engine (Phase X)
Coordinates multi-target competition, camera reservations, preemption,
anti-thrashing hysteresis, and automatic expiration for PTZ camera assets.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("PTZArbitrationEngine")


class CameraResourceState(str, Enum):
    AVAILABLE = "AVAILABLE"    # Idle or executing general sweep, available for reservation
    RESERVED = "RESERVED"      # Temporarily reserved for incoming target by predictive handover
    LOCKED = "LOCKED"          # Actively tracking an acquired live target (Level 2/3 inspection)
    RELEASING = "RELEASING"    # Returning home or settling optics


@dataclass
class CameraReservation:
    reservation_id: str
    camera_id: str
    entity_id: str
    priority_score: float
    prediction_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=time.time)
    status: str = "ACTIVE"  # ACTIVE, PREEMPTED, EXPIRED, RELEASED
    reason: str = "PREDICTIVE_HANDOVER"


class PTZArbitrationEngine:
    def __init__(self, default_reservation_ttl_sec: float = 25.0):
        self.default_ttl = default_reservation_ttl_sec
        # camera_id -> CameraResourceState
        self.camera_states: Dict[str, CameraResourceState] = {}
        # camera_id -> CameraReservation
        self.active_reservations: Dict[str, CameraReservation] = {}
        # camera_id -> locked_entity_id (if LOCKED)
        self.camera_locks: Dict[str, str] = {}
        self._res_counter = 0

    def reset(self) -> None:
        self.camera_states.clear()
        self.active_reservations.clear()
        self.camera_locks.clear()
        self._res_counter = 0

    def _next_res_id(self) -> str:
        self._res_counter += 1
        return f"RES-{self._res_counter:06d}"

    def get_camera_state(self, camera_id: str) -> CameraResourceState:
        return self.camera_states.get(camera_id, CameraResourceState.AVAILABLE)

    def calculate_priority_score(
        self,
        risk_score: int,
        is_active_incident: bool = False,
        is_restricted_zone: bool = False,
        prediction_confidence: float = 0.8,
        eta_sec: float = 15.0
    ) -> float:
        """
        Computes arbitration score [0.0 - 150.0] for target contention.
        CRITICAL (85+) > HIGH (65+) > active incident (+25) > restricted (+20) > ETA urgency
        """
        score = float(risk_score)
        if is_active_incident:
            score += 25.0
        if is_restricted_zone:
            score += 20.0
        score += round(prediction_confidence * 15.0, 1)

        # ETA urgency: closer target receives up to +10 bonus
        urgency = max(0.0, min(10.0, (30.0 - eta_sec) / 3.0))
        score += urgency

        return round(score, 1)

    def request_camera_reservation(
        self,
        camera_id: str,
        entity_id: str,
        priority_score: float,
        prediction_id: Optional[str] = None,
        duration_sec: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str, Optional[CameraReservation]]:
        """
        Attempts to reserve a camera for an incoming target.
        Handles preemption if current reservation has lower priority.
        """
        now = timestamp if timestamp is not None else time.time()
        ttl = duration_sec if duration_sec is not None else self.default_ttl
        curr_state = self.get_camera_state(camera_id)

        # 1. If camera is actively LOCKED on a live target, cannot reserve unless preemption allowed
        if curr_state == CameraResourceState.LOCKED:
            locked_entity = self.camera_locks.get(camera_id, "UNKNOWN")
            if locked_entity == entity_id:
                # Target already holds lock
                return True, "TARGET_ALREADY_HOLDING_LOCK", None
            return False, f"Camera {camera_id} is actively LOCKED tracking live entity {locked_entity}", None

        # 2. If camera is RESERVED, evaluate priority for preemption
        if curr_state == CameraResourceState.RESERVED:
            curr_res = self.active_reservations.get(camera_id)
            if curr_res and curr_res.status == "ACTIVE":
                if curr_res.entity_id == entity_id:
                    # Refresh reservation
                    curr_res.expires_at = now + ttl
                    return True, "RESERVATION_EXTENDED", curr_res

                if priority_score > (curr_res.priority_score + 10.0): # Anti-thrashing delta threshold
                    logger.info(f"Preempting reservation {curr_res.reservation_id} on {camera_id}: "
                                f"New priority {priority_score:.1f} > Current {curr_res.priority_score:.1f}")
                    curr_res.status = "PREEMPTED"
                    publish_event("camera.preempted", {
                        "camera_id": camera_id,
                        "preempted_entity": curr_res.entity_id,
                        "new_entity": entity_id,
                        "new_priority": priority_score
                    })
                else:
                    return False, f"Contention rejected: Current reservation priority {curr_res.priority_score:.1f} >= {priority_score:.1f}", None

        # 3. Grant reservation
        res_id = self._next_res_id()
        res = CameraReservation(
            reservation_id=res_id,
            camera_id=camera_id,
            entity_id=entity_id,
            priority_score=priority_score,
            prediction_id=prediction_id,
            created_at=now,
            expires_at=now + ttl,
            status="ACTIVE",
            reason="PREDICTIVE_HANDOVER"
        )
        self.active_reservations[camera_id] = res
        self.camera_states[camera_id] = CameraResourceState.RESERVED

        publish_event("camera.reserved", {
            "reservation_id": res_id,
            "camera_id": camera_id,
            "entity_id": entity_id,
            "priority": priority_score,
            "expires_at": now + ttl
        })

        logger.info(f"Reserved camera {camera_id} for entity {entity_id} (Score: {priority_score:.1f}, TTL: {ttl:.1f}s)")
        return True, "RESERVATION_GRANTED", res

    def lock_camera(self, camera_id: str, entity_id: str, timestamp: Optional[float] = None) -> bool:
        """Transitions camera to LOCKED state upon optical acquisition."""
        self.camera_states[camera_id] = CameraResourceState.LOCKED
        self.camera_locks[camera_id] = entity_id
        logger.info(f"Camera {camera_id} LOCKED onto acquired target {entity_id}.")
        return True

    def get_active_reservation(self, camera_id: str) -> Optional[CameraReservation]:
        """Returns the active reservation for the specified camera if present."""
        return self.active_reservations.get(camera_id)

    def release_camera(self, camera_id: str, reason: str = "TARGET_HANDED_OFF") -> bool:
        """Releases lock or reservation back to AVAILABLE."""
        self.camera_states[camera_id] = CameraResourceState.AVAILABLE
        self.camera_locks.pop(camera_id, None)
        if camera_id in self.active_reservations:
            self.active_reservations[camera_id].status = "RELEASED"
            del self.active_reservations[camera_id]

        publish_event("camera.released", {
            "camera_id": camera_id,
            "reason": reason
        })
        logger.info(f"Released camera {camera_id} back to AVAILABLE: {reason}")
        return True

    def expire_stale_reservations(self, current_time: Optional[float] = None) -> List[str]:
        """Automatically frees reservations where expected arrival window has lapsed."""
        now = current_time if current_time is not None else time.time()
        expired = []

        for cam_id, res in list(self.active_reservations.items()):
            if res.status == "ACTIVE" and now > res.expires_at:
                res.status = "EXPIRED"
                self.release_camera(cam_id, reason="RESERVATION_EXPIRED")
                expired.append(res.reservation_id)
                logger.info(f"Reservation {res.reservation_id} on {cam_id} expired and freed.")

        return expired


ptz_arbitration_engine = PTZArbitrationEngine()
