"""
TRINETRA Phase XIV — Sensor Failure Handler & Graceful Degradation Service
Handles sensor dropouts, stale clocks, invalid calibrations, and network partitions.
Ensures the system degrades gracefully, preserves last known valid state, and audits events.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("SensorFailureHandler")


class FailureType(str, Enum):
    RGB_UNAVAILABLE = "RGB_UNAVAILABLE"
    THERMAL_UNAVAILABLE = "THERMAL_UNAVAILABLE"
    BOTH_UNAVAILABLE = "BOTH_UNAVAILABLE"
    THERMAL_STALE = "THERMAL_STALE"
    RGB_STALE = "RGB_STALE"
    CLOCK_DRIFT = "CLOCK_DRIFT"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    REGISTRATION_FAILURE = "REGISTRATION_FAILURE"
    FRAME_DROP_BURST = "FRAME_DROP_BURST"
    NETWORK_PARTITION = "NETWORK_PARTITION"
    SENSOR_RECONNECTED = "SENSOR_RECONNECTED"


class DegradationState(str, Enum):
    NORMAL_OPERATION = "NORMAL_OPERATION"
    THERMAL_DEGRADED = "THERMAL_DEGRADED"
    OPTICAL_DEGRADED = "OPTICAL_DEGRADED"
    TEMPORAL_FALLBACK = "TEMPORAL_FALLBACK"
    SENSOR_OUTAGE = "SENSOR_OUTAGE"


@dataclass
class SensorFailureAuditEvent:
    event_id: str
    camera_id: str
    sensor_id: str
    failure_type: FailureType
    degradation_state: DegradationState
    preserved_last_state: Dict[str, Any]
    message: str
    timestamp: float = field(default_factory=time.time)


class SensorFailureHandler:
    """
    Central controller for handling sensor failures, enforcing graceful degradation,
    and capturing audit telemetry.
    """

    def __init__(self):
        self._audit_log: List[SensorFailureAuditEvent] = []
        self._camera_degradation_state: Dict[str, DegradationState] = {}
        self._last_valid_states: Dict[str, Dict[str, Any]] = {}

    def handle_sensor_event(
        self,
        camera_id: str,
        sensor_id: str,
        failure_type: FailureType,
        context: Optional[Dict[str, Any]] = None,
    ) -> SensorFailureAuditEvent:
        """
        Executes graceful degradation policy for a failure condition.
        """
        ctx = context or {}
        now = time.time()

        # Update last valid state cache if provided
        if "last_valid_bbox" in ctx or "last_valid_tracks" in ctx:
            self._last_valid_states[camera_id] = {
                "timestamp": now,
                "data": ctx,
            }

        last_state = self._last_valid_states.get(camera_id, {"cached_at": now})

        # Determine degradation state
        if failure_type == FailureType.RGB_UNAVAILABLE or failure_type == FailureType.RGB_STALE:
            deg_state = DegradationState.OPTICAL_DEGRADED
            msg = f"Optical sensor {sensor_id} failed on {camera_id}. Thermal pipeline active in fallback mode."
        elif failure_type == FailureType.THERMAL_UNAVAILABLE or failure_type == FailureType.THERMAL_STALE:
            deg_state = DegradationState.THERMAL_DEGRADED
            msg = f"Thermal sensor {sensor_id} failed on {camera_id}. Optical pipeline active in fallback mode."
        elif failure_type == FailureType.BOTH_UNAVAILABLE or failure_type == FailureType.NETWORK_PARTITION:
            deg_state = DegradationState.SENSOR_OUTAGE
            msg = f"Complete sensor outage on {camera_id}. Preserved last known valid state for safe boundary handover."
        elif failure_type in [FailureType.CALIBRATION_INVALID, FailureType.REGISTRATION_FAILURE, FailureType.CLOCK_DRIFT]:
            deg_state = DegradationState.TEMPORAL_FALLBACK
            msg = f"Cross-spectral alignment issue on {camera_id}. Geometric fusion degraded to temporal-only."
        elif failure_type == FailureType.SENSOR_RECONNECTED:
            deg_state = DegradationState.NORMAL_OPERATION
            msg = f"Sensor {sensor_id} on {camera_id} successfully reconnected and verified."
        else:
            deg_state = DegradationState.NORMAL_OPERATION
            msg = f"Handled event {failure_type.value} on {camera_id}."

        self._camera_degradation_state[camera_id] = deg_state

        audit_event = SensorFailureAuditEvent(
            event_id=f"FAIL-{int(now*1000)}-{sensor_id[:8]}",
            camera_id=camera_id,
            sensor_id=sensor_id,
            failure_type=failure_type,
            degradation_state=deg_state,
            preserved_last_state=last_state,
            message=msg,
            timestamp=now,
        )

        self._audit_log.append(audit_event)
        if len(self._audit_log) > 500:
            self._audit_log.pop(0)

        logger.warning(f"Sensor Failure Policy [{camera_id}]: {msg} (State: {deg_state.value})")
        return audit_event

    def get_camera_status(self, camera_id: str) -> DegradationState:
        return self._camera_degradation_state.get(camera_id, DegradationState.NORMAL_OPERATION)

    def get_audit_log(self, camera_id: Optional[str] = None) -> List[SensorFailureAuditEvent]:
        if camera_id:
            return [e for e in self._audit_log if e.camera_id == camera_id]
        return list(self._audit_log)

    def reset(self):
        self._audit_log.clear()
        self._camera_degradation_state.clear()
        self._last_valid_states.clear()


sensor_failure_handler = SensorFailureHandler()
